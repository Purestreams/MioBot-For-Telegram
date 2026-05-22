"""FastAPI application for MioBot web administration."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Optional

from app.runtime_config import bootstrap_runtime_environment

bootstrap_runtime_environment()

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.database import (
    archive_global_memory_fact,
    archive_user_memory_fact,
    consume_webadmin_login_token,
    get_global_memory_facts,
    get_latest_display_name_for_user,
    get_user_memory,
    get_user_memory_facts,
    get_webadmin_dashboard_stats,
    init_db,
    list_global_memory_chat_overviews,
    list_user_memory_candidates,
    list_user_memory_overviews,
    list_webadmin_chat_messages,
    update_global_memory_fact,
    update_user_memory_fact,
    upsert_global_memory_facts,
    upsert_user_memory,
)
from app.user_memory import accept_user_memory_candidate, reject_user_memory_candidate
from webadmin.schemas import (
    GlobalFactCreateRequest,
    GlobalFactUpdateRequest,
    MemorySummaryUpdateRequest,
    TokenLoginRequest,
    UserFactUpdateRequest,
)
from webadmin.security import (
    SESSION_COOKIE_NAME,
    create_session_cookie,
    hash_login_token,
    parse_session_cookie,
    session_ttl_seconds,
    webadmin_cookie_secure,
    webadmin_host,
    webadmin_port,
)

PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
INDEX_FILE = STATIC_DIR / "index.html"


def _to_dict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, list):
        return [_to_dict(item) for item in value]
    if isinstance(value, tuple):
        return [_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_dict(item) for key, item in value.items()}
    return value


def _bounded_limit(value: int, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, maximum))


def require_session(request: Request) -> dict[str, Any]:
    session = parse_session_cookie(request.cookies.get(SESSION_COOKIE_NAME, ""))
    if not session:
        raise HTTPException(status_code=401, detail="authentication_required")
    return session


def create_app() -> FastAPI:
    init_db()
    application = FastAPI(title="MioBot Web Admin", docs_url=None, redoc_url=None)
    application.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @application.get("/")
    async def index() -> FileResponse:
        return FileResponse(INDEX_FILE)

    @application.get("/api/me")
    async def me(request: Request) -> dict[str, Any]:
        session = parse_session_cookie(request.cookies.get(SESSION_COOKIE_NAME, ""))
        if not session:
            return {"authenticated": False}
        return {"authenticated": True, "admin": session}

    @application.post("/api/auth/login")
    async def login(payload: TokenLoginRequest, response: Response) -> dict[str, Any]:
        token_row = await consume_webadmin_login_token(hash_login_token(payload.token))
        if not token_row:
            raise HTTPException(status_code=401, detail="invalid_or_expired_token")

        ttl_seconds = session_ttl_seconds()
        session_cookie = create_session_cookie(
            admin_user_id=token_row.admin_user_id,
            admin_username=token_row.admin_username,
            ttl_seconds=ttl_seconds,
        )
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_cookie,
            max_age=ttl_seconds,
            httponly=True,
            secure=webadmin_cookie_secure(),
            samesite="lax",
            path="/",
        )
        return {
            "authenticated": True,
            "admin": {
                "admin_user_id": token_row.admin_user_id,
                "admin_username": token_row.admin_username,
            },
        }

    @application.post("/api/auth/logout")
    async def logout(response: Response) -> dict[str, bool]:
        response.delete_cookie(SESSION_COOKIE_NAME, path="/")
        return {"ok": True}

    @application.get("/api/dashboard")
    async def dashboard(_session: dict[str, Any] = Depends(require_session)) -> dict[str, Any]:
        return _to_dict(await get_webadmin_dashboard_stats())

    @application.get("/api/chats")
    async def chats(
        limit: int = Query(80, ge=1, le=500),
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        rows = await list_global_memory_chat_overviews(limit=_bounded_limit(limit, default=80, maximum=500))
        return {"chats": _to_dict(rows)}

    @application.get("/api/chats/{chat_id}/messages")
    async def chat_messages(
        chat_id: int,
        limit: int = Query(100, ge=1, le=500),
        before_id: Optional[int] = Query(default=None, ge=1),
        q: str = "",
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        rows = await list_webadmin_chat_messages(
            chat_id,
            limit=_bounded_limit(limit, default=100, maximum=500),
            before_id=before_id,
            search=q,
        )
        return {"messages": _to_dict(rows)}

    @application.get("/api/memory/users")
    async def memory_users(
        limit: int = Query(120, ge=1, le=500),
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        rows = await list_user_memory_overviews(limit=_bounded_limit(limit, default=120, maximum=500))
        return {"users": _to_dict(rows)}

    @application.get("/api/memory/users/{telegram_user_key}")
    async def memory_user_detail(
        telegram_user_key: str,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        current = await get_user_memory(telegram_user_key)
        facts = await get_user_memory_facts(telegram_user_key, limit=100, min_confidence=0.0)
        candidates = await list_user_memory_candidates(telegram_user_key, status="pending", limit=100)
        return {
            "memory": _to_dict(current),
            "facts": _to_dict(facts),
            "candidates": _to_dict(candidates),
        }

    @application.put("/api/memory/users/{telegram_user_key}/summary")
    async def update_memory_summary(
        telegram_user_key: str,
        payload: MemorySummaryUpdateRequest,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        current = await get_user_memory(telegram_user_key)
        latest_display_name = await get_latest_display_name_for_user(telegram_user_key)
        await upsert_user_memory(
            telegram_user_key,
            latest_display_name=latest_display_name or (current.latest_display_name if current else "") or telegram_user_key,
            memory_text=payload.memory_text.strip(),
            last_refreshed_date=current.last_refreshed_date if current else None,
        )
        return {"memory": _to_dict(await get_user_memory(telegram_user_key))}

    @application.patch("/api/memory/facts/{fact_id}")
    async def patch_user_fact(
        fact_id: int,
        payload: UserFactUpdateRequest,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        updated = await update_user_memory_fact(
            fact_id,
            fact_type=payload.fact_type,
            fact_text=payload.fact_text,
            confidence=payload.confidence,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="fact_not_found_or_no_update")
        return {"ok": True}

    @application.delete("/api/memory/facts/{fact_id}")
    async def delete_user_fact(
        fact_id: int,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, bool]:
        archived = await archive_user_memory_fact(fact_id)
        if not archived:
            raise HTTPException(status_code=404, detail="fact_not_found")
        return {"ok": True}

    @application.post("/api/memory/candidates/{candidate_id}/accept")
    async def accept_candidate(
        candidate_id: int,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, bool]:
        accepted = await accept_user_memory_candidate(candidate_id)
        if not accepted:
            raise HTTPException(status_code=404, detail="candidate_not_found")
        return {"ok": True}

    @application.post("/api/memory/candidates/{candidate_id}/reject")
    async def reject_candidate(
        candidate_id: int,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, bool]:
        rejected = await reject_user_memory_candidate(candidate_id)
        if not rejected:
            raise HTTPException(status_code=404, detail="candidate_not_found")
        return {"ok": True}

    @application.get("/api/global-memory/chats")
    async def global_memory_chats(
        limit: int = Query(120, ge=1, le=500),
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        rows = await list_global_memory_chat_overviews(limit=_bounded_limit(limit, default=120, maximum=500))
        return {"chats": _to_dict(rows)}

    @application.get("/api/global-memory/chats/{chat_id}")
    async def global_memory_detail(
        chat_id: int,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        facts = await get_global_memory_facts(chat_id, limit=100, min_confidence=0.0)
        return {"facts": _to_dict(facts)}

    @application.post("/api/global-memory/chats/{chat_id}/facts")
    async def create_global_fact(
        chat_id: int,
        payload: GlobalFactCreateRequest,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, Any]:
        await upsert_global_memory_facts(
            chat_id,
            [
                {
                    "fact_type": payload.fact_type,
                    "fact_text": payload.fact_text,
                    "confidence": payload.confidence,
                    "evidence_message_ids": [],
                }
            ],
        )
        return {"facts": _to_dict(await get_global_memory_facts(chat_id, limit=100, min_confidence=0.0))}

    @application.patch("/api/global-memory/chats/{chat_id}/facts/{fact_id}")
    async def patch_global_fact(
        chat_id: int,
        fact_id: int,
        payload: GlobalFactUpdateRequest,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, bool]:
        updated = await update_global_memory_fact(
            chat_id,
            fact_id,
            fact_type=payload.fact_type,
            fact_text=payload.fact_text,
            confidence=payload.confidence,
        )
        if not updated:
            raise HTTPException(status_code=404, detail="fact_not_found_or_no_update")
        return {"ok": True}

    @application.delete("/api/global-memory/chats/{chat_id}/facts/{fact_id}")
    async def delete_global_fact(
        chat_id: int,
        fact_id: int,
        _session: dict[str, Any] = Depends(require_session),
    ) -> dict[str, bool]:
        archived = await archive_global_memory_fact(chat_id, fact_id)
        if not archived:
            raise HTTPException(status_code=404, detail="fact_not_found")
        return {"ok": True}

    @application.get("/{full_path:path}")
    async def spa_fallback(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="not_found")
        return FileResponse(INDEX_FILE)

    return application


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("webadmin.app:app", host=webadmin_host(), port=webadmin_port(), reload=False)


if __name__ == "__main__":
    main()
