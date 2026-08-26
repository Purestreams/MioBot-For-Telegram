"""Non-blocking startup maintenance for the Telegram application."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.database import log_embedding_health_report, reindex_message_embeddings
from app.runtime_config import get_runtime_value

logger = logging.getLogger(__name__)


def _rag_reindex_mode() -> str:
    mode = (get_runtime_value("RAG_REINDEX_ON_STARTUP") or "background").strip().lower()
    if mode in {"background", "blocking", "disabled"}:
        return mode
    logger.warning("Invalid RAG_REINDEX_ON_STARTUP=%s; using background.", mode)
    return "background"


async def _reindex_embeddings_in_background() -> None:
    """Run a full rebuild without delaying Telegram polling startup."""
    try:
        result = await reindex_message_embeddings()
        logger.info(
            "Background embedding reindex finished. reindexed=%s signature=%s",
            result.get("reindexed"),
            result.get("signature"),
        )
        await log_embedding_health_report()
    except Exception:
        logger.exception("Background embedding reindex failed; run `miobot-rag reindex` to retry.")


async def prepare_embedding_index(application: Any) -> None:
    """Check index compatibility and schedule a rebuild according to runtime policy.

    The default ``background`` mode keeps the bot available while a potentially
    large embedding corpus is rebuilt. ``blocking`` is available for operators
    who require a fully current index before accepting updates; ``disabled``
    leaves reindexing to the explicit maintenance CLI.
    """
    report = await log_embedding_health_report()
    if not report.get("needs_reindex"):
        return

    mode = _rag_reindex_mode()
    if mode == "disabled":
        logger.warning(
            "Embedding index needs reindexing but startup reindexing is disabled. "
            "Run `miobot-rag reindex`. db=%s",
            report.get("db_file", "(unknown)"),
        )
        return

    logger.warning(
        "Embedding index requires reindexing. mode=%s db=%s",
        mode,
        report.get("db_file", "(unknown)"),
    )
    if mode == "blocking":
        await _reindex_embeddings_in_background()
        return

    coroutine = _reindex_embeddings_in_background()
    create_task = getattr(application, "create_task", None)
    if callable(create_task):
        create_task(coroutine)
    else:  # pragma: no cover - compatibility fallback for non-PTB callers
        asyncio.create_task(coroutine)
