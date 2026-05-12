"""Daily personal memory refresh for Telegram users."""

from __future__ import annotations

import ast
import datetime as dt
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from app.ai_model import chat_completion_text
from app.database import (
    MessageRow,
    UserMemoryCandidateRow,
    archive_user_memory_facts,
    get_user_memory,
    get_user_memory_candidate,
    get_user_memory_facts,
    get_user_messages_for_memory,
    list_user_memory_overviews,
    list_user_memory_candidates,
    mark_user_memory_candidates_status,
    update_user_memory_candidate_status,
    upsert_user_memory_candidate,
    upsert_user_memory_facts,
    upsert_user_memory,
)
from app.runtime_config import get_runtime_bool, get_runtime_int


logger = logging.getLogger(__name__)
MAX_SOURCE_MESSAGES = 200
MAX_CANDIDATES_FOR_REFRESH = 30


@dataclass(frozen=True)
class MemoryRefreshPayload:
    memory_text: str
    facts: list[dict[str, Any]]
    archive_fact_ids: list[int]


@dataclass(frozen=True)
class MemoryTextAuditRow:
    telegram_user_key: str
    latest_display_name: str
    stored_length: int
    normalized_length: int
    issue_types: list[str]
    preview: str


def _personal_memory_context_max_chars() -> int:
    return get_runtime_int("PERSONAL_MEMORY_CONTEXT_MAX_CHARS", 1800)


def _memory_candidate_auto_refresh_count() -> int:
    return get_runtime_int("MEMORY_CANDIDATE_AUTO_REFRESH_COUNT", 3)


def _memory_candidate_extraction_enabled() -> bool:
    return get_runtime_bool("MEMORY_CANDIDATE_EXTRACTION_ENABLED", True)


def _yesterday_utc(today: Optional[dt.date] = None) -> dt.date:
    base = today or dt.datetime.now(dt.timezone.utc).date()
    return base - dt.timedelta(days=1)


def _format_messages_for_memory(rows: list[MessageRow]) -> str:
    grouped: dict[str, list[MessageRow]] = defaultdict(list)
    for row in rows:
        grouped[str(row.timestamp).split(" ", 1)[0]].append(row)

    blocks: list[str] = []
    for day in sorted(grouped):
        blocks.append(f"## {day}")
        for row in grouped[day]:
            blocks.append(f"[message_id:{row.id}] [{row.timestamp}] chat {row.chat_id} {row.username}: {row.content}")
    return "\n".join(blocks)


def _normalize_memory_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _trim_text(text: str, *, max_chars: int) -> str:
    value = (text or "").strip()
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[: max_chars - 1].rstrip() + "…"


def _format_facts_for_prompt(facts) -> str:
    if not facts:
        return "(empty)"
    return "\n".join(
        f"- #{fact.id} [{fact.fact_type}] {fact.fact_text} "
        f"(confidence={fact.confidence:.2f}, evidence={fact.evidence_message_ids})"
        for fact in facts
    )


def _format_candidates_for_prompt(candidates: list[UserMemoryCandidateRow]) -> str:
    if not candidates:
        return "(empty)"
    return "\n".join(
        f"- candidate #{candidate.id} [{candidate.priority}/{candidate.fact_type}] {candidate.fact_text} "
        f"(confidence={candidate.confidence:.2f}, evidence={candidate.evidence_message_ids})"
        for candidate in candidates
    )


def _has_memory_content(current, facts) -> bool:
    return bool((current and current.memory_text.strip()) or facts)


def _build_memory_messages(
    *,
    display_name: str,
    existing_memory: str,
    existing_facts: str,
    pending_candidates: str,
    source_messages: str,
    target_end_date: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "You maintain a concise long-term personal memory for one Telegram user.\n"
        "Update the memory using the provided historical memory, structured facts, pending candidates, and source messages.\n"
        "Focus on stable preferences, repeated interests, ongoing projects, habits, tone, social context, and recurring facts that help future replies.\n"
        "Do not include one-off trivial chatter unless it reveals a durable preference or a continuing thread.\n"
        "Return valid JSON only with keys memory_text, facts, and archive_fact_ids.\n"
        "memory_text must be 4 to 8 short plain-text lines, one durable memory per line.\n"
        "facts must be a list of atomic facts with keys type, text, confidence, and evidence_message_ids.\n"
        "archive_fact_ids must list old fact ids to deactivate when a pending candidate or source message contradicts or replaces them.\n"
        "Allowed fact types: preference, project, style, identity, relationship, goal, note.\n"
        "Use message ids from [message_id:N] markers as evidence_message_ids.\n"
        "Prefer updating/archiving stale facts over keeping contradictory memories active.\n"
        "If there is not enough durable information, keep memory_text sparse and facts empty instead of inventing facts."
    )
    user_prompt = (
        "Existing memory:\n"
        f"{existing_memory or '(empty)'}\n\n"
        "Existing structured facts:\n"
        f"{existing_facts}\n\n"
        "Pending memory candidates:\n"
        f"{pending_candidates}\n\n"
        "New source messages to fold in:\n"
        f"{source_messages or '(empty)'}\n\n"
        "Update metadata:\n"
        f"User display name: {display_name}\n"
        f"Update coverage end date (UTC): {target_end_date}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _strip_markdown_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    in_string = False
    escape = False
    depth = 0
    start = -1

    for index, char in enumerate(text):
        if start == -1:
            if char == "{":
                start = index
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _extract_first_json_array(text: str) -> Optional[str]:
    in_string = False
    escape = False
    depth = 0
    start = -1

    for index, char in enumerate(text):
        if start == -1:
            if char == "[":
                start = index
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _coerce_memory_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, dict):
        return _coerce_memory_text(value.get("memory_text") or value.get("summary") or "")

    if isinstance(value, (list, tuple)):
        lines: list[str] = []
        for item in value:
            normalized = _coerce_memory_text(item)
            if normalized:
                lines.extend(normalized.splitlines())
        return _normalize_memory_text("\n".join(lines))

    text = _strip_markdown_code_fence(str(value or ""))
    if not text:
        return ""

    if text[0] in "[{":
        try:
            return _coerce_memory_text(json.loads(text))
        except json.JSONDecodeError:
            pass

        try:
            literal = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            literal = None
        if literal is not None:
            normalized = _coerce_memory_text(literal)
            if normalized:
                return normalized

        decoder = json.JSONDecoder()
        for key in ('"memory_text"', '"summary"'):
            start = text.find(key)
            if start == -1:
                continue

            candidate = text[start + len(key):]
            _sep, colon, candidate = candidate.partition(":")
            if not colon:
                continue
            candidate = candidate.lstrip()

            try:
                parsed, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                normalized = _coerce_memory_text(parsed)
                if normalized:
                    return normalized

            if candidate.startswith("["):
                array_text = _extract_first_json_array(candidate)
                if array_text:
                    try:
                        normalized = _coerce_memory_text(json.loads(array_text))
                    except json.JSONDecodeError:
                        normalized = ""
                    if normalized:
                        return normalized

    return _normalize_memory_text(text)


def _memory_text_issue_types(raw_text: str) -> list[str]:
    stripped = (raw_text or "").lstrip()
    issues: list[str] = []
    if not stripped:
        return issues
    if stripped.startswith("["):
        issues.append("list-literal")
    if stripped.startswith("{"):
        issues.append("json-blob")
    if stripped.startswith("```"):
        issues.append("code-fence")

    normalized = _coerce_memory_text(raw_text)
    if normalized != (raw_text or "").strip():
        issues.append("normalizes-differently")
    return issues


def _message_ids_from_rows(rows: list[MessageRow]) -> list[int]:
    return [row.id for row in rows]


def _coerce_evidence_ids(value: Any, fallback_ids: list[int]) -> list[int]:
    if value is None:
        return fallback_ids[:]
    source = value if isinstance(value, list) else [value]
    ids: list[int] = []
    for item in source:
        try:
            message_id = int(item)
        except (TypeError, ValueError):
            continue
        if message_id not in ids:
            ids.append(message_id)
    return ids or fallback_ids[:]


def _fact_candidates_from_lines(memory_text: str, source_rows: list[MessageRow]) -> list[dict[str, Any]]:
    evidence_ids = _message_ids_from_rows(source_rows)
    facts: list[dict[str, Any]] = []
    for line in _normalize_memory_text(memory_text).splitlines():
        cleaned = re.sub(r"^[-*•]\s*", "", line).strip()
        if not cleaned:
            continue
        facts.append(
            {
                "fact_type": "note",
                "fact_text": cleaned,
                "confidence": 0.55,
                "evidence_message_ids": evidence_ids,
            }
        )
    return facts


def _coerce_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    source = value if isinstance(value, list) else [value]
    ids: list[int] = []
    for item in source:
        try:
            item_id = int(item)
        except (TypeError, ValueError):
            continue
        if item_id > 0 and item_id not in ids:
            ids.append(item_id)
    return ids


def _parse_memory_refresh_payload(result_text: str, source_rows: list[MessageRow]) -> MemoryRefreshPayload:
    raw = _strip_markdown_code_fence(result_text or "")
    json_candidate = _extract_first_json_object(raw) or raw
    fallback_ids = _message_ids_from_rows(source_rows)

    try:
        payload = json.loads(json_candidate)
    except json.JSONDecodeError:
        memory_text = _coerce_memory_text(raw)
        return MemoryRefreshPayload(memory_text, _fact_candidates_from_lines(memory_text, source_rows), [])

    if not isinstance(payload, dict):
        memory_text = _coerce_memory_text(raw)
        return MemoryRefreshPayload(memory_text, _fact_candidates_from_lines(memory_text, source_rows), [])

    memory_text = _coerce_memory_text(payload.get("memory_text") or payload.get("summary") or "")
    facts: list[dict[str, Any]] = []
    raw_facts = payload.get("facts")
    if isinstance(raw_facts, list):
        for item in raw_facts:
            if not isinstance(item, dict):
                continue
            fact_text = str(item.get("text") or item.get("fact_text") or "").strip()
            if not fact_text:
                continue
            facts.append(
                {
                    "fact_type": str(item.get("type") or item.get("fact_type") or "note"),
                    "fact_text": fact_text,
                    "confidence": item.get("confidence", 0.6),
                    "evidence_message_ids": _coerce_evidence_ids(item.get("evidence_message_ids"), fallback_ids),
                }
            )

    if not memory_text and facts:
        memory_text = "\n".join(fact["fact_text"] for fact in facts)
    return MemoryRefreshPayload(memory_text, facts, _coerce_int_list(payload.get("archive_fact_ids")))


def _candidate_fact_type(text: str) -> str:
    lower = text.lower()
    if re.search(r"喜欢|不喜欢|更喜欢|prefer|like|dislike", lower):
        return "preference"
    if re.search(r"回答|回复|风格|语气|style|tone|answer|reply", lower):
        return "style"
    if re.search(r"项目|正在做|在做|working on|project|building", lower):
        return "project"
    if re.search(r"我是|我在|我的|i am|i'm|my name|i work", lower):
        return "identity"
    if re.search(r"目标|想要|希望|goal|want to|trying to", lower):
        return "goal"
    return "note"


def _candidate_priority(text: str) -> str:
    lower = text.lower()
    if re.search(r"记住|记一下|以后|别忘|remember|from now on|以后你|please remember", lower):
        return "fast"
    return "slow"


def _looks_like_memory_candidate(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 8:
        return False
    lower = stripped.lower()
    patterns = (
        r"记住|记一下|别忘|以后|我喜欢|我不喜欢|更喜欢|我是|我在|我的|正在做|项目|目标|想要|希望",
        r"remember|from now on|i prefer|i like|i dislike|i am|i'm|my name|working on|my project|my goal|want to",
    )
    return any(re.search(pattern, lower) for pattern in patterns)


def _normalize_candidate_text(text: str) -> str:
    value = re.sub(r"\s+", " ", text).strip()
    return value[:500].rstrip()


def build_memory_candidate_from_message(message_text: str) -> Optional[dict[str, Any]]:
    if not _looks_like_memory_candidate(message_text):
        return None
    fact_text = _normalize_candidate_text(message_text)
    priority = _candidate_priority(fact_text)
    return {
        "fact_type": _candidate_fact_type(fact_text),
        "fact_text": fact_text,
        "confidence": 0.78 if priority == "fast" else 0.58,
        "priority": priority,
    }


async def extract_user_memory_candidate_from_message(
    *,
    telegram_user_key: str,
    message_text: str,
    message_id: Optional[int],
) -> Optional[int]:
    if not _memory_candidate_extraction_enabled() or not telegram_user_key:
        return None

    candidate = build_memory_candidate_from_message(message_text)
    if not candidate:
        return None
    return await upsert_user_memory_candidate(
        telegram_user_key,
        fact_type=candidate["fact_type"],
        fact_text=candidate["fact_text"],
        confidence=candidate["confidence"],
        evidence_message_ids=[message_id] if message_id is not None else [],
        source_message_id=message_id,
        priority=candidate["priority"],
    )


async def accept_user_memory_candidate(candidate_id: int) -> bool:
    candidate = await get_user_memory_candidate(candidate_id)
    if not candidate or candidate.status != "pending":
        return False

    await upsert_user_memory_facts(
        candidate.telegram_user_key,
        [
            {
                "fact_type": candidate.fact_type,
                "fact_text": candidate.fact_text,
                "confidence": candidate.confidence,
                "evidence_message_ids": candidate.evidence_message_ids,
            }
        ],
    )
    await update_user_memory_candidate_status(candidate.id, "accepted", review_note="accepted by admin")
    return True


async def reject_user_memory_candidate(candidate_id: int, *, note: Optional[str] = None) -> bool:
    return await update_user_memory_candidate_status(candidate_id, "rejected", review_note=note or "rejected by admin")


async def get_personal_memory_context(
    telegram_user_key: str,
    *,
    max_facts: int = 6,
    max_chars: Optional[int] = None,
) -> Optional[str]:
    if not telegram_user_key:
        return None

    current = await get_user_memory(telegram_user_key)
    facts = await get_user_memory_facts(telegram_user_key, limit=max_facts, min_confidence=0.2)
    normalized_memory_text = _coerce_memory_text(current.memory_text) if current else ""

    sections: list[str] = []
    if facts:
        sections.append("structured_facts:")
        sections.extend(f"- [{fact.fact_type}] {fact.fact_text}" for fact in facts)
    if normalized_memory_text:
        sections.append("summary:")
        sections.extend(normalized_memory_text.splitlines())

    if not sections:
        return None

    return _trim_text("\n".join(sections), max_chars=max_chars or _personal_memory_context_max_chars()) or None


async def audit_user_memory_texts(*, limit: Optional[int] = 200) -> list[MemoryTextAuditRow]:
    rows = await list_user_memory_overviews(limit=limit)
    findings: list[MemoryTextAuditRow] = []
    for row in rows:
        raw_text = row.memory_text or ""
        if not raw_text.strip():
            continue
        issues = _memory_text_issue_types(raw_text)
        if not issues:
            continue
        normalized = _coerce_memory_text(raw_text)
        findings.append(
            MemoryTextAuditRow(
                telegram_user_key=row.telegram_user_key,
                latest_display_name=row.latest_display_name,
                stored_length=len(raw_text),
                normalized_length=len(normalized),
                issue_types=issues,
                preview=raw_text.strip().replace("\n", " | "),
            )
        )
    return findings


async def refresh_user_memory_if_due(
    *,
    telegram_user_key: str,
    latest_display_name: str,
    today_utc: Optional[dt.date] = None,
    model: Optional[str] = None,
    force: bool = False,
) -> Optional[str]:
    if not telegram_user_key:
        return None

    today = today_utc or dt.datetime.now(dt.timezone.utc).date()
    default_target_date = _yesterday_utc(today)
    current = await get_user_memory(telegram_user_key)
    existing_facts = await get_user_memory_facts(telegram_user_key, limit=12, min_confidence=0.0)
    pending_candidates = await list_user_memory_candidates(
        telegram_user_key,
        status="pending",
        limit=MAX_CANDIDATES_FOR_REFRESH,
    )
    memory_is_empty = not _has_memory_content(current, existing_facts)
    candidate_trigger = bool(
        pending_candidates
        and (
            force
            or any(candidate.priority == "fast" for candidate in pending_candidates)
            or len(pending_candidates) >= _memory_candidate_auto_refresh_count()
        )
    )

    target_date = today if force else default_target_date
    target_date_str = target_date.isoformat()

    if (
        not force
        and not memory_is_empty
        and not candidate_trigger
        and current
        and current.last_refreshed_date
        and current.last_refreshed_date >= target_date_str
    ):
        return current.memory_text or None

    start_date_exclusive = None if force or memory_is_empty else current.last_refreshed_date if current else None
    rows = await get_user_messages_for_memory(
        telegram_user_key,
        start_date_exclusive=start_date_exclusive,
        end_date_inclusive=target_date_str,
        limit=None if force or memory_is_empty else MAX_SOURCE_MESSAGES,
    )

    if not rows and not pending_candidates:
        empty_target_date_str = default_target_date.isoformat() if memory_is_empty and not force else target_date_str
        await upsert_user_memory(
            telegram_user_key,
            latest_display_name=latest_display_name,
            memory_text=current.memory_text if current else "",
            last_refreshed_date=empty_target_date_str,
        )
        return current.memory_text if current and current.memory_text else None

    source_messages = _format_messages_for_memory(rows)
    completion = await chat_completion_text(
        messages=_build_memory_messages(
            display_name=latest_display_name,
            existing_memory=current.memory_text if current else "",
            existing_facts=_format_facts_for_prompt(existing_facts),
            pending_candidates=_format_candidates_for_prompt(pending_candidates),
            source_messages=source_messages,
            target_end_date=target_date_str,
        ),
        model=model,
        temperature=0.2,
        max_tokens=700,
    )
    payload = _parse_memory_refresh_payload(completion, rows)
    if not payload.memory_text:
        logger.warning("User memory refresh returned empty content for %s", telegram_user_key)
        return current.memory_text if current and current.memory_text else None

    await archive_user_memory_facts(payload.archive_fact_ids)
    await upsert_user_memory(
        telegram_user_key,
        latest_display_name=latest_display_name,
        memory_text=payload.memory_text,
        last_refreshed_date=target_date_str,
    )
    await upsert_user_memory_facts(telegram_user_key, payload.facts)
    await mark_user_memory_candidates_status(
        [candidate.id for candidate in pending_candidates],
        "accepted",
        review_note="consolidated by memory refresh",
    )
    return payload.memory_text