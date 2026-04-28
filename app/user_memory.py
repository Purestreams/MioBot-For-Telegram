"""Daily personal memory refresh for Telegram users."""

from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from typing import Optional

from app.ai_model import chat_completion_text
from app.database import (
    MessageRow,
    get_user_memory,
    get_user_messages_for_memory,
    upsert_user_memory,
)


logger = logging.getLogger(__name__)
MAX_SOURCE_MESSAGES = 200


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
            blocks.append(f"[{row.timestamp}] chat {row.chat_id} {row.username}: {row.content}")
    return "\n".join(blocks)


def _normalize_memory_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return "\n".join(lines).strip()


def _build_memory_messages(
    *,
    display_name: str,
    existing_memory: str,
    source_messages: str,
    target_end_date: str,
) -> list[dict[str, str]]:
    system_prompt = (
        "You maintain a concise long-term personal memory for one Telegram user.\n"
        "Update the memory using the provided historical memory plus the user's unsummarized messages up to yesterday.\n"
        "Focus on stable preferences, repeated interests, ongoing projects, habits, tone, social context, and recurring facts that help future replies.\n"
        "Do not include one-off trivial chatter unless it reveals a durable preference or a continuing thread.\n"
        "Write 4 to 8 short bullet-like lines as plain text, one fact per line, with no markdown heading and no JSON.\n"
        "If there is not enough durable information, keep the memory sparse instead of inventing facts."
    )
    user_prompt = (
        f"User display name: {display_name}\n"
        f"Update coverage end date (UTC): {target_end_date}\n\n"
        "Existing memory:\n"
        f"{existing_memory or '(empty)'}\n\n"
        "New source messages to fold in:\n"
        f"{source_messages or '(empty)'}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def refresh_user_memory_if_due(
    *,
    telegram_user_key: str,
    latest_display_name: str,
    today_utc: Optional[dt.date] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    if not telegram_user_key:
        return None

    target_date = _yesterday_utc(today_utc)
    target_date_str = target_date.isoformat()
    current = await get_user_memory(telegram_user_key)

    if current and current.last_refreshed_date and current.last_refreshed_date >= target_date_str:
        return current.memory_text or None

    start_date_exclusive = current.last_refreshed_date if current else None
    rows = await get_user_messages_for_memory(
        telegram_user_key,
        start_date_exclusive=start_date_exclusive,
        end_date_inclusive=target_date_str,
        limit=MAX_SOURCE_MESSAGES,
    )

    if not rows:
        await upsert_user_memory(
            telegram_user_key,
            latest_display_name=latest_display_name,
            memory_text=current.memory_text if current else "",
            last_refreshed_date=target_date_str,
        )
        return current.memory_text if current and current.memory_text else None

    source_messages = _format_messages_for_memory(rows)
    completion = await chat_completion_text(
        messages=_build_memory_messages(
            display_name=latest_display_name,
            existing_memory=current.memory_text if current else "",
            source_messages=source_messages,
            target_end_date=target_date_str,
        ),
        model=model,
        temperature=0.2,
        max_tokens=400,
    )
    memory_text = _normalize_memory_text(completion)
    if not memory_text:
        logger.warning("User memory refresh returned empty content for %s", telegram_user_key)
        return current.memory_text if current and current.memory_text else None

    await upsert_user_memory(
        telegram_user_key,
        latest_display_name=latest_display_name,
        memory_text=memory_text,
        last_refreshed_date=target_date_str,
    )
    return memory_text