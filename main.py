"""Telegram bot entrypoint and handler orchestration."""

# general imports
import asyncio
import datetime
import io
import logging
import multiprocessing
import os
import time
from typing import Any, Optional
from urllib.parse import quote

from telegram import InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.error import Conflict
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


# private imports
from app.runtime_config import (
    bootstrap_runtime_environment,
    get_ark_chat_completions_endpoint,
    get_runtime_bool,
    get_runtime_int,
    get_runtime_value,
)

# Load config/runtime env before importing modules that read env at import time.
bootstrap_runtime_environment()

from app.md2jpg import md_to_image
from app.text2md import plain_text_to_markdown
from app.twitter_downloader import (
    TwitterDownloader,
    build_twitter_caption,
    format_tweet_text_for_reply,
    is_twitter_status_url,
    summarize_tweet_text,
)
from app.zhihu_dl import download_image_media as download_zhihu_image_media
from app.zhihu_dl import parse_link as parse_zhihu_link
from app.youtube_dl import (
    download_video_to_file,
    compress_video_if_needed,
    resolve_caption_url,
)
from app.reply2message import (
    ReplyActivationDecision,
    choose_reply_sticker,
    direct_reply_activation_decision,
    generate_group_reply,
    reply_activation_decision_context_lines,
    should_activate_reply,
)
from app.rag_embeddings import ensure_fastembed_ready
from app.user_memory import (
    accept_user_memory_candidate,
    audit_user_memory_texts,
    extract_user_memory_candidate_from_message,
    get_global_memory_context,
    get_personal_memory_context,
    refresh_user_memory_if_due,
    reject_user_memory_candidate,
)
from app.database import (
    add_message,
    archive_global_memory_fact,
    archive_user_memory_fact,
    create_webadmin_login_token,
    find_sticker_reply_candidates,
    get_global_memory_facts,
    get_latest_display_name_for_user,
    list_global_memory_chat_overviews,
    get_prompt_context_parts,
    get_sticker_text,
    get_user_memory,
    get_user_memory_facts,
    init_db,
    list_user_memory_candidates,
    list_user_memory_overviews,
    log_embedding_health_report,
    reindex_message_embeddings,
    record_sticker_reply_usage,
    search_user_memories,
    update_user_memory_fact,
    upsert_global_memory_facts,
    upsert_user_memory,
    upsert_sticker_text,
)
from webadmin.security import (
    format_ttl,
    generate_login_token,
    hash_login_token,
    parse_login_token_ttl_seconds,
    webadmin_base_url,
    validate_webadmin_security_configuration,
)
from app.image2text import StickerUnderstanding, image_to_text, sticker_to_understanding

from app.cryto import get_Allez_APR, get_Allez_USDC_APR, get_Price_Coinbase

from app.med import MedRenderError, generate_jpg_from_med_json, generate_med
from app.ai_model import configure_llm
from app.main_helpers import (
    OUTPUT_DIR,
    MD2JPG_REGEX,
    TEXT2JPG_REGEX,
    _build_output_path,
    _remove_file_if_exists,
    _delete_message_if_exists,
    _extract_video_url,
    _is_reply_to_this_bot,
    _classify_group_reply_trigger,
    _display_name_from_user,
    is_zhihu_answer_url,
    is_zhihu_url,
    _telegram_user_key_from_user,
    _build_reply_relation_payload,
    _match_command_payload,
    _build_rag_query_from_message,
    _is_group_chat,
    _extract_search_keywords,
    extract_supported_links,
    extract_supported_links_from_message,
)

AZURE_OPENAI_ENDPOINT = get_runtime_value("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_runtime_value("AZURE_OPENAI_API_KEY")
TELEGRAM_BOT_USERNAME = get_runtime_value("TELEGRAM_BOT_USERNAME")
TELEGRAM_BOT_KEY = get_runtime_value("TELEGRAM_BOT_KEY")
ARK_ENDPOINT = get_ark_chat_completions_endpoint()
ARK_API_KEY = get_runtime_value("ARK_API_KEY")

AZURE_OPENAI_API_VERSION = get_runtime_value("AZURE_OPENAI_API_VERSION")

# Models: Phi-4-mini-instruct, Phi-4 or gpt-4.1-nano
AZURE_OPENAI_DEPLOYMENT_NAME = get_runtime_value("AZURE_OPENAI_DEPLOYMENT_NAME")

ARK_API_KEY = get_runtime_value("ARK_API_KEY")
ARK_MODEL = get_runtime_value("ARK_MODEL")
LLM_PROVIDER = get_runtime_value("LLM_PROVIDER")
if LLM_PROVIDER:
    normalized_provider = LLM_PROVIDER.strip().lower()
    if normalized_provider in {"azure_openai", "azure-openai", "azureopenai"}:
        LLM_PROVIDER = "azure"
    else:
        LLM_PROVIDER = normalized_provider


def _warn_missing_runtime_env(provider: str) -> None:
    required = ["TELEGRAM_BOT_KEY"]
    if provider == "azure":
        required += ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"]
    elif provider == "ark":
        required += ["ARK_API_KEY", "ARK_API_ENDPOINT", "ARK_MODEL"]
    elif provider == "ollama":
        required += ["OLLAMA_ENDPOINT", "OLLAMA_MODEL"]

    missing = [name for name in required if not get_runtime_value(name)]
    if missing:
        logger.warning(
            "Missing runtime env values at startup: %s. Check config/runtime.env or config/runtime.local.env.",
            ", ".join(missing),
        )


_warn_missing_runtime_env(LLM_PROVIDER or "ark")

configure_llm(
    provider=LLM_PROVIDER,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_api_key=AZURE_OPENAI_API_KEY,
    azure_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    ark_api_key=ARK_API_KEY,
    ark_model=ARK_MODEL,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
TELEGRAM_CAPTION_LIMIT = 1024
TELEGRAM_TEXT_LIMIT = 4096
_BACKGROUND_TASKS: set[asyncio.Task] = set()
_USER_MEMORY_REFRESH_KEYS: set[str] = set()
MAX_RENDER_DOCUMENT_BYTES = max(1, get_runtime_int("MAX_RENDER_DOCUMENT_BYTES", 10 * 1024 * 1024))
MAX_MEDIA_LINKS_PER_MESSAGE = max(1, get_runtime_int("MAX_MEDIA_LINKS_PER_MESSAGE", 4))
MEDIA_PROCESSING_CONCURRENCY = max(1, get_runtime_int("MEDIA_PROCESSING_CONCURRENCY", 2))
_MEDIA_PROCESSING_SEMAPHORE = asyncio.Semaphore(MEDIA_PROCESSING_CONCURRENCY)


def _configured_admin_user_ids() -> set[int]:
    raw_value = get_runtime_value("TELEGRAM_ADMIN_USER_IDS")
    admin_ids: set[int] = set()
    for token in raw_value.replace(",", " ").split():
        value = token.strip()
        if value.startswith("tg_user:"):
            value = value.split(":", 1)[1]
        try:
            admin_ids.add(int(value))
        except ValueError:
            continue
    return admin_ids


def _has_configured_admins() -> bool:
    return bool(_configured_admin_user_ids())


def _is_private_chat(update: Update) -> bool:
    return getattr(update.effective_chat, "type", None) == "private"


def _is_admin_update(update: Update) -> bool:
    user_id = getattr(update.effective_user, "id", None)
    return isinstance(user_id, int) and user_id in _configured_admin_user_ids()


def _admin_command_args(context: ContextTypes.DEFAULT_TYPE) -> list[str]:
    return [str(arg) for arg in (getattr(context, "args", None) or [])]


def _normalize_admin_memory_key(value: str) -> str:
    candidate = (value or "").strip()
    if candidate.startswith("tg_user:"):
        return candidate
    if candidate.isdigit():
        return f"tg_user:{candidate}"
    return candidate


def _admin_payload_after_first_arg(args: list[str]) -> str:
    if len(args) <= 1:
        return ""
    return " ".join(args[1:]).strip()


def _compact_admin_text(text: str, *, max_chars: int = 700) -> str:
    value = (text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def _truncate_admin_reply(text: str) -> str:
    if len(text) <= TELEGRAM_TEXT_LIMIT:
        return text
    return text[: TELEGRAM_TEXT_LIMIT - 3].rstrip() + "..."


async def _ensure_admin_private_chat(update: Update) -> bool:
    if not update.message:
        return False
    if not _is_private_chat(update):
        await update.message.reply_text("Memory admin commands are only available in a private chat.")
        return False
    if not _has_configured_admins():
        await update.message.reply_text("Memory admin commands are disabled. Set TELEGRAM_ADMIN_USER_IDS first.")
        return False
    if not _is_admin_update(update):
        await update.message.reply_text("You are not allowed to use memory admin commands.")
        return False
    return True


async def handle_memory_admin_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return
    await message.reply_text(
        "Memory admin commands:\n"
        "/memory_help - show this help\n"
        "/memories - list users with message history or memory\n"
        "/memory_audit [limit] - inspect malformed summaries (0 = full scan)\n"
        "/memory <telegram_user_id|tg_user:key> - view one user's memory\n"
        "/memory_search <keyword> - search summaries and facts\n"
        "/memory_refresh <telegram_user_id|tg_user:key> - regenerate one user's memory from history\n"
        "/memory_set <telegram_user_id|tg_user:key> <text> - replace one user's summary memory\n"
        "/memory_candidates [telegram_user_id|tg_user:key] - list pending memory candidates\n"
        "/memory_accept <candidate_id> - accept a candidate into facts\n"
        "/memory_reject <candidate_id> - reject a candidate\n"
        "/memory_fact_set <fact_id> <text> - edit an active fact\n"
        "/memory_fact_delete <fact_id> - archive an active fact\n"
        "/global_memory <chat_id> - view group-scoped global memory\n"
        "/global_memory_set <chat_id> <fact_type> <text> - add/update group-scoped global memory\n"
        "/global_memory_delete <chat_id> <fact_id> - archive a group-scoped global memory fact\n"
        "/webadmin_token [10m] - create a one-time web admin login token"
    )


async def handle_webadmin_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    ttl_seconds = parse_login_token_ttl_seconds(args[0] if args else None)
    raw_token = generate_login_token()
    user_id = getattr(update.effective_user, "id", None)
    username = str(getattr(update.effective_user, "username", "") or "").lstrip("@")
    token_row = await create_webadmin_login_token(
        hash_login_token(raw_token),
        admin_user_id=user_id if isinstance(user_id, int) else None,
        admin_username=username,
        ttl_seconds=ttl_seconds,
    )
    login_url = f"{webadmin_base_url()}/?token={quote(raw_token)}"
    await message.reply_text(
        _truncate_admin_reply(
            "Web admin login token created.\n"
            f"URL: {login_url}\n"
            f"Token: {raw_token}\n"
            f"Expires at UTC: {token_row.expires_at}\n"
            f"TTL: {format_ttl(ttl_seconds)}\n"
            "The token is single-use."
        )
    )


async def handle_memory_admin_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    rows = await list_user_memory_overviews(limit=40)
    if not rows:
        await message.reply_text("No users with message history or memory yet.")
        return

    lines = [f"Users with message history or memory ({len(rows)} shown):"]
    for row in rows:
        display_name = row.latest_display_name or row.telegram_user_key
        refreshed = row.last_refreshed_date or "never"
        latest_message = row.latest_message_at or "none"
        summary_state = "summary=yes" if row.memory_text.strip() else "summary=no"
        lines.append(
            f"- {display_name} | {row.telegram_user_key} | facts={row.fact_count} | "
            f"{summary_state} | refreshed={refreshed} | latest={latest_message}"
        )
    await message.reply_text(_truncate_admin_reply("\n".join(lines)))


async def handle_memory_admin_audit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    limit = _parse_non_negative_int(args[0]) if args else 200
    if args and limit is None:
        await message.reply_text("Usage: /memory_audit [non_negative_limit], use 0 for full scan")
        return

    audit_limit = None if limit == 0 else limit
    scope_text = "all rows" if audit_limit is None else f"the latest {audit_limit} rows"
    rows = await audit_user_memory_texts(limit=audit_limit)
    if not rows:
        await message.reply_text(f"Memory audit found no malformed summaries in {scope_text}.")
        return

    lines = [f"Memory audit found {len(rows)} malformed summaries in {scope_text}:"]
    for row in rows:
        display_name = row.latest_display_name or row.telegram_user_key
        lines.append(
            f"- {display_name} | {row.telegram_user_key} | issues={','.join(row.issue_types)} | "
            f"len={row.stored_length}->{row.normalized_length} | {_compact_admin_text(row.preview, max_chars=180)}"
        )
    await message.reply_text(_truncate_admin_reply("\n".join(lines)))


async def handle_memory_admin_view(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    if not args:
        await message.reply_text("Usage: /memory <telegram_user_id|tg_user:key>")
        return

    telegram_user_key = _normalize_admin_memory_key(args[0])
    current = await get_user_memory(telegram_user_key)
    facts = await get_user_memory_facts(telegram_user_key, limit=25, min_confidence=0.0)
    latest_display_name = await get_latest_display_name_for_user(telegram_user_key)
    if not current and not facts and not latest_display_name:
        await message.reply_text(f"No memory or message history found for {telegram_user_key}.")
        return

    display_name = latest_display_name or (current.latest_display_name if current else "") or telegram_user_key
    lines = [
        f"Memory for {display_name}",
        f"key: {telegram_user_key}",
        f"last_refreshed_date: {current.last_refreshed_date if current else 'never'}",
        "",
        "summary:",
        _compact_admin_text(current.memory_text if current else "") or "(empty)",
        "",
        "facts:",
    ]
    if facts:
        for fact in facts:
            lines.append(f"- #{fact.id} [{fact.fact_type}] {fact.fact_text} (confidence={fact.confidence:.2f})")
    else:
        lines.append("(empty)")
    await message.reply_text(_truncate_admin_reply("\n".join(lines)))


async def handle_memory_admin_search(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    query = " ".join(_admin_command_args(context)).strip()
    if not query:
        await message.reply_text("Usage: /memory_search <keyword>")
        return

    rows = await search_user_memories(query, limit=25)
    if not rows:
        await message.reply_text(f"No memory matches for {query}.")
        return

    lines = [f"Memory search results for {query} ({len(rows)} shown):"]
    for row in rows:
        display_name = row.latest_display_name or row.telegram_user_key
        lines.append(f"- {display_name} | {row.telegram_user_key} | {row.source}: {_compact_admin_text(row.text, max_chars=220)}")
    await message.reply_text(_truncate_admin_reply("\n".join(lines)))


async def handle_memory_admin_refresh(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    if not args:
        await message.reply_text("Usage: /memory_refresh <telegram_user_id|tg_user:key>")
        return

    telegram_user_key = _normalize_admin_memory_key(args[0])
    latest_display_name = await get_latest_display_name_for_user(telegram_user_key)
    if not latest_display_name:
        await message.reply_text(f"No message history found for {telegram_user_key}.")
        return

    await message.reply_text(f"Refreshing memory for {latest_display_name} ({telegram_user_key})...")
    memory_text = await refresh_user_memory_if_due(
        telegram_user_key=telegram_user_key,
        latest_display_name=latest_display_name,
        force=True,
    )
    await message.reply_text(
        "Memory refresh finished.\n"
        f"key: {telegram_user_key}\n"
        f"summary:\n{_compact_admin_text(memory_text or '(empty)')}"
    )


async def handle_memory_admin_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    if len(args) < 2:
        await message.reply_text("Usage: /memory_set <telegram_user_id|tg_user:key> <summary text>")
        return

    telegram_user_key = _normalize_admin_memory_key(args[0])
    memory_text = _admin_payload_after_first_arg(args)
    if not memory_text:
        await message.reply_text("Usage: /memory_set <telegram_user_id|tg_user:key> <summary text>")
        return

    current = await get_user_memory(telegram_user_key)
    latest_display_name = await get_latest_display_name_for_user(telegram_user_key)
    await upsert_user_memory(
        telegram_user_key,
        latest_display_name=latest_display_name or (current.latest_display_name if current else "") or telegram_user_key,
        memory_text=memory_text,
        last_refreshed_date=current.last_refreshed_date if current else None,
    )
    await message.reply_text(
        "Memory summary updated.\n"
        f"key: {telegram_user_key}\n"
        f"summary:\n{_compact_admin_text(memory_text)}"
    )


def _parse_positive_int(value: str) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _parse_non_negative_int(value: str) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _parse_chat_id(value: str) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed != 0 else None


def _resolve_global_memory_chat_args(args: list[str]) -> tuple[Optional[int], list[str], Optional[str]]:
    chat_id = _parse_chat_id(args[0]) if args else None
    if chat_id is None:
        return None, args, "A chat_id is required for global memory commands."
    return chat_id, args[1:], None


async def _build_global_memory_chat_list_text(*, prefix: str = "") -> str:
    rows = await list_global_memory_chat_overviews(limit=40)
    lines: list[str] = []
    if prefix:
        lines.append(prefix.rstrip())
        lines.append("")

    if not rows:
        lines.append("No chats found yet. Global memory commands require an explicit chat_id once chats exist.")
        return _truncate_admin_reply("\n".join(lines))

    lines.append(f"Available chat_ids ({len(rows)} shown):")
    for row in rows:
        latest = row.latest_message_at or "none"
        latest_user = row.latest_message_username or "unknown"
        preview = _compact_admin_text(row.latest_message_preview or "", max_chars=120) or "(no preview)"
        lines.append(
            f"- chat_id={row.chat_id} | messages={row.message_count} | global_facts={row.global_fact_count} | "
            f"latest={latest} | {latest_user}: {preview}"
        )
    return _truncate_admin_reply("\n".join(lines))


async def _reply_global_memory_chat_list(message, *, prefix: str = "") -> None:
    await message.reply_text(await _build_global_memory_chat_list_text(prefix=prefix))


async def handle_global_memory_admin_view(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    chat_id, _remaining_args, error = _resolve_global_memory_chat_args(_admin_command_args(context))
    if error or chat_id is None:
        await _reply_global_memory_chat_list(
            message,
            prefix="Usage: /global_memory <chat_id>\nA chat_id is required. Choose one below.",
        )
        return

    facts = await get_global_memory_facts(chat_id, limit=25, min_confidence=0.0)
    lines = [f"Global memory for chat_id={chat_id}:"]
    if facts:
        for fact in facts:
            lines.append(f"- #{fact.id} [{fact.fact_type}] {fact.fact_text} (confidence={fact.confidence:.2f})")
    else:
        lines.append("(empty)")
    await message.reply_text(_truncate_admin_reply("\n".join(lines)))


async def handle_global_memory_admin_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    chat_id, remaining_args, error = _resolve_global_memory_chat_args(_admin_command_args(context))
    if error or chat_id is None or len(remaining_args) < 2:
        await _reply_global_memory_chat_list(
            message,
            prefix="Usage: /global_memory_set <chat_id> <fact_type> <text>\nChoose a chat_id below.",
        )
        return

    fact_type = remaining_args[0].strip().lower()
    fact_text = " ".join(remaining_args[1:]).strip()
    if not fact_type or not fact_text:
        await _reply_global_memory_chat_list(
            message,
            prefix="Usage: /global_memory_set <chat_id> <fact_type> <text>\nChoose a chat_id below.",
        )
        return

    await upsert_global_memory_facts(
        chat_id,
        [
            {
                "fact_type": fact_type,
                "fact_text": fact_text,
                "confidence": 0.9,
                "evidence_message_ids": [],
            }
        ],
    )
    await message.reply_text(
        "Global memory fact saved.\n"
        f"chat_id: {chat_id}\n"
        f"[{fact_type}] {_compact_admin_text(fact_text)}"
    )


async def handle_global_memory_admin_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    chat_id, remaining_args, error = _resolve_global_memory_chat_args(_admin_command_args(context))
    fact_id = _parse_positive_int(remaining_args[0]) if remaining_args else None
    if error or chat_id is None or fact_id is None:
        await _reply_global_memory_chat_list(
            message,
            prefix="Usage: /global_memory_delete <chat_id> <fact_id>\nChoose a chat_id below, then run /global_memory <chat_id> to see fact IDs.",
        )
        return

    archived = await archive_global_memory_fact(chat_id, fact_id)
    await message.reply_text(f"Global memory fact #{fact_id} archived for chat_id={chat_id}." if archived else f"Global memory fact #{fact_id} was not found for chat_id={chat_id}.")


async def handle_memory_admin_candidates(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    telegram_user_key = _normalize_admin_memory_key(args[0]) if args else None
    candidates = await list_user_memory_candidates(telegram_user_key, status="pending", limit=30)
    if not candidates:
        await message.reply_text("No pending memory candidates.")
        return

    lines = [f"Pending memory candidates ({len(candidates)} shown):"]
    for candidate in candidates:
        evidence = ",".join(str(message_id) for message_id in candidate.evidence_message_ids) or "none"
        lines.append(
            f"- #{candidate.id} | {candidate.telegram_user_key} | {candidate.priority}/{candidate.fact_type} | "
            f"confidence={candidate.confidence:.2f} | evidence={evidence} | "
            f"{_compact_admin_text(candidate.fact_text, max_chars=220)}"
        )
    await message.reply_text(_truncate_admin_reply("\n".join(lines)))


async def handle_memory_admin_accept(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    candidate_id = _parse_positive_int(args[0]) if args else None
    if candidate_id is None:
        await message.reply_text("Usage: /memory_accept <candidate_id>")
        return

    accepted = await accept_user_memory_candidate(candidate_id)
    await message.reply_text(f"Candidate #{candidate_id} accepted." if accepted else f"Candidate #{candidate_id} was not found or is not pending.")


async def handle_memory_admin_reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    candidate_id = _parse_positive_int(args[0]) if args else None
    if candidate_id is None:
        await message.reply_text("Usage: /memory_reject <candidate_id>")
        return

    rejected = await reject_user_memory_candidate(candidate_id)
    await message.reply_text(f"Candidate #{candidate_id} rejected." if rejected else f"Candidate #{candidate_id} was not found.")


async def handle_memory_admin_fact_set(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    fact_id = _parse_positive_int(args[0]) if args else None
    fact_text = _admin_payload_after_first_arg(args)
    if fact_id is None or not fact_text:
        await message.reply_text("Usage: /memory_fact_set <fact_id> <text>")
        return

    updated = await update_user_memory_fact(fact_id, fact_text=fact_text)
    await message.reply_text(
        f"Fact #{fact_id} updated.\n{_compact_admin_text(fact_text)}"
        if updated
        else f"Fact #{fact_id} was not found."
    )


async def handle_memory_admin_fact_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not await _ensure_admin_private_chat(update):
        return
    message = update.message
    if not message:
        return

    args = _admin_command_args(context)
    fact_id = _parse_positive_int(args[0]) if args else None
    if fact_id is None:
        await message.reply_text("Usage: /memory_fact_delete <fact_id>")
        return

    archived = await archive_user_memory_fact(fact_id)
    await message.reply_text(f"Fact #{fact_id} archived." if archived else f"Fact #{fact_id} was not found.")


def _truncate_caption_text(text: str, max_chars: int = TELEGRAM_CAPTION_LIMIT) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _truncate_message_text(text: str, max_chars: int = TELEGRAM_TEXT_LIMIT) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _build_detailed_media_error_message(exc: Exception, *, max_chars: int = TELEGRAM_TEXT_LIMIT) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc

    while current and id(current) not in seen:
        seen.add(id(current))
        detail = str(current).strip() or repr(current)
        label = type(current).__name__
        parts.append(f"{label}: {detail}" if detail else label)
        current = current.__cause__ or current.__context__

    error_text = "\nCaused by:\n".join(parts) if parts else type(exc).__name__
    message = f"Media link processing failed.\n{error_text}"
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3].rstrip() + "..."


def _build_med_error_message(exc: Exception, *, max_chars: int = TELEGRAM_TEXT_LIMIT) -> str:
    if isinstance(exc, ValueError):
        prefix = "Failed to generate valid MED JSON from the provided text."
    elif isinstance(exc, MedRenderError):
        prefix = "MED image rendering failed."
    else:
        prefix = "Sorry, I encountered an error while creating your MED image."

    detail = f"{type(exc).__name__}: {exc}".strip()
    message = f"{prefix}\n{detail}"
    if len(message) <= max_chars:
        return message
    return message[: max_chars - 3].rstrip() + "..."


def _build_group_reply_runtime_state(
    *,
    sender_display: str,
    trigger_type: str,
    direct_addressed: bool,
) -> list[str]:
    now_utc = datetime.datetime.now(datetime.timezone.utc)
    return [
        f"current_date_utc: {now_utc.date().isoformat()}",
        f"current_weekday_utc: {now_utc.strftime('%A')}",
        f"sender_display: {sender_display}",
        f"trigger_type: {trigger_type}",
        f"direct_addressed: {str(direct_addressed).lower()}",
    ]


def _build_group_reply_memory_subjects(update: Update, *, sender_user, sender_display: str, telegram_user_key: Optional[str]) -> list[dict[str, str]]:
    subjects: list[dict[str, str]] = []
    if telegram_user_key:
        subjects.append(
            {
                "key": "sender",
                "role": "latest_message_author",
                "display": sender_display,
                "telegram_user_key": telegram_user_key,
            }
        )

    replied_message = getattr(getattr(update, "message", None), "reply_to_message", None)
    replied_user = getattr(replied_message, "from_user", None) if replied_message else None
    replied_user_key = _telegram_user_key_from_user(replied_user)
    if replied_user_key and replied_user_key != telegram_user_key and not getattr(replied_user, "is_bot", False):
        subjects.append(
            {
                "key": "replied_to_author",
                "role": "replied_message_author",
                "display": _display_name_from_user(replied_user),
                "telegram_user_key": replied_user_key,
            }
        )
    return subjects


def _coerce_activation_decision(value: object, *, direct_reason: str = "direct trigger") -> ReplyActivationDecision:
    if isinstance(value, ReplyActivationDecision):
        return value
    if isinstance(value, bool):
        if value:
            return direct_reply_activation_decision(reason=direct_reason)
        return ReplyActivationDecision(should_reply=False, reason="activation probe declined")
    should_reply = bool(getattr(value, "should_reply", False))
    if should_reply:
        memory_focus = getattr(value, "memory_focus", None)
        return direct_reply_activation_decision(
            memory_focus=memory_focus if isinstance(memory_focus, list) else None,
            reason=str(getattr(value, "reason", direct_reason) or direct_reason),
        )
    return ReplyActivationDecision(should_reply=False, reason=str(getattr(value, "reason", "activation probe declined") or "activation probe declined"))


def _memory_subjects_by_key(subjects: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {subject["key"]: subject for subject in subjects if subject.get("key")}


def _activation_memory_query(
    *,
    message_text: str,
    additional_context: list[str],
    sender_display: str,
    decision: ReplyActivationDecision,
) -> str:
    parts = [decision.rag_query_hint, decision.conversation_intent, message_text]
    for line in additional_context:
        if line.startswith(("replied_to_content:", "message_relation:", "input_type:", "sticker_")):
            parts.append(line)
    parts.append(sender_display)
    return "\n".join(part.strip() for part in parts if part and part.strip())


async def _load_group_reply_memory_context(
    *,
    chat_id: int,
    memory_subjects: list[dict[str, str]],
    decision: ReplyActivationDecision,
    query_text: str,
) -> list[str]:
    context_lines: list[str] = []

    try:
        global_memory = await get_global_memory_context(
            chat_id,
            query_text=query_text,
            intent=decision.conversation_intent,
        )
        if global_memory:
            context_lines.append(global_memory)
    except Exception as exc:
        logger.exception("Failed to read global memory context: %s", exc)

    subjects_by_key = _memory_subjects_by_key(memory_subjects)
    focus_keys = decision.memory_focus or (["sender"] if "sender" in subjects_by_key and decision.should_reply else [])
    for focus_key in focus_keys[:3]:
        subject = subjects_by_key.get(focus_key)
        if not subject:
            continue
        telegram_user_key = subject.get("telegram_user_key")
        if not telegram_user_key:
            continue
        try:
            personal_memory = await get_personal_memory_context(
                telegram_user_key,
                query_text=query_text,
                intent=decision.conversation_intent,
            )
            if personal_memory:
                context_lines.append(f"user_memory_key[{focus_key}]: {telegram_user_key}")
                context_lines.append(f"user_personal_memory: subject={focus_key}; display={subject.get('display') or telegram_user_key}\n{personal_memory}")
        except Exception as exc:
            logger.exception("Failed to read personal memory for %s: %s", telegram_user_key, exc)
    return context_lines


def _sticker_reply_enabled() -> bool:
    return get_runtime_bool("STICKER_REPLY_ENABLED", True)


def _sticker_reply_candidate_limit() -> int:
    return max(0, get_runtime_int("STICKER_REPLY_CANDIDATE_LIMIT", 12))


def _build_sticker_reply_query(
    *,
    message_text: str,
    ai_reply: str,
    additional_context: Optional[list[str]] = None,
) -> str:
    parts = [message_text, ai_reply]
    for line in additional_context or []:
        if line.startswith(("input_type:", "sticker_", "replied_to_content:", "message_relation:")):
            parts.append(line)
    return "\n".join(part.strip() for part in parts if part and part.strip())


def _sticker_candidate_prompt_payload(candidate) -> dict:
    return {
        "file_unique_id": candidate.file_unique_id,
        "emoji": candidate.emoji,
        "set_name": candidate.set_name,
        "description": candidate.description,
        "tags": getattr(candidate, "tags", []),
        "mood": getattr(candidate, "mood", None),
        "safe_for_reply": getattr(candidate, "safe_for_reply", True),
        "use_count": getattr(candidate, "use_count", 0),
        "last_used_at": getattr(candidate, "last_used_at", None),
        "is_animated": candidate.is_animated,
        "is_video": candidate.is_video,
    }


async def _choose_sticker_reply_candidate(
    *,
    update: Update,
    context: Optional[ContextTypes.DEFAULT_TYPE],
    message_text: str,
    ai_reply: str,
    additional_context: Optional[list[str]],
    runtime_state: Optional[list[str]],
) -> Optional[tuple[object, bool]]:
    if not _sticker_reply_enabled() or not update.message or context is None:
        return None
    if getattr(context, "bot", None) is None:
        return None

    limit = _sticker_reply_candidate_limit()
    if limit <= 0:
        return None

    try:
        query_text = _build_sticker_reply_query(
            message_text=message_text,
            ai_reply=ai_reply,
            additional_context=additional_context,
        )
        candidates = await find_sticker_reply_candidates(query_text, limit=limit)
        if not candidates:
            return None

        candidate_payloads = [_sticker_candidate_prompt_payload(candidate) for candidate in candidates]
        choice = await choose_reply_sticker(
            latest_message=message_text,
            reply_text=ai_reply,
            sticker_candidates=candidate_payloads,
            additional_context=additional_context,
            runtime_state=runtime_state,
        )
        if not choice:
            return None

        if isinstance(choice, str):
            selected_unique_id = choice
            send_text = True
        else:
            selected_unique_id = getattr(choice, "file_unique_id", None)
            send_text = bool(getattr(choice, "send_text", True))
        if not selected_unique_id:
            return None

        selected = next((candidate for candidate in candidates if candidate.file_unique_id == selected_unique_id), None)
        if not selected or not selected.file_id:
            return None

        return selected, send_text
    except Exception as exc:
        logger.exception("Error choosing sticker reply: %s", exc)
        return None


async def _send_sticker_reply(
    *,
    update: Update,
    chat_id: int,
    sender_display: str,
    selected,
) -> bool:
    if not update.message or not getattr(selected, "file_id", None):
        return False

    try:
        sent_message = await update.message.reply_sticker(sticker=selected.file_id)
    except Exception as exc:
        logger.exception("Error sending sticker reply: %s", exc)
        return False

    try:
        await record_sticker_reply_usage(selected.file_unique_id)
    except Exception as exc:
        logger.exception("Sticker reply sent but usage logging failed: %s", exc)

    try:
        await add_message(
            chat_id=chat_id,
            username="mioo_bot",
            content=f"sticker reply: {selected.description}",
            telegram_message_id=getattr(sent_message, "message_id", None),
            reply_to_telegram_message_id=getattr(update.message, "message_id", None),
            reply_to_username=sender_display,
        )
    except Exception as exc:
        logger.exception("Sticker reply sent but history logging failed: %s", exc)

    logger.info("Sent sticker reply %s from set %s", selected.file_unique_id, selected.set_name or "(none)")
    return True


def _fallback_sticker_description(sticker) -> str:
    parts: list[str] = []
    if getattr(sticker, "is_animated", False):
        parts.append("animated sticker")
    elif getattr(sticker, "is_video", False):
        parts.append("video sticker")
    else:
        parts.append("sticker")

    emoji = getattr(sticker, "emoji", None)
    if emoji:
        parts.append(f"emoji {emoji}")

    set_name = getattr(sticker, "set_name", None)
    if set_name:
        parts.append(f"from set {set_name}")

    return ", ".join(parts)


async def _describe_sticker(update: Update, context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str, list[str], Optional[str], bool]:
    if not update.message or not update.message.sticker:
        return "sticker", "fallback", [], None, True

    sticker = update.message.sticker
    file_unique_id = getattr(sticker, "file_unique_id", None) or ""
    cached_description = await get_sticker_text(file_unique_id)
    if cached_description:
        return cached_description, "cache", [], None, True

    visual_file_id = getattr(sticker, "file_id", None)
    extension = "webp"
    source = "sticker_file"

    if getattr(sticker, "is_animated", False) or getattr(sticker, "is_video", False):
        thumbnail = getattr(sticker, "thumbnail", None)
        if thumbnail and getattr(thumbnail, "file_id", None):
            visual_file_id = thumbnail.file_id
            extension = "jpg"
            source = "thumbnail"
        else:
            visual_file_id = None

    understanding: Optional[StickerUnderstanding] = None
    output_path = None
    try:
        if visual_file_id and context.bot:
            output_path = _build_output_path("sticker", update.message.message_id, extension=extension)
            tg_file = await context.bot.get_file(visual_file_id)
            await tg_file.download_to_drive(custom_path=output_path)
            understanding = await sticker_to_understanding(
                output_path,
                emoji=getattr(sticker, "emoji", None),
                set_name=getattr(sticker, "set_name", None),
            )
    except Exception as exc:
        logger.warning("Sticker understanding failed for %s: %s", file_unique_id or "(unknown)", exc)
    finally:
        _remove_file_if_exists(output_path)

    if understanding:
        description = understanding.description
        tags = understanding.tags
        mood = understanding.mood
        safe_for_reply = understanding.safe_for_reply
    else:
        description = _fallback_sticker_description(sticker)
        tags = []
        mood = None
        safe_for_reply = True
        source = "fallback"

    await upsert_sticker_text(
        file_unique_id=file_unique_id,
        file_id=getattr(sticker, "file_id", None),
        emoji=getattr(sticker, "emoji", None),
        set_name=getattr(sticker, "set_name", None),
        description=description,
        description_source=source,
        tags=tags,
        mood=mood,
        safe_for_reply=safe_for_reply,
        is_animated=bool(getattr(sticker, "is_animated", False)),
        is_video=bool(getattr(sticker, "is_video", False)),
    )
    return description, source, tags, mood, safe_for_reply


async def _handle_twitter_media_message(
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    video_url: str,
    sender_display: str,
    status_message,
    delete_source_message: bool = True,
) -> bool:
    """Handle Twitter/X media. Returns True when request is fully handled."""
    if not update.message or not update.effective_chat:
        return False

    twitter_downloader = TwitterDownloader()
    media_list, text_dict = await asyncio.to_thread(twitter_downloader.extract_twitter_media, video_url)

    tweet_text = summarize_tweet_text(text_dict)
    raw_message_text = (getattr(update.message, "text", None) or getattr(update.message, "caption", None) or video_url).strip()

    image_medias = [media for media_type, media in media_list if media_type == 'pic']
    video_medias = [media for media_type, media in media_list if media_type == 'vid']
    gif_medias = [media for media_type, media in media_list if media_type == 'gif']
    raw_text_caption = (tweet_text[:900] + "...") if len(tweet_text) > 900 else tweet_text
    text_caption = format_tweet_text_for_reply(raw_text_caption, video_url)

    if not image_medias and not video_medias and not gif_medias and not text_caption:
        raise ValueError(
            "Could not extract video, images, or text from this tweet. "
            "It may be deleted, protected, region-restricted, or blocked by auth/cookie settings."
        )

    sender_user = getattr(update, "effective_user", None)
    reply_to_message = getattr(update.message, "reply_to_message", None)
    try:
        await add_message(
            chat_id=update.effective_chat.id,
            username=sender_display,
            content=_build_twitter_history_message(
                raw_message_text=raw_message_text,
                twitter_url=video_url,
                tweet_text=tweet_text,
                image_count=len(image_medias),
                video_count=len(video_medias),
                gif_count=len(gif_medias),
            ),
            telegram_user_key=_telegram_user_key_from_user(sender_user),
            telegram_message_id=getattr(update.message, "message_id", None),
            reply_to_telegram_message_id=getattr(reply_to_message, "message_id", None) if reply_to_message else None,
            reply_to_username=_display_name_from_user(getattr(reply_to_message, "from_user", None)) if reply_to_message else None,
        )
    except Exception:
        logger.exception("Failed to persist parsed Twitter/X content for %s", video_url)

    if len(image_medias) > 1:
        album_caption = build_twitter_caption(text_caption, sender_display, video_url)
        media_group = []
        for index, image_bytes in enumerate(image_medias, start=1):
            image_buffer = io.BytesIO(image_bytes)
            image_buffer.name = f"tweet_image_{index}.jpg"
            is_first_image = index == 1
            media_group.append(
                InputMediaPhoto(
                    media=image_buffer,
                    caption=album_caption if is_first_image else None,
                    parse_mode=ParseMode.HTML if is_first_image else None,
                )
            )
        await context.bot.send_media_group(
            chat_id=update.effective_chat.id,
            media=media_group,
            ##reply_to_message_id=update.message.message_id,
        )
    elif len(image_medias) == 1:
        image_caption = build_twitter_caption(text_caption, sender_display, video_url)
        image_buffer = io.BytesIO(image_medias[0])
        image_buffer.name = "tweet_image_1.jpg"
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=image_buffer,
            ##reply_to_message_id=update.message.message_id,
            caption=image_caption,
            parse_mode=ParseMode.HTML,
        )

    if video_medias:
        for index, video_bytes in enumerate(video_medias, start=1):
            video_buffer = io.BytesIO(video_bytes)
            video_buffer.name = f"tweet_video_{index}.mp4"
            is_first_video = index == 1
            video_caption = build_twitter_caption(text_caption, sender_display, video_url) if is_first_video else None
            await context.bot.send_video(
                chat_id=update.effective_chat.id,
                video=video_buffer,
                ##reply_to_message_id=update.message.message_id if is_first_video else None,
                caption=video_caption,
                parse_mode=ParseMode.HTML if is_first_video else None,
            )

    for index, gif_bytes in enumerate(gif_medias, start=1):
        gif_buffer = io.BytesIO(gif_bytes)
        gif_buffer.name = f"tweet_gif_{index}.mp4"
        is_primary_document = not image_medias and not video_medias and index == 1
        gif_caption = build_twitter_caption(text_caption, sender_display, video_url) if is_primary_document else None
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=gif_buffer,
            ##reply_to_message_id=update.message.message_id if is_primary_document else None,
            caption=gif_caption,
            parse_mode=ParseMode.HTML if gif_caption else None,
        )

    if text_caption and not image_medias and not video_medias and not gif_medias:
        text_body = build_twitter_caption(text_caption, sender_display, video_url)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=text_body,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    await _delete_message_if_exists(status_message)
    if delete_source_message:
        await _delete_message_if_exists(update.message)
    return True


async def _send_zhihu_image_media(*, context, chat_id, image_media) -> None:
    """Send extracted Zhihu images as photo albums, with large-image fallback."""
    photo_media = []
    document_media = []
    for item in image_media or []:
        payload = item.get("content") if isinstance(item, dict) else None
        if not payload:
            continue
        filename = str(item.get("filename") or "zhihu_image.jpg") if isinstance(item, dict) else "zhihu_image.jpg"
        image_buffer = io.BytesIO(payload)
        image_buffer.name = filename
        content_type = str(item.get("content_type") or "").split(";", 1)[0].lower() if isinstance(item, dict) else ""
        photo_compatible = not content_type or content_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}
        if len(payload) <= 10 * 1024 * 1024 and photo_compatible:
            photo_media.append(image_buffer)
        else:
            document_media.append(image_buffer)

    for start in range(0, len(photo_media), 10):
        batch = photo_media[start:start + 10]
        if len(batch) == 1:
            await context.bot.send_photo(
                chat_id=chat_id,
                photo=batch[0],
            )
            continue
        await context.bot.send_media_group(
            chat_id=chat_id,
            media=[InputMediaPhoto(media=image_buffer) for image_buffer in batch],
        )

    for image_buffer in document_media:
        await context.bot.send_document(
            chat_id=chat_id,
            document=image_buffer,
        )


async def _handle_zhihu_link_message(
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    video_url: str,
    sender_display: str,
    status_message,
    delete_source_message: bool = True,
) -> bool:
    """Handle Zhihu answers, articles, posts, and question links."""
    if not update.message or not update.effective_chat:
        return False

    zhihu_result = await asyncio.to_thread(parse_zhihu_link, video_url)
    raw_message_text = (getattr(update.message, "text", None) or getattr(update.message, "caption", None) or video_url).strip()

    content_type = str(zhihu_result.get("content_type") or "answer")
    question = str(zhihu_result.get("title") or zhihu_result.get("question") or "(无标题)")
    author = str(zhihu_result.get("author") or "(匿名)")
    author_url = str(zhihu_result.get("author_url") or "")
    content = str(zhihu_result.get("content") or "（无内容）")
    time_text = str(zhihu_result.get("time") or "未知")
    image_urls = [str(url) for url in (zhihu_result.get("image_urls") or []) if str(url).strip()]
    image_media = []
    if image_urls:
        try:
            image_media = await asyncio.to_thread(download_zhihu_image_media, image_urls)
        except Exception:
            logger.exception("Failed to download Zhihu images for %s", video_url)

    sender_user = getattr(update, "effective_user", None)
    reply_to_message = getattr(update.message, "reply_to_message", None)
    try:
        await add_message(
            chat_id=update.effective_chat.id,
            username=sender_display,
            content=_build_zhihu_history_message(
                raw_message_text=raw_message_text,
                zhihu_url=video_url,
                question=question,
                author=author,
                content=content,
                content_type=content_type,
                image_count=len(image_urls),
            ),
            telegram_user_key=_telegram_user_key_from_user(sender_user),
            telegram_message_id=getattr(update.message, "message_id", None),
            reply_to_telegram_message_id=getattr(reply_to_message, "message_id", None) if reply_to_message else None,
            reply_to_username=_display_name_from_user(getattr(reply_to_message, "from_user", None)) if reply_to_message else None,
        )
    except Exception:
        logger.exception("Failed to persist parsed Zhihu content for %s", video_url)

    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=_build_zhihu_reply_text(
            zhihu_url=video_url,
            question=question,
            author=author,
            author_url=author_url,
            content=content,
            sender_display=sender_display,
            time_text=time_text,
            content_type=content_type,
        ),
        disable_web_page_preview=True,
    )

    if image_media:
        await _send_zhihu_image_media(
            context=context,
            chat_id=update.effective_chat.id,
            image_media=image_media,
        )

    await _delete_message_if_exists(status_message)
    if delete_source_message:
        await _delete_message_if_exists(update.message)
    return True


async def _render_and_send_image_from_markdown(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    markdown_input: str,
    output_file_path: str,
) -> None:
    if not update.message or not update.effective_chat:
        return

    await md_to_image(md_text=markdown_input, output_path=output_file_path, theme='formal_code')
    with open(output_file_path, 'rb') as photo:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=photo,
            reply_to_message_id=update.message.message_id,
        )


async def _handle_md2jpg_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    markdown_input: str,
) -> None:
    if not update.message:
        return

    if not markdown_input:
        await update.message.reply_text("Please provide some markdown content inside the triple quotes.")
        return

    output_file_path = _build_output_path("md", update.message.message_id)
    status_message = None
    try:
        status_message = await update.message.reply_text("Generating your image, please wait a moment...")
        await _render_and_send_image_from_markdown(update, context, markdown_input, output_file_path)
        await _delete_message_if_exists(status_message)
    except Exception as e:
        logger.error(f"Error during image generation or sending: {e}")
        await update.message.reply_text("Sorry, I encountered an error while creating your image.")
        await _delete_message_if_exists(status_message)
    finally:
        _remove_file_if_exists(output_file_path)


async def _handle_text2jpg_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    plain_text_input: str,
) -> None:
    if not update.message:
        return

    if not plain_text_input:
        await update.message.reply_text("Please provide some text content inside the triple quotes.")
        return

    output_file_path = _build_output_path("text", update.message.message_id)
    status_message = None
    try:
        status_message = await update.message.reply_text("Converting your text to markdown, please wait a moment...")
        generated_markdown = await plain_text_to_markdown(plain_text_input)
        await status_message.edit_text("Generating your image from markdown, please wait a moment...")
        await _render_and_send_image_from_markdown(update, context, generated_markdown, output_file_path)
        await _delete_message_if_exists(status_message)
    except Exception as e:
        logger.error(f"Error during image generation or sending: {e}")
        await update.message.reply_text("Sorry, I encountered an error while creating your image.")
        await _delete_message_if_exists(status_message)
    finally:
        _remove_file_if_exists(output_file_path)


# -------- Telegram Bot Handlers --------


def _build_help_text() -> str:
    return (
        "MioBot help\n\n"
        "General commands:\n"
        "/start - Show a short intro\n"
        "/help - Show this feature list\n\n"
        "Text to image:\n"
        "/md2jpg ,,,...,,, - Render Markdown as an image\n"
        "/text2jpg ,,,...,,, - Convert plain text to Markdown, then render it\n"
        "Upload a .txt or .md file - Render it as an image\n\n"
        "Media handling:\n"
        "Send a YouTube, Bilibili, Twitter/X, or Zhihu link - Download media or parse supported text (Zhihu answers, articles, posts, and questions)\n\n"
        "Group AI replies:\n"
        "In group chats, text/photo/sticker messages can trigger contextual replies. Direct triggers include replying to the bot, mentioning @BotUsername, or saying mioo / 小小宫.\n\n"
        "Extra commands:\n"
        "/med2jpg <request> - Generate a prescription-style MED image\n"
        "/crypto - Show crypto prices and Allez APR snapshots\n\n"
        "Private admin memory tools:\n"
        "/memory_help - Show memory admin commands\n"
        "/memories - List users with memory/history\n"
        "/memory_audit [limit] - Inspect malformed memory summaries (0 = full scan)\n"
        "/memory <user> - View one user's memory\n"
        "/memory_search <keyword> - Search summaries and facts\n"
        "/memory_refresh <user> - Regenerate memory from history\n"
        "/memory_set <user> <text> - Replace summary memory\n"
        "/memory_candidates [user] - Review pending memory candidates\n"
        "/memory_accept <id> / /memory_reject <id> - Accept or reject a candidate\n"
        "/memory_fact_set <id> <text> / /memory_fact_delete <id> - Edit or archive facts\n\n"
        "Admin memory commands only work in private chat for users listed in TELEGRAM_ADMIN_USER_IDS."
    )


def _build_twitter_history_message(
    *,
    raw_message_text: str,
    twitter_url: str,
    tweet_text: str,
    image_count: int,
    video_count: int,
    gif_count: int,
    max_tweet_chars: int = 1500,
) -> str:
    user_comment = " ".join((raw_message_text or "").replace(twitter_url, " ").split()).strip()
    normalized_tweet_text = " ".join((tweet_text or "").split()).strip()
    if len(normalized_tweet_text) > max_tweet_chars:
        normalized_tweet_text = normalized_tweet_text[: max_tweet_chars - 1].rstrip() + "…"

    media_parts: list[str] = []
    if image_count:
        media_parts.append(f"{image_count} image(s)")
    if video_count:
        media_parts.append(f"{video_count} video(s)")
    if gif_count:
        media_parts.append(f"{gif_count} gif(s)")
    if not media_parts:
        media_parts.append("text-only")

    lines = [
        f"shared_twitter_link: {twitter_url}",
        f"shared_twitter_media: {', '.join(media_parts)}",
    ]
    if user_comment:
        lines.append(f"user_comment: {user_comment}")
    if normalized_tweet_text:
        lines.append(f"tweet_text: {normalized_tweet_text}")
    return "\n".join(lines)


def _build_zhihu_history_message(
    *,
    raw_message_text: str,
    zhihu_url: str,
    question: str,
    author: str,
    content: str,
    content_type: str = "answer",
    image_count: int = 0,
    max_content_chars: int = 1500,
) -> str:
    user_comment = " ".join((raw_message_text or "").replace(zhihu_url, " ").split()).strip()
    normalized_content = " ".join((content or "").split()).strip()
    if len(normalized_content) > max_content_chars:
        normalized_content = normalized_content[: max_content_chars - 1].rstrip() + "…"

    lines = [f"shared_zhihu_link: {zhihu_url}"]
    if image_count:
        lines.append(f"shared_zhihu_media: {image_count} image(s)")
    if content_type == "answer":
        lines.extend(
            [
                f"zhihu_question: {' '.join((question or '').split()).strip()}",
                f"zhihu_author: {' '.join((author or '').split()).strip()}",
            ]
        )
    else:
        lines.extend(
            [
                f"zhihu_type: {content_type}",
                f"zhihu_title: {' '.join((question or '').split()).strip()}",
                f"zhihu_author: {' '.join((author or '').split()).strip()}",
            ]
        )
    if user_comment:
        lines.append(f"user_comment: {user_comment}")
    if normalized_content:
        lines.append(
            f"zhihu_answer: {normalized_content}"
            if content_type == "answer"
            else f"zhihu_content: {normalized_content}"
        )
    return "\n".join(lines)


def _build_zhihu_reply_text(
    *,
    zhihu_url: str,
    question: str,
    author: str,
    author_url: str,
    content: str,
    sender_display: str,
    time_text: str,
    content_type: str = "answer",
    max_content_chars: int = 3200,
) -> str:
    author_line = author or "(匿名)"
    if author_url:
        author_line = f"{author_line} (@{author_url})"

    question_line = (question or "(无标题)").strip() or "(无标题)"
    trimmed_content = (content or "（无内容）").strip() or "（无内容）"
    if len(trimmed_content) > max_content_chars:
        trimmed_content = trimmed_content[: max_content_chars - 1].rstrip() + "…"

    content_labels = {
        "article": "文章",
        "post": "想法/帖子",
        "question": "问题",
    }
    title_prefix = f"[知乎{content_labels[content_type]}] " if content_type in content_labels else ""

    message = "\n".join(
        [
            f"{title_prefix}{question_line}",
            "",
            trimmed_content,
            f"-- {author_line} · {time_text or '未知'}",
            "",
            zhihu_url,
            f"Requested by: {sender_display}",
        ]
    )
    return _truncate_message_text(message)


# Start command handler
# This handler sends a welcome message when the /start command is issued.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    if not update.message:
        return

    await update.message.reply_text(
        "Hi! I can render text to images, download media links, and join group chats with contextual replies. Send /help to see all features."
    )


async def handle_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show all currently supported bot features."""
    if not update.message:
        return

    await update.message.reply_text(_build_help_text())


async def handle_md2jpg_and_text2jpg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /md2jpg and /text2jpg commands to generate images."""
    if not update.message or not update.message.text:
        return
    logger.info(f"Received text for rendering: {update.message.text if update.message else 'No message text'}")
    message_text = update.message.text

    markdown_input = _match_command_payload(message_text, MD2JPG_REGEX)
    if markdown_input is not None:
        await _handle_md2jpg_request(update, context, markdown_input)

    plain_text_input = _match_command_payload(message_text, TEXT2JPG_REGEX)
    if plain_text_input is not None:
        await _handle_text2jpg_request(update, context, plain_text_input)


# Handle .txt or .md files to render as image
async def handle_text_or_markdown_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle .txt or .md files to render as image."""
    if not update.message or not update.message.document:
        return

    document_file = update.message.document
    file_name = document_file.file_name
    if not file_name:
        return

    extension = os.path.splitext(file_name)[1].lower()
    if extension not in {'.txt', '.md'}:
        return
    try:
        declared_size = int(getattr(document_file, "file_size", 0) or 0)
    except (TypeError, ValueError):
        declared_size = 0
    if declared_size > MAX_RENDER_DOCUMENT_BYTES:
        await update.message.reply_text(
            f"This document is too large to render (limit: {MAX_RENDER_DOCUMENT_BYTES // (1024 * 1024)} MB)."
        )
        return

    is_already_markdown = extension == '.md'
    # Do not use the user-controlled Telegram filename as a local path.
    downloaded_path = _build_output_path("source", update.message.message_id, extension=extension.lstrip('.'))
    output_file_path = _build_output_path("file", update.message.message_id)
    status_message = None
    try:
        tg_file = await document_file.get_file()
        await tg_file.download_to_drive(custom_path=downloaded_path)
        if os.path.getsize(downloaded_path) > MAX_RENDER_DOCUMENT_BYTES:
            raise ValueError(f"Document exceeds {MAX_RENDER_DOCUMENT_BYTES} byte limit.")

        with open(downloaded_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        try:
            status_message = await update.message.reply_text("Converting your file to markdown, please wait a moment...")

            if not is_already_markdown:
                generated_markdown = await plain_text_to_markdown(file_content)
            else:
                generated_markdown = file_content

            await status_message.edit_text("Generating your image from markdown, please wait a moment...")

            await _render_and_send_image_from_markdown(update, context, generated_markdown, output_file_path)
            await _delete_message_if_exists(status_message)
        except Exception as e:
            logger.error(f"Error during image generation or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while creating your image.")
            await _delete_message_if_exists(status_message)
    except Exception as exc:
        logger.warning("Rejected or failed to read document render request: %s", exc)
        await update.message.reply_text("Sorry, I could not read that text document safely.")
    finally:
        _remove_file_if_exists(output_file_path)
        _remove_file_if_exists(downloaded_path)


# Handle Group AI Replies
async def handle_group_ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle group messages and occasionally reply using AI."""
    if not update.message or not update.message.text:
        return

    await _handle_group_ai_reply_pipeline(update, update.message.text, context=context)


def _resolve_group_ai_sender(update: Update):
    message = getattr(update, "message", None)
    return getattr(message, "from_user", None) or getattr(update, "effective_user", None)


async def _handle_group_ai_reply_pipeline(
    update: Update,
    message_text: str,
    *,
    additional_context: Optional[list[str]] = None,
    context: Optional[ContextTypes.DEFAULT_TYPE] = None,
) -> None:
    """Shared group-reply flow for text-like content."""
    if not update.message:
        return

    if not update.effective_chat or not update.effective_user:
        return

    sender_user = _resolve_group_ai_sender(update)
    if sender_user and getattr(sender_user, "is_bot", False):
        logger.info("Skipping group AI reply pipeline for bot sender: %s", getattr(sender_user, "username", None) or getattr(sender_user, "id", None))
        return

    chat_id = update.effective_chat.id
    sender_display = _display_name_from_user(sender_user)
    telegram_user_key = _telegram_user_key_from_user(sender_user)
    stored_message_text, relation_context = _build_reply_relation_payload(update, message_text)
    merged_additional_context = list(additional_context or []) + relation_context
    replied_message = update.message.reply_to_message
    reply_to_tg_id = getattr(replied_message, "message_id", None) if replied_message else None
    reply_to_username = _display_name_from_user(getattr(replied_message, "from_user", None)) if replied_message else None
    raw_user_text = (getattr(update.message, "text", None) or getattr(update.message, "caption", None) or "").strip()
    memory_subjects = _build_group_reply_memory_subjects(
        update,
        sender_user=sender_user,
        sender_display=sender_display,
        telegram_user_key=telegram_user_key,
    )

    if telegram_user_key:
        _schedule_personal_memory_refresh(context, telegram_user_key, sender_display)

    logger.info("Adding message to history for %s", sender_display)
    message_db_id = await add_message(
        chat_id=chat_id,
        username=sender_display,
        content=stored_message_text,
        telegram_user_key=telegram_user_key,
        telegram_message_id=getattr(update.message, "message_id", None),
        reply_to_telegram_message_id=reply_to_tg_id,
        reply_to_username=reply_to_username,
    )
    _schedule_memory_candidate_extraction(
        context,
        telegram_user_key=telegram_user_key,
        message_text=stored_message_text,
        message_db_id=message_db_id,
        latest_display_name=sender_display,
    )

    bot_user_id = getattr(getattr(context, "bot", None), "id", None)
    is_reply_to_bot = _is_reply_to_this_bot(update, TELEGRAM_BOT_USERNAME, bot_user_id=bot_user_id)
    trigger_type = "reply_to_bot" if is_reply_to_bot else _classify_group_reply_trigger(raw_user_text, TELEGRAM_BOT_USERNAME)
    is_mentioned = trigger_type in {"username_mention", "alias_mention"}
    is_directly_addressed = is_reply_to_bot or is_mentioned
    runtime_state = _build_group_reply_runtime_state(
        sender_display=sender_display,
        trigger_type=trigger_type,
        direct_addressed=is_directly_addressed,
    )

    if is_directly_addressed:
        logger.info("User %s directly triggered the bot via %s.", sender_display, trigger_type)
        activation_decision = direct_reply_activation_decision(reason=trigger_type)
    else:
        probe_history_messages, _ = await get_prompt_context_parts(chat_id, query="")
        probe_result = await should_activate_reply(
            message_history=probe_history_messages,
            additional_context=merged_additional_context or None,
            runtime_state=runtime_state,
            is_reply_to_bot=is_reply_to_bot,
            is_mentioned=is_mentioned,
            available_memory_subjects=memory_subjects,
            return_decision=True,
        )
        activation_decision = _coerce_activation_decision(probe_result, direct_reason="activation probe approved")
        if not activation_decision.should_reply:
            return

    memory_query = _activation_memory_query(
        message_text=message_text,
        additional_context=merged_additional_context,
        sender_display=sender_display,
        decision=activation_decision,
    )
    merged_additional_context.extend(
        await _load_group_reply_memory_context(
            chat_id=chat_id,
            memory_subjects=memory_subjects,
            decision=activation_decision,
            query_text=memory_query,
        )
    )
    merged_additional_context.extend(reply_activation_decision_context_lines(activation_decision))

    rag_query = _build_rag_query_from_message(
        activation_decision.rag_query_hint or message_text,
        additional_context=merged_additional_context,
        sender_display=sender_display,
    ) if activation_decision.needs_rag else ""
    history_messages, rag_related_messages = await get_prompt_context_parts(chat_id, query=rag_query)

    ai_reply = await generate_group_reply(
        message_history=history_messages,
        rag_related_messages=rag_related_messages,
        additional_context=merged_additional_context or None,
        is_reply_to_bot=is_reply_to_bot,
        is_mentioned=is_mentioned,
        runtime_state=runtime_state,
    )

    if ai_reply:
        try:
            sticker_reply = await _choose_sticker_reply_candidate(
                update=update,
                context=context,
                message_text=message_text,
                ai_reply=ai_reply,
                additional_context=merged_additional_context or None,
                runtime_state=runtime_state,
            )
            if sticker_reply:
                selected_sticker, send_text = sticker_reply
                if not send_text:
                    sent_sticker = await _send_sticker_reply(
                        update=update,
                        chat_id=chat_id,
                        sender_display=sender_display,
                        selected=selected_sticker,
                    )
                    if sent_sticker:
                        return

            sent_message = await update.message.reply_text(ai_reply)
            await add_message(
                chat_id=chat_id,
                username="mioo_bot",
                content=ai_reply,
                telegram_message_id=getattr(sent_message, "message_id", None),
                reply_to_telegram_message_id=getattr(update.message, "message_id", None),
                reply_to_username=sender_display,
            )
            if sticker_reply:
                selected_sticker, _ = sticker_reply
                await _send_sticker_reply(
                    update=update,
                    chat_id=chat_id,
                    sender_display=sender_display,
                    selected=selected_sticker,
                )
        except Exception as e:
            logger.error(f"Error sending AI reply: {e}")


def _schedule_background_task(context: Optional[ContextTypes.DEFAULT_TYPE], coroutine) -> None:
    application = getattr(context, "application", None)
    if application is not None and hasattr(application, "create_task"):
        application.create_task(coroutine)
        return
    task = asyncio.create_task(coroutine)
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


async def _extract_memory_candidate_background(
    *,
    telegram_user_key: str,
    message_text: str,
    message_db_id: Optional[int],
    latest_display_name: str,
) -> None:
    try:
        candidate_id = await extract_user_memory_candidate_from_message(
            telegram_user_key=telegram_user_key,
            message_text=message_text,
            message_id=message_db_id,
        )
        if candidate_id is not None:
            await refresh_user_memory_if_due(
                telegram_user_key=telegram_user_key,
                latest_display_name=latest_display_name,
            )
    except Exception as exc:
        logger.exception("Background memory candidate extraction failed for %s: %s", telegram_user_key, exc)


def _schedule_memory_candidate_extraction(
    context: Optional[ContextTypes.DEFAULT_TYPE],
    *,
    telegram_user_key: Optional[str],
    message_text: str,
    message_db_id: Optional[int],
    latest_display_name: str,
) -> None:
    if not telegram_user_key or message_db_id is None or not message_text.strip():
        return
    _schedule_background_task(
        context,
        _extract_memory_candidate_background(
            telegram_user_key=telegram_user_key,
            message_text=message_text,
            message_db_id=message_db_id,
            latest_display_name=latest_display_name,
        ),
    )


async def _refresh_user_memory_background(telegram_user_key: str, latest_display_name: str) -> None:
    try:
        await refresh_user_memory_if_due(
            telegram_user_key=telegram_user_key,
            latest_display_name=latest_display_name,
        )
    except Exception as exc:
        logger.exception("Background personal memory refresh failed for %s: %s", telegram_user_key, exc)
    finally:
        _USER_MEMORY_REFRESH_KEYS.discard(telegram_user_key)


def _schedule_personal_memory_refresh(
    context: Optional[ContextTypes.DEFAULT_TYPE],
    telegram_user_key: Optional[str],
    latest_display_name: str,
) -> None:
    if not telegram_user_key or telegram_user_key in _USER_MEMORY_REFRESH_KEYS:
        return
    _USER_MEMORY_REFRESH_KEYS.add(telegram_user_key)
    _schedule_background_task(
        context,
        _refresh_user_memory_background(
            telegram_user_key=telegram_user_key,
            latest_display_name=latest_display_name,
        ),
    )


async def _process_video_link_request(
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    video_url: str,
    sender_display: str,
    status_message,
    delete_source_message: bool = True,
) -> bool:
    cleanup_paths: set[str] = set()
    try:
        if is_zhihu_url(video_url):
            return await _handle_zhihu_link_message(
                update=update,
                context=context,
                video_url=video_url,
                sender_display=sender_display,
                status_message=status_message,
                delete_source_message=delete_source_message,
            )

        if is_twitter_status_url(video_url):
            return await _handle_twitter_media_message(
                update=update,
                context=context,
                video_url=video_url,
                sender_display=sender_display,
                status_message=status_message,
                delete_source_message=delete_source_message,
            )

        if not update.message or not update.effective_chat:
            return False

        output_file_name = f"{update.message.message_id}_{str(datetime.datetime.now().timestamp())}.mp4"
        output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

        video_title = await download_video_to_file(video_url, output_file_path)

        cleanup_paths.add(output_file_path)

        file_to_send_path = await compress_video_if_needed(output_file_path)
        cleanup_paths.add(file_to_send_path)

        await status_message.edit_text("Download completed successfully. Sending the video...")
        caption_url = await resolve_caption_url(video_url)
        video_caption = _truncate_caption_text(
            f'{video_title}\n<a href="{caption_url}">original link</a>\nRequested by: {sender_display}'
        )

        with open(file_to_send_path, 'rb') as video:
            await context.bot.send_video(
                chat_id=update.effective_chat.id,
                video=video,
                reply_to_message_id=update.message.message_id,
                caption=video_caption,
                parse_mode=ParseMode.HTML,
                supports_streaming=True,
            )

        await _delete_message_if_exists(status_message)
        if delete_source_message:
            await _delete_message_if_exists(update.message)
        return True
    except Exception as e:
        logger.exception("Error during video download or sending")
        message = getattr(update, "message", None)
        if message:
            await message.reply_text(_build_detailed_media_error_message(e))
        await _delete_message_if_exists(status_message)
        return False
    finally:
        for path in cleanup_paths:
            _remove_file_if_exists(path)


async def _process_video_link_batch(
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    video_urls: list[str],
    sender_display: str,
    status_messages: list,
    delete_source_message: bool = True,
) -> None:
    """Process all links from one message before deleting that source message."""
    all_succeeded = False
    try:
        async def _process_with_capacity(video_url: str, status_message) -> bool:
            async with _MEDIA_PROCESSING_SEMAPHORE:
                return await _process_video_link_request(
                    update=update,
                    context=context,
                    video_url=video_url,
                    sender_display=sender_display,
                    status_message=status_message,
                    delete_source_message=False,
                )

        results = await asyncio.gather(
            *(
                _process_with_capacity(video_url, status_message)
                for video_url, status_message in zip(video_urls, status_messages)
            ),
            return_exceptions=True,
        )
        all_succeeded = True
        for result in results:
            if isinstance(result, Exception) or result is not True:
                all_succeeded = False
                logger.error("One link in a mixed media message failed: %s", result)
    finally:
        if delete_source_message and all_succeeded:
            await _delete_message_if_exists(getattr(update, "message", None))


# Handle text messages: download media links or parse Zhihu links, else pass to group AI handler
async def handle_text_for_youtube_or_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages: download supported media links or parse Zhihu links, else pass to group AI handler."""
    if not update.message or not update.message.text:
        return

    if not update.effective_chat or not update.effective_user:
        return

    sender_display = _display_name_from_user(update.effective_user)
    message_text = update.message.text.strip()
    video_urls = extract_supported_links_from_message(update.message)
    # Preserve compatibility with integrations/tests that override the legacy
    # single-link helper while still processing every URL in normal messages.
    legacy_video_url = _extract_video_url(message_text)
    if legacy_video_url and legacy_video_url not in video_urls:
        video_urls.insert(0, legacy_video_url)

    if video_urls:
        if len(video_urls) > MAX_MEDIA_LINKS_PER_MESSAGE:
            total_links = len(video_urls)
            video_urls = video_urls[:MAX_MEDIA_LINKS_PER_MESSAGE]
            await update.message.reply_text(
                f"Processing the first {MAX_MEDIA_LINKS_PER_MESSAGE} of {total_links} supported links."
            )
        status_messages = []
        for index, video_url in enumerate(video_urls):
            status_text = (
                "Parsing your Zhihu link, please wait a moment..."
                if is_zhihu_answer_url(video_url)
                else "Parsing your Zhihu content, please wait a moment..."
                if is_zhihu_url(video_url)
                else "Downloading your video, please wait a moment..."
            )
            status_message = await update.message.reply_text(status_text)
            status_messages.append(status_message)
        _schedule_background_task(
            context,
            _process_video_link_batch(
                update=update,
                context=context,
                video_urls=video_urls,
                sender_display=sender_display,
                status_messages=status_messages,
                delete_source_message=True,
            ),
        )
        return
    else:
        if update.effective_chat.type in ['group', 'supergroup']:
            logger.info(f"Non-video-link message in group chat: {message_text}")
            await handle_group_ai_reply(update, context)
        return


async def handle_photo_for_group_ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram photos in groups by converting image content to text and reusing AI reply flow."""
    if not update.message or not update.message.photo:
        return
    if not _is_group_chat(update):
        return
    if getattr(_resolve_group_ai_sender(update), "is_bot", False):
        return

    # Captions can contain the same supported links as text messages.  Keep
    # the photo in the AI pipeline while extracting any linked post/video in
    # parallel; the link worker must not delete the source photo message.
    caption_links = extract_supported_links_from_message(update.message)[:MAX_MEDIA_LINKS_PER_MESSAGE]
    for caption_link in caption_links:
        status_text = (
            "Parsing your Zhihu link, please wait a moment..."
            if is_zhihu_answer_url(caption_link)
            else "Parsing your Zhihu content, please wait a moment..."
            if is_zhihu_url(caption_link)
            else "Downloading your video, please wait a moment..."
        )
        status_message = await update.message.reply_text(status_text)
        _schedule_background_task(
            context,
            _process_video_link_request(
                update=update,
                context=context,
                video_url=caption_link,
                sender_display=_display_name_from_user(_resolve_group_ai_sender(update)),
                status_message=status_message,
                delete_source_message=False,
            ),
        )

    photo = update.message.photo[-1]
    photo_path = _build_output_path("photo", update.message.message_id, extension="jpg")

    try:
        tg_file = await context.bot.get_file(photo.file_id)
        await tg_file.download_to_drive(custom_path=photo_path)

        image_text = await image_to_text(photo_path)
        caption_text = (update.message.caption or "").strip()

        if not image_text and not caption_text:
            return

        combined_parts = []
        if image_text:
            combined_parts.append(image_text)
        if caption_text:
            combined_parts.append(f"caption: {caption_text}")
        synthesized_text = "\n".join(combined_parts)

        await _handle_group_ai_reply_pipeline(
            update,
            synthesized_text,
            additional_context=[
                "input_type: image",
            ],
            context=context,
        )
    except Exception as e:
        logger.error(f"Error handling group image message: {e}")
    finally:
        _remove_file_if_exists(photo_path)


async def handle_sticker_for_group_ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram stickers in groups by caching a textual description and reusing group reply flow."""
    if not update.message or not update.message.sticker:
        return
    if not _is_group_chat(update):
        return
    if getattr(_resolve_group_ai_sender(update), "is_bot", False):
        return

    sticker = update.message.sticker
    description, description_source, tags, mood, safe_for_reply = await _describe_sticker(update, context)
    synthesized_text = f"sticker: {description}"

    await _handle_group_ai_reply_pipeline(
        update,
        synthesized_text,
        additional_context=[
            "input_type: sticker",
            f"sticker_emoji: {getattr(sticker, 'emoji', '(none)') or '(none)'}",
            f"sticker_set_name: {getattr(sticker, 'set_name', '(none)') or '(none)'}",
            f"sticker_cached: {str(description_source == 'cache').lower()}",
            f"sticker_description_source: {description_source}",
            f"sticker_description: {description}",
            f"sticker_tags: {', '.join(tags) if tags else '(none)'}",
            f"sticker_mood: {mood or '(none)'}",
            f"sticker_safe_for_reply: {str(safe_for_reply).lower()}",
        ],
        context=context,
    )


async def handle_crypto_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    '''
    the get_crypto_prices, get_Allez_APR, get_Allez_USDC_APR will return something like:
    {'SOL': 181.888, 'ETH': 3827.993, 'BTC': 398.588, 'USDC': 1.0, 'USDT': 1.0}
    {'name': 'Allez SOL', 'APR_24H': '11.65%', 'APR_7D': '6.33%', 'APR_30D': '5.78%', 'APR_90D': '5.88%', 'Total_Supply': '10.43M'}
    {'name': 'Allez USDC', 'APR_24H': '3.57%', 'APR_7D': '4.61%', 'APR_30D': '5.01%', 'APR_90D': '10.85%', 'Total_Supply': '59.94M'}'''
    if not update.message:
        return
    
    try:
        # prices = await get_Price(["BTC", "ETH", "SOL"])
        prices = await get_Price_Coinbase(["SOL", "USDC", "BTC", "ETH", "USDT"])
        # Sort of prices by key
        prices = dict(sorted(prices.items()))
        allez_sol_apr = await get_Allez_APR()
        allez_usdc_apr = await get_Allez_USDC_APR()

        price_lines = [f"{token}: ${price}" for token, price in prices.items()]
        price_message = "Current Crypto Prices:\n" + "\n".join(price_lines)

        allez_sol_lines = [f"{key}: {value}" for key, value in allez_sol_apr.items()]
        allez_sol_message = "\n\n <a href=\"https://kamino.com/lend/allez-sol\">Allez SOL</a> APR Info:\n" + "\n".join(allez_sol_lines)

        allez_usdc_lines = [f"{key}: {value}" for key, value in allez_usdc_apr.items()]
        allez_usdc_message = '\n\n <a href="https://kamino.com/lend/allez-usdc">Allez USDC</a> APR Info:\n' + "\n".join(allez_usdc_lines)

        full_message = price_message + allez_sol_message + allez_usdc_message

        await update.message.reply_text(full_message, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Error fetching crypto prices: {e}")
        await update.message.reply_text("Sorry, I encountered an error while fetching crypto prices.")


async def handle_medjpg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /med2jpg command to generate med image from text."""
    if not update.message or not update.message.text:
        return
    logger.info(f"Received text for MED rendering: {update.message.text if update.message else 'No message text'}")
    message_text = update.message.text
    output_file_path = _build_output_path("med", update.message.message_id)
    status_message = None
    try:
        await update.message.reply_text("Processing your MED image request...")
        json_prompt = await generate_med(message_text)
        if not json_prompt:
            raise ValueError("The model returned an empty MED JSON payload.")

        status_message = await update.message.reply_text("Generating your MED image, please wait a moment...")

        # Convert the generated prescription data straight to JPG
        jpg_path = await generate_jpg_from_med_json(json_prompt, output_file_path, raise_on_failure=True)
        if not jpg_path or not os.path.exists(jpg_path):
            raise FileNotFoundError(f"MED JPG not created at {jpg_path}")

        if not update.effective_chat:
            return

        with open(jpg_path, 'rb') as photo:
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=photo,
                reply_to_message_id=update.message.message_id
            )
        await _delete_message_if_exists(status_message)
    except Exception as e:
        logger.exception("Error during MED image generation or sending: %s", e)
        await update.message.reply_text(_build_med_error_message(e))
        await _delete_message_if_exists(status_message)
    finally:
        _remove_file_if_exists(output_file_path)


async def handle_application_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler for Telegram application-level exceptions."""
    err = context.error
    if isinstance(err, Conflict):
        logger.warning(
            "Telegram polling conflict: another bot instance is calling getUpdates. "
            "Stop other instances or switch to webhook mode."
        )
        return

    logger.exception("Unhandled telegram error", exc_info=err)


def register_handlers(application: Application) -> None:
    """Register all command and message handlers in one place."""
    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", handle_help))

    # Memory administration commands
    application.add_handler(CommandHandler("memory_help", handle_memory_admin_help, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memories", handle_memory_admin_list, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_audit", handle_memory_admin_audit, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory", handle_memory_admin_view, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_search", handle_memory_admin_search, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_refresh", handle_memory_admin_refresh, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_set", handle_memory_admin_set, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_candidates", handle_memory_admin_candidates, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_accept", handle_memory_admin_accept, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_reject", handle_memory_admin_reject, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_fact_set", handle_memory_admin_fact_set, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("memory_fact_delete", handle_memory_admin_fact_delete, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("global_memory", handle_global_memory_admin_view, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("global_memory_set", handle_global_memory_admin_set, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("global_memory_delete", handle_global_memory_admin_delete, filters=filters.ChatType.PRIVATE))
    application.add_handler(CommandHandler("webadmin_token", handle_webadmin_token, filters=filters.ChatType.PRIVATE))

    # Commands for rendering to image
    application.add_handler(CommandHandler("md2jpg", handle_md2jpg_and_text2jpg))
    application.add_handler(CommandHandler("text2jpg", handle_md2jpg_and_text2jpg))

    # Command for rendering med
    application.add_handler(CommandHandler("med2jpg", handle_medjpg))

    # Documents (.txt, .md)
    application.add_handler(MessageHandler(filters.Document.ALL, handle_text_or_markdown_document))

    # General text: YouTube downloads or group AI replies
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_for_youtube_or_group))

    # Group images: convert to text and pass through AI reply process
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo_for_group_ai_reply))

    # Group stickers: cache sticker descriptions and pass them through AI reply process
    application.add_handler(MessageHandler(filters.Sticker.ALL, handle_sticker_for_group_ai_reply))

    # Cryto info command
    application.add_handler(CommandHandler("crypto", handle_crypto_command))


async def _ensure_embedding_index_ready() -> None:
    report = await log_embedding_health_report()
    if not report.get("needs_reindex"):
        return

    logger.warning(
        "Detected legacy or drifted embeddings in %s. Reindexing automatically before bot startup.",
        report.get("db_file", "(unknown db)"),
    )
    result = await reindex_message_embeddings()
    logger.info(
        "Automatic embedding reindex finished. reindexed=%s signature=%s",
        result.get("reindexed"),
        result.get("signature"),
    )
    await log_embedding_health_report()


def _run_webadmin_server() -> None:
    from webadmin.app import main as webadmin_main

    webadmin_main()


def _start_webadmin_process() -> multiprocessing.Process:
    validate_webadmin_security_configuration()
    process = multiprocessing.Process(target=_run_webadmin_server, name="miobot-webadmin", daemon=True)
    process.start()
    logger.info("Started webadmin process pid=%s", process.pid)
    return process


def _stop_webadmin_process(process: Optional[multiprocessing.Process]) -> None:
    if process is None:
        return
    if process.is_alive():
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)


def main() -> None:
    """Start the bot."""

    asyncio.run(ensure_fastembed_ready())

    # Initialize the database
    init_db()
    asyncio.run(_ensure_embedding_index_ready())

    webadmin_process = _start_webadmin_process()

    try:
        # Create the Application and pass it your bot's token.
        application = Application.builder().token(TELEGRAM_BOT_KEY).read_timeout(30).write_timeout(30).build()

        register_handlers(application)
        application.add_error_handler(handle_application_error)

        # Run the bot until the user presses Ctrl-C
        application.run_polling()
    finally:
        _stop_webadmin_process(webadmin_process)


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Start the bot
    main()
