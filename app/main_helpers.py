"""Utility helpers extracted from main entrypoint logic."""

import os
import re
from typing import Optional

OUTPUT_DIR = "output"

# URL regex patterns
YOUTUBE_URL_REGEX = (
    r'(https?://)?(www\.)?'
    r'(youtube\.com/|youtu\.be/|youtube-nocookie\.com/)'
    r'(?:watch\?v=|embed/|v/|shorts/|live/)?'
    r'([a-zA-Z0-9_-]{11})'
)
BILIBILI_URL_REGEX = (
    r'(https?://)?(?:www\.|m\.)?'
    r'(bilibili\.com/|b23\.tv/)'
    r'(?:video/|watch\?bvid=)?'
    r'([A-Za-z0-9_-]{6,12})'
    r'(?:[/?#][^\s]*)?'
)
TWITTER_URL_REGEX = (
    # Keep protocol optional to match existing YouTube/Bilibili behavior.
    r'(https?://)?(?:www\.)?'
    r'(twitter\.com/|x\.com/)'
    r'[A-Za-z0-9_]+/status/\d+'
    r'(?:[/?#][^\s]*)?'
)

MD2JPG_REGEX = r'/md2jpg(?:@\w+)?\s*,,,(.*),,,'
TEXT2JPG_REGEX = r'/text2jpg(?:@\w+)?\s*,,,(.*),,,'

RAG_KEYWORD_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "and", "or", "but", "if", "then",
    "this", "that", "it", "as", "at", "by", "from", "about", "just", "very",
    "you", "your", "me", "my", "we", "our", "they", "their", "he", "she", "his", "her",
}


def _build_output_path(prefix: str, message_id: int, extension: str = "jpg") -> str:
    return os.path.join(OUTPUT_DIR, f"{prefix}_{message_id}.{extension}")


def _remove_file_if_exists(path) -> None:
    if path and os.path.exists(path):
        os.remove(path)


async def _delete_message_if_exists(message) -> None:
    if message:
        await message.delete()


def _extract_video_url(message_text: str) -> Optional[str]:
    youtube_match = re.search(YOUTUBE_URL_REGEX, message_text)
    bilibili_match = re.search(BILIBILI_URL_REGEX, message_text)
    twitter_match = re.search(TWITTER_URL_REGEX, message_text)

    if youtube_match:
        return youtube_match.group(0)
    if bilibili_match:
        return bilibili_match.group(0)
    if twitter_match:
        return twitter_match.group(0)
    return None


def _is_reply_to_this_bot(update, bot_username: Optional[str]) -> bool:
    message = getattr(update, "message", None)
    if not message or not message.reply_to_message:
        return False

    from_user = message.reply_to_message.from_user
    return bool(
        from_user
        and from_user.is_bot
        and from_user.username == bot_username
    )


def _classify_group_reply_trigger(message_text: Optional[str], bot_username: Optional[str]) -> str:
    text = (message_text or "").strip()
    if not text:
        return "ambient"

    normalized_username = (bot_username or "").strip().lstrip("@")
    if normalized_username:
        username_pattern = rf"(?<![A-Za-z0-9_])@{re.escape(normalized_username)}(?![A-Za-z0-9_])"
        if re.search(username_pattern, text, flags=re.IGNORECASE):
            return "username_mention"

    if re.search(r"(?<![A-Za-z0-9_])mioo(?![A-Za-z0-9_])", text, flags=re.IGNORECASE):
        return "alias_mention"

    if "小小宫" in text:
        return "alias_mention"

    return "ambient"


def _display_name_from_user(user) -> str:
    if not user:
        return "unknown_user @unknown"

    nickname = None
    if getattr(user, "full_name", None):
        nickname = str(user.full_name)
    elif getattr(user, "username", None):
        nickname = str(user.username)
    else:
        nickname = "unknown_user"

    username = getattr(user, "username", None)
    if username:
        handle = f"@{username}"
    else:
        user_id = getattr(user, "id", None)
        id_part = str(user_id) if user_id is not None else "unknown"
        handle = f"@[{id_part}]"

    return f"{nickname} {handle}"


def _telegram_user_key_from_user(user) -> Optional[str]:
    user_id = getattr(user, "id", None)
    if user_id is None:
        return None
    return f"tg_user:{user_id}"


def _single_line_text(text: Optional[str], *, max_chars: int = 240) -> str:
    value = (text or "").replace("\r\n", " ").replace("\n", " ").strip()
    if not value:
        return "[non-text message]"
    if len(value) > max_chars:
        return value[: max_chars - 1] + "…"
    return value


def _build_reply_relation_payload(update, message_text: str) -> tuple[str, list[str]]:
    """Return (stored_content, additional_context) with reply relation metadata."""
    if not update.message or not update.message.reply_to_message:
        return message_text, []

    current_message = update.message
    replied_message = update.message.reply_to_message
    current_author = _display_name_from_user(getattr(current_message, "from_user", None))
    replied_author = _display_name_from_user(getattr(replied_message, "from_user", None))
    current_message_id = getattr(current_message, "message_id", None)
    replied_message_id = getattr(replied_message, "message_id", None)
    replied_content = _single_line_text(getattr(replied_message, "text", None) or getattr(replied_message, "caption", None))
    current_content = _single_line_text(message_text)

    relation_context = [
        "message_relation: current message replies to another message",
        f"user_reply_relation: {current_author} replies to {replied_author}",
        f"message_reply_relation: message {current_message_id} replies to message {replied_message_id}",
        f"replied_to_author: {replied_author}",
        f"replied_to_content: {replied_content}",
        f"current_message_content: {current_content}",
    ]
    return message_text, relation_context


def _match_command_payload(message_text: str, regex_pattern: str) -> Optional[str]:
    match = re.search(regex_pattern, message_text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_search_keywords(message_text: str, *, max_keywords: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]{2,}", message_text.lower())
    keywords: list[str] = []
    seen = set()

    for token in tokens:
        if token in RAG_KEYWORD_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _build_rag_query_from_message(message_text: str) -> str:
    keywords = _extract_search_keywords(message_text)
    if keywords:
        return " ".join(keywords)
    return message_text


def _is_group_chat(update) -> bool:
    return bool(update.effective_chat and update.effective_chat.type in ['group', 'supergroup'])
