"""Utility helpers extracted from main entrypoint logic."""

import logging
import os
import re
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

OUTPUT_DIR = "output"
logger = logging.getLogger(__name__)

# URL regex patterns
YOUTUBE_URL_REGEX = (
    r'(?<![\w@.])(https?://)?(www\.)?'
    r'(youtube\.com/|youtu\.be/|youtube-nocookie\.com/)'
    r'(?:watch\?v=|embed/|v/|shorts/|live/)?'
    r'([a-zA-Z0-9_-]{11})(?![a-zA-Z0-9_-])'
    r'(?:[/?#&][^\s]*)?'
)
BILIBILI_URL_REGEX = (
    r'(?<![\w@.])(?:https?://)?(?:www\.|m\.)?'
    r'(?:bilibili\.com/(?:video/(?:BV[0-9A-Za-z]{10}|av\d+)|watch\?bvid=(?:BV[0-9A-Za-z]{10}|av\d+))'
    r'|b23\.tv/[A-Za-z0-9_-]{6,32})'
    r'(?![A-Za-z0-9_-])'
    r'(?:[/?#&][^\s]*)?'
)
TWITTER_URL_REGEX = (
    # Keep protocol optional to match existing YouTube/Bilibili behavior.
    r'(?<![\w@.])(https?://)?(?:www\.)?'
    r'(twitter\.com/|x\.com/)'
    r'[A-Za-z0-9_]+/status/\d+(?![A-Za-z0-9_])'
    r'(?:[/?#][^\s]*)?'
)
ZHIHU_ANSWER_URL_REGEX = (
    r'(?<![\w@.])(?:https?://)?(?:www\.)?'
    r'zhihu\.com/'
    r'(?:question/\d+(?![A-Za-z0-9_])/answer/\d+(?![A-Za-z0-9_])|answer/\d+(?![A-Za-z0-9_]))'
    r'(?:[/?#][^\s]*)?'
)
ZHIHU_ARTICLE_URL_REGEX = (
    r'(?<![\w@.])(?:https?://)?'
    r'(?:(?:www\.)?zhihu\.com/(?:article/\d+(?![A-Za-z0-9_])|column/[^/\s]+/p/\d+(?![A-Za-z0-9_]))|'
    r'zhuanlan\.zhihu\.com/p/\d+(?![A-Za-z0-9_]))'
    r'(?:[/?#][^\s]*)?'
)
ZHIHU_POST_URL_REGEX = (
    r'(?<![\w@.])(?:https?://)?(?:www\.)?zhihu\.com/'
    r'(?:(?:pin|p)/\d+(?![A-Za-z0-9_])|people/[^/\s]+/(?:pins|posts)/\d+(?![A-Za-z0-9_]))'
    r'(?:[/?#][^\s]*)?'
)
ZHIHU_QUESTION_URL_REGEX = (
    r'(?<![\w@.])(?:https?://)?(?:www\.)?'
    r'zhihu\.com/'
    r'question/\d+(?![A-Za-z0-9_])'
    r'(?:[/?#][^\s]*)?'
)
ZHIHU_URL_REGEX = "(?:" + "|".join(
    (
        ZHIHU_ANSWER_URL_REGEX,
        ZHIHU_ARTICLE_URL_REGEX,
        ZHIHU_POST_URL_REGEX,
        ZHIHU_QUESTION_URL_REGEX,
    )
) + ")"

MD2JPG_REGEX = r'/md2jpg(?:@\w+)?\s*,,,(.*),,,'
TEXT2JPG_REGEX = r'/text2jpg(?:@\w+)?\s*,,,(.*),,,'

RAG_KEYWORD_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "and", "or", "but", "if", "then",
    "this", "that", "it", "as", "at", "by", "from", "about", "just", "very",
    "you", "your", "me", "my", "we", "our", "they", "their", "he", "she", "his", "her",
}

RAG_QUERY_CONTEXT_PREFIXES = {
    "user_reply_relation",
    "message_reply_relation",
    "replied_to_author",
    "replied_to_content",
    "current_message_content",
    "caption",
    "sticker_emoji",
    "sticker_set_name",
    "sticker_description",
    "input_type",
}


def _build_output_path(prefix: str, message_id: int, extension: str = "jpg") -> str:
    return os.path.join(OUTPUT_DIR, f"{prefix}_{message_id}.{extension}")


def _remove_file_if_exists(path) -> None:
    if path and os.path.exists(path):
        os.remove(path)


async def _delete_message_if_exists(message) -> None:
    if message:
        try:
            await message.delete()
        except Exception as exc:
            logger.warning("Failed to delete Telegram message during cleanup: %s", exc)


def _extract_video_url(message_text: str) -> Optional[str]:
    """Return the first supported link, preserving the legacy API.

    New callers should use :func:`extract_supported_links` when a message may
    contain more than one link.  Keeping this wrapper avoids changing callers
    that only need one URL.
    """
    links = extract_supported_links(message_text)
    return links[0] if links else None


_SUPPORTED_LINK_PATTERNS = (
    ("youtube", re.compile(YOUTUBE_URL_REGEX, re.IGNORECASE)),
    ("bilibili", re.compile(BILIBILI_URL_REGEX, re.IGNORECASE)),
    ("twitter", re.compile(TWITTER_URL_REGEX, re.IGNORECASE)),
    ("zhihu", re.compile(ZHIHU_URL_REGEX, re.IGNORECASE)),
)
_URL_TRAILING_PUNCTUATION = ".,!?;:'\"”’)]}>。，！？；："
_QUERY_ALLOWLISTS = {
    # Keep only fields that select the requested video/page or intentional
    # playback position.  Share IDs and campaign parameters are discarded.
    "youtube": frozenset({"v", "list", "index", "start", "end", "t"}),
    "bilibili": frozenset({"bvid", "aid", "p", "t"}),
    "twitter": frozenset(),
    "zhihu": frozenset(),
}


def _clean_extracted_url(url: str, provider: Optional[str] = None) -> str:
    """Normalize a supported URL and remove non-essential tracking fields."""
    cleaned = (url or "").strip().lstrip("<")
    cleaned = cleaned.rstrip(_URL_TRAILING_PUNCTUATION)
    if cleaned and not re.match(r"https?://", cleaned, re.IGNORECASE):
        cleaned = f"https://{cleaned}"
    if not cleaned:
        return ""

    try:
        parsed = urlsplit(cleaned)
    except ValueError:
        return cleaned

    allowed_query_keys = _QUERY_ALLOWLISTS.get(provider or "", frozenset())
    query_items = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() in allowed_query_keys
    ]
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path, urlencode(query_items, doseq=True), "")
    )


def extract_supported_links(message_text: str) -> list[str]:
    """Extract every supported media/article link in source-text order.

    Telegram messages frequently contain prose around a link, multiple links,
    or links wrapped in Markdown/HTML punctuation.  Searching each supported
    provider independently and sorting by match position keeps all links while
    retaining the provider priority used by the old single-link helper.
    """
    text = str(message_text or "")
    matches: list[tuple[int, int, int, str]] = []
    for priority, (provider, pattern) in enumerate(_SUPPORTED_LINK_PATTERNS):
        for match in pattern.finditer(text):
            url = _clean_extracted_url(match.group(0), provider)
            if url:
                matches.append((match.start(), match.end(), priority, url))

    matches.sort(key=lambda item: (item[0], item[1], item[2]))
    links: list[str] = []
    seen: set[str] = set()
    occupied_until = -1
    for start, end, _priority, url in matches:
        normalized = url.casefold()
        if normalized in seen or start < occupied_until:
            continue
        seen.add(normalized)
        links.append(url)
        occupied_until = end
    return links


def extract_supported_links_from_message(message) -> list[str]:
    """Extract supported links from text, captions, and Telegram text links."""
    links: list[str] = []
    for value in (
        getattr(message, "text", None),
        getattr(message, "caption", None),
    ):
        links.extend(extract_supported_links(value or ""))

    # A Telegram ``TEXT_LINK`` entity has no URL in the visible text.  Include
    # it when present; regular URL entities are already found by the text scan.
    for entities_attr in ("entities", "caption_entities"):
        for entity in getattr(message, entities_attr, None) or []:
            entity_url = getattr(entity, "url", None)
            if entity_url:
                links.extend(extract_supported_links(str(entity_url)))

    deduped: list[str] = []
    seen: set[str] = set()
    for link in links:
        normalized = link.casefold()
        if normalized not in seen:
            seen.add(normalized)
            deduped.append(link)
    return deduped


def is_zhihu_answer_url(url: str) -> bool:
    return bool(url and re.search(ZHIHU_ANSWER_URL_REGEX, url, flags=re.IGNORECASE))


def classify_zhihu_url(url: str) -> Optional[str]:
    """Return the Zhihu content kind represented by a URL."""
    value = str(url or "")
    for kind, pattern in (
        ("answer", ZHIHU_ANSWER_URL_REGEX),
        ("article", ZHIHU_ARTICLE_URL_REGEX),
        ("post", ZHIHU_POST_URL_REGEX),
        ("question", ZHIHU_QUESTION_URL_REGEX),
    ):
        if re.search(pattern, value, flags=re.IGNORECASE):
            return kind
    return None


def is_zhihu_url(url: str) -> bool:
    return classify_zhihu_url(url) is not None


def _normalize_telegram_username(value: Optional[str]) -> str:
    return str(value or "").strip().lstrip("@").lower()


def _is_reply_to_this_bot(update, bot_username: Optional[str], bot_user_id: Optional[int] = None) -> bool:
    message = getattr(update, "message", None)
    if not message or not message.reply_to_message:
        return False

    from_user = message.reply_to_message.from_user
    if not (from_user and from_user.is_bot):
        return False

    if bot_user_id is not None and getattr(from_user, "id", None) == bot_user_id:
        return True

    normalized_from_username = _normalize_telegram_username(getattr(from_user, "username", None))
    normalized_bot_username = _normalize_telegram_username(bot_username)
    return bool(normalized_bot_username and normalized_from_username == normalized_bot_username)


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


STOP_REPLY_TEXT = "好，我先不打扰了"
_STOP_REPLY_RE = re.compile(
    r"(?:闭嘴|閉嘴|别说了|別說了|不要再说(?:了)?|不要再說(?:了)?|别再说话|別再說話|"
    r"别吵了|別吵了|不要吵了|谁问你|誰問你|受够你(?:了)?|受夠你(?:了)?)"
)
_HISTORY_QUERY_RE = re.compile(
    r"(?:之前|上次|刚才|剛才|刚刚|剛剛|还记得|還記得|记得吗|記得嗎|"
    r"以前说过|以前說過|你说过|你說過|说过的|說過的|过去聊过|過去聊過|那次|"
    r"\bremember(?:\s+when)?\b|\blast time\b|\bearlier\b|\byou said\b|"
    r"\bwhat did (?:i|we|you) say\b)",
    flags=re.IGNORECASE,
)


def _mentions_another_bot_only(message_text: Optional[str], bot_username: Optional[str]) -> bool:
    mentions = re.findall(
        r"(?<![A-Za-z0-9_])@([A-Za-z0-9_]+bot)(?![A-Za-z0-9_])",
        message_text or "",
        flags=re.IGNORECASE,
    )
    own = (bot_username or "").strip().lstrip("@").lower()
    return bool(mentions) and own not in {mention.lower() for mention in mentions}


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
    tokens = re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]{2,}", message_text.lower())
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


def _build_rag_query_from_message(
    message_text: str,
    *,
    additional_context: Optional[list[str]] = None,
    sender_display: Optional[str] = None,
    max_chars: int = 800,
) -> str:
    keywords = _extract_search_keywords(message_text)
    parts: list[str] = []
    if keywords:
        parts.append(" ".join(keywords))
    elif message_text.strip():
        parts.append(message_text.strip())

    if sender_display:
        parts.append(sender_display)

    for line in additional_context or []:
        key, sep, value = line.partition(":")
        if not sep:
            continue
        if key.strip() in RAG_QUERY_CONTEXT_PREFIXES and value.strip():
            parts.append(value.strip())

    deduped: list[str] = []
    seen: set[str] = set()
    for part in parts:
        normalized = " ".join(str(part).split())
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)

    query = " | ".join(deduped) if deduped else message_text
    if len(query) > max_chars:
        return query[: max_chars - 1].rstrip() + "…"
    return query


def _is_group_chat(update) -> bool:
    return bool(update.effective_chat and update.effective_chat.type in ['group', 'supergroup'])
