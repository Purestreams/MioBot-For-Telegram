import base64
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx
from app.ai_model import LLMProvider, get_settings
from app.runtime_config import get_ark_responses_endpoint, get_runtime_value

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StickerUnderstanding:
    description: str
    tags: list[str]
    mood: Optional[str]
    safe_for_reply: bool = True


def _read_base64_file(file_path: str) -> str:
    with open(file_path, "rb") as read_file:
        return base64.b64encode(read_file.read()).decode("utf-8")


def _guess_mime_type(file_path: str) -> str:
    lower = file_path.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith(".gif"):
        return "image/gif"
    return "image/jpeg"


def _extract_text_from_responses_payload(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = payload.get("output")
    if isinstance(output, list):
        texts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") in {"output_text", "text"}:
                    text = block.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
        if texts:
            return "\n".join(texts)

    return ""


def _extract_text_from_chat_payload(payload: dict[str, Any]) -> str:
    """Read text from an OpenAI-compatible chat-completions response."""
    try:
        content = payload["choices"][0]["message"].get("content", "")
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", "")).strip()
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"} and item.get("text")
        ).strip()
    return ""


def _build_sticker_prompt(*, emoji: Optional[str] = None, set_name: Optional[str] = None) -> str:
    hints: list[str] = []
    if emoji:
        hints.append(f"Known sticker emoji: {emoji}.")
    if set_name:
        hints.append(f"Sticker set name: {set_name}.")

    prompt = (
        "Describe this Telegram sticker in one short plain-text line. "
        "Mention the main subject, notable visible text if any, and the emotion or reaction it conveys. "
        "Do not use bullet points, labels, markdown, or JSON."
    )
    if hints:
        prompt += " " + " ".join(hints)
    return prompt


def _build_sticker_understanding_prompt(*, emoji: Optional[str] = None, set_name: Optional[str] = None) -> str:
    hints: list[str] = []
    if emoji:
        hints.append(f"Known sticker emoji: {emoji}.")
    if set_name:
        hints.append(f"Sticker set name: {set_name}.")

    prompt = (
        "Analyze this Telegram sticker for a chat bot sticker-reply cache. "
        "Return only compact JSON with exactly these keys: "
        "description, tags, mood, safe_for_reply. "
        "description must be one short plain-text line about the visible subject, text, and reaction. "
        "tags must be 3 to 8 short lowercase English keywords useful for search, such as laugh, angry, thanks, cute. "
        "mood must be one short lowercase label. "
        "safe_for_reply must be false for sexual, hateful, violent, graphic, self-harm, private-data, or otherwise risky content; otherwise true."
    )
    if hints:
        prompt += " " + " ".join(hints)
    return prompt


def _strip_markdown_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    body_lines = lines[1:]
    if body_lines and body_lines[-1].strip().startswith("```"):
        body_lines = body_lines[:-1]
    return "\n".join(body_lines).strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    in_string = False
    escape = False
    depth = 0
    start = -1

    for index, char in enumerate(text or ""):
        if start == -1:
            if char == "{":
                start = index
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
                continue
            if char == "\\":
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


def _coerce_sticker_tags(value: Any) -> list[str]:
    if isinstance(value, str):
        raw_tags = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, list):
        raw_tags = [str(part).strip() for part in value]
    else:
        raw_tags = []

    tags: list[str] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        tag = " ".join(raw_tag.lower().split())[:40]
        if not tag or tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= 8:
            break
    return tags


def _parse_sticker_understanding(text: str) -> Optional[StickerUnderstanding]:
    raw = _strip_markdown_code_fence(text or "")
    if not raw:
        return None

    payload: dict[str, Any] | None = None
    for candidate in (raw, _extract_first_json_object(raw) or ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break

    if payload is None:
        return StickerUnderstanding(description=" ".join(raw.split()), tags=[], mood=None, safe_for_reply=True)

    description = str(payload.get("description") or payload.get("summary") or "").strip()
    if not description:
        return None

    mood = payload.get("mood")
    normalized_mood = " ".join(str(mood).lower().split())[:40] if mood else None
    safe_value = payload.get("safe_for_reply", True)
    safe_for_reply = safe_value if isinstance(safe_value, bool) else str(safe_value).strip().lower() not in {"0", "false", "no", "unsafe"}

    return StickerUnderstanding(
        description=" ".join(description.split()),
        tags=_coerce_sticker_tags(payload.get("tags")),
        mood=normalized_mood or None,
        safe_for_reply=bool(safe_for_reply),
    )


async def image_to_text(
    image_path: str,
    *,
    prompt: str = (
        "Return concise output with exactly 2 sections:\n"
        "TEXT_IN_IMAGE: key visible words/numbers only; '(none)' if absent.\n"
        "VISUAL_SUMMARY: 1-2 short sentences about non-text visual content."
    ),
    model: Optional[str] = None,
    raise_errors: bool = False,
) -> Optional[str]:
    """Convert an image file into text using the active provider's vision API."""
    settings = get_settings()
    base64_file = await asyncio.to_thread(_read_base64_file, image_path)
    mime_type = _guess_mime_type(image_path)

    if settings.provider == LLMProvider.ZAN:
        if not settings.zan_api_key or not settings.zan_endpoint:
            logger.warning("ZAN vision configuration is incomplete; skipping image-to-text.")
            return None
        response_url = settings.zan_endpoint
        selected_model = model or get_runtime_value("ZAN_VISION_MODEL") or settings.zan_model
        api_key = settings.zan_api_key
        payload = {
            "model": selected_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{base64_file}"},
                        },
                    ],
                }
            ],
        }
        extract_text = _extract_text_from_chat_payload
    else:
        api_key = get_runtime_value("ARK_API_KEY")
        if not api_key:
            logger.warning("ARK_API_KEY is not configured; skipping image-to-text.")
            return None
        response_url = get_ark_responses_endpoint()
        selected_model = model or get_runtime_value("ARK_VISION_MODEL") or get_runtime_value("ARK_MODEL")
        payload = {
            "model": selected_model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:{mime_type};base64,{base64_file}"},
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
        }
        extract_text = _extract_text_from_responses_payload

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(response_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        extracted = extract_text(data)
        return extracted or None
    except Exception as e:
        logger.warning(f"image_to_text failed: {e}")
        if raise_errors:
            raise
        return None


async def sticker_to_text(
    image_path: str,
    *,
    emoji: Optional[str] = None,
    set_name: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Describe a Telegram sticker in a single natural-language line."""
    description = await image_to_text(
        image_path,
        prompt=_build_sticker_prompt(emoji=emoji, set_name=set_name),
        model=model,
    )
    if not description:
        return None
    return " ".join(description.split()) or None


async def sticker_to_understanding(
    image_path: str,
    *,
    emoji: Optional[str] = None,
    set_name: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[StickerUnderstanding]:
    """Describe and tag a Telegram sticker for outbound reply selection."""
    raw_understanding = await image_to_text(
        image_path,
        prompt=_build_sticker_understanding_prompt(emoji=emoji, set_name=set_name),
        model=model,
    )
    if not raw_understanding:
        return None
    return _parse_sticker_understanding(raw_understanding)
