import base64
import logging
import os
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


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


async def image_to_text(
    image_path: str,
    *,
    prompt: str = (
        "Return concise output with exactly 2 sections:\n"
        "TEXT_IN_IMAGE: key visible words/numbers only; '(none)' if absent.\n"
        "VISUAL_SUMMARY: 1-2 short sentences about non-text visual content."
    ),
    model: Optional[str] = None,
) -> Optional[str]:
    """Convert an image file into text using Ark Responses API."""
    api_key = os.getenv("ARK_API_KEY")
    if not api_key:
        logger.warning("ARK_API_KEY is not configured; skipping image-to-text.")
        return None

    response_url = os.getenv("ARK_RESPONSES_ENDPOINT", "https://ark.cn-beijing.volces.com/api/v3/responses")
    selected_model = model or os.getenv("ARK_VISION_MODEL") or os.getenv("ARK_MODEL") or "doubao-seed-1-6-251015"

    base64_file = _read_base64_file(image_path)
    mime_type = _guess_mime_type(image_path)

    payload = {
        "model": selected_model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{base64_file}",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(response_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
        extracted = _extract_text_from_responses_payload(data)
        return extracted or None
    except Exception as e:
        logger.warning(f"image_to_text failed: {e}")
        return None
