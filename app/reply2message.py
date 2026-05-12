import json
import logging
import re
from pathlib import Path
from typing import Optional

from app.ai_model import chat_completion


logger = logging.getLogger(__name__)
INFO_FILE_PATH = Path(__file__).resolve().parent.parent / "config" / "info.txt"


def _strip_markdown_code_fence(text: str) -> str:
    stripped = (text or "").strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if not lines:
        return stripped

    # Remove opening fence (``` or ```json) and closing fence if present.
    body_lines = lines[1:]
    if body_lines and body_lines[-1].strip().startswith("```"):
        body_lines = body_lines[:-1]
    return "\n".join(body_lines).strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    in_string = False
    escape = False
    depth = 0
    start = -1

    for i, ch in enumerate(text):
        if start == -1:
            if ch == "{":
                start = i
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def _parse_reply_payload(result_text: str) -> Optional[dict]:
    candidates = []
    raw = (result_text or "").strip()
    if raw:
        candidates.append(raw)

    unfenced = _strip_markdown_code_fence(raw)
    if unfenced and unfenced != raw:
        candidates.append(unfenced)

    extracted = _extract_first_json_object(unfenced or raw)
    if extracted:
        candidates.append(extracted)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _load_information_lines() -> list[str]:
    try:
        with open(INFO_FILE_PATH, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle if line.strip()]
    except FileNotFoundError:
        logger.warning("config/info.txt not found; using empty information.")
    except Exception as exc:
        logger.error("Failed to read config/info.txt: %s", exc)
    return []


def _build_direct_address_state(*, is_reply_to_bot: bool, is_mentioned: bool) -> list[str]:
    directly_addressed = is_reply_to_bot or is_mentioned
    return [
        f"is_reply_to_bot: {str(is_reply_to_bot).lower()}",
        f"is_mentioned: {str(is_mentioned).lower()}",
        f"directly_addressed: {str(directly_addressed).lower()}",
    ]


def _split_latest_message(message_history: list[str]) -> tuple[list[str], str]:
    if not message_history:
        return [], "(empty)"
    return message_history[:-1], message_history[-1]


def _split_additional_context(additional_context: Optional[list[str]]) -> tuple[list[str], list[str]]:
    durable_context: list[str] = []
    message_specific_context: list[str] = []

    for line in additional_context or []:
        if line.startswith("user_memory_key:") or line.startswith("user_personal_memory:"):
            durable_context.append(line)
        else:
            message_specific_context.append(line)

    return durable_context, message_specific_context


def _build_user_prompt(
    message_history: list[str],
    rag_related_messages: Optional[list[str]] = None,
    additional_context: Optional[list[str]] = None,
    runtime_state: Optional[list[str]] = None,
    direct_address_state: Optional[list[str]] = None,
) -> str:
    earlier_history, latest_message = _split_latest_message(message_history)
    durable_context, message_specific_context = _split_additional_context(additional_context)

    history_block = "\n".join(earlier_history) if earlier_history else "(empty)"
    rag_block = "\n".join(rag_related_messages or []) if rag_related_messages else "(empty)"
    durable_block = "\n".join(durable_context) if durable_context else "(none)"
    message_specific_block = "\n".join(message_specific_context) if message_specific_context else "(none)"
    direct_address_block = "\n".join(direct_address_state or []) if direct_address_state else "(none)"
    runtime_block = "\n".join(runtime_state or []) if runtime_state else "(none)"

    return (
        "Here is the prompt context in 7 parts, ordered from the most reusable context to the most volatile context. "
        "The final section is the newest message.\n\n"
        "### PART 1: EARLIER HISTORY\n"
        f"{history_block}\n\n"
        "### PART 2: DURABLE CONTEXT\n"
        f"{durable_block}\n\n"
        "### PART 3: RAG RELATED MESSAGES\n"
        f"{rag_block}\n\n"
        "### PART 4: MESSAGE-SPECIFIC CONTEXT\n"
        f"{message_specific_block}\n\n"
        "### PART 5: DIRECT ADDRESS FLAGS\n"
        f"{direct_address_block}\n\n"
        "### PART 6: RUNTIME STATE\n"
        f"{runtime_block}\n\n"
        "### PART 7: LATEST MESSAGE TO RESPOND TO\n"
        f"{latest_message}\n"
    )


def _build_probe_system_prompt() -> str:
    return f"""
You decide whether Mioo / 小小宫 should reply to the latest message in a Telegram group chat.

Rules:
- Return valid JSON with exactly two keys: \"should_reply\" and \"reason\".
- The section named \"LATEST MESSAGE TO RESPOND TO\" is the newest message.
- The section named \"EARLIER HISTORY\" excludes that newest message and remains ordered from oldest to newest.
- Decide only whether the bot should reply. Do not draft the reply itself.
- Prefer silence unless the latest message clearly invites the bot in.
- Reply when the latest message directly addresses the bot, asks the bot a question, gives the bot a task, or continues an active back-and-forth with the bot.
- Stay silent for low-signal chat, acknowledgements, emoji-only reactions, or human-to-human banter that does not need the bot.
- If the chat clearly says not to reply, prefer silence.
- Keep \"reason\" short and specific.
""".strip()


def _build_generation_system_prompt(
    *,
    information_lines: list[str],
) -> str:
    information = "\n".join(f"- {line}" for line in information_lines) if information_lines else "(none)"
    return f"""
You are Mioo, also called 小小宫 in Chinese, speaking as a participant in a Telegram group chat.

Background:
{information}

Rules:
- Reply in the same language as the latest message.
- Sound like a real group chat participant, not a formal assistant.
- Keep it short: usually one line, or one to two short sentences.
- Answer the immediate moment instead of giving a broad explanation.
- No markdown, no bullet lists, no JSON, and no roleplay framing.
- Keep punctuation light and natural.
- Do not force cat-girl tics like nya unless the room already sounds like that.
- Keep the Mioo / 小小宫 identity light, warm, and understated.
- If directly addressed, answer the direct ask instead of staying silent.
- If you refer to yourself in Chinese, use 小小宫. Otherwise use Mioo.
- Use the direct-address flags and runtime state from the user context only as supporting context.
- Return only the final reply text.
""".strip()


def _normalize_reply_content(result_text: str) -> Optional[str]:
    raw = _strip_markdown_code_fence((result_text or "").strip())
    if not raw:
        return None

    payload = _parse_reply_payload(raw)
    if payload is not None and isinstance(payload.get("reply_content"), str):
        raw = payload.get("reply_content", "")

    raw = re.sub(r"^(?:mioo|小小宫|assistant)\s*[:：]\s*", "", raw.strip(), flags=re.IGNORECASE)

    for prefix in (
        "Sure,",
        "Sure.",
        "Of course,",
        "Of course.",
        "Here is the reply:",
        "Here's the reply:",
    ):
        if raw.lower().startswith(prefix.lower()):
            raw = raw[len(prefix):].lstrip()
            break

    raw = " ".join(raw.split())
    raw = re.sub(r"([!?.,~。！？])\1+", r"\1", raw)
    return raw or None


def _parse_json_boolean(value: object) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    return None


async def should_activate_reply(
    message_history: list[str],
    *,
    rag_related_messages: Optional[list[str]] = None,
    additional_context: Optional[list[str]] = None,
    is_reply_to_bot: bool = False,
    is_mentioned: bool = False,
    runtime_state: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> bool:
    direct_address_state = _build_direct_address_state(
        is_reply_to_bot=is_reply_to_bot,
        is_mentioned=is_mentioned,
    )
    try:
        completion = await chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": _build_probe_system_prompt(),
                },
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        message_history,
                        rag_related_messages=rag_related_messages,
                        additional_context=additional_context,
                        runtime_state=runtime_state,
                        direct_address_state=direct_address_state,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            model=model,
        )

        result_json = _parse_reply_payload(completion.content or "")
        if result_json is None:
            logger.warning("Model returned invalid activation payload. Raw prefix: %r", (completion.content or "")[:200])
            return False

        should_reply = _parse_json_boolean(result_json.get("should_reply"))
        if should_reply is None:
            logger.warning("Model returned non-boolean should_reply field: %r", result_json.get("should_reply"))
            return False

        logger.info("Reply activation probe: %s", result_json)
        return should_reply
    except Exception as exc:
        logger.exception("An error occurred in should_activate_reply: %s", exc)
        return False


async def generate_group_reply(
    message_history: list[str],
    *,
    rag_related_messages: Optional[list[str]] = None,
    additional_context: Optional[list[str]] = None,
    is_reply_to_bot: bool = False,
    is_mentioned: bool = False,
    runtime_state: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    direct_address_state = _build_direct_address_state(
        is_reply_to_bot=is_reply_to_bot,
        is_mentioned=is_mentioned,
    )
    try:
        completion = await chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": _build_generation_system_prompt(
                        information_lines=_load_information_lines(),
                    ),
                },
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        message_history,
                        rag_related_messages=rag_related_messages,
                        additional_context=additional_context,
                        runtime_state=runtime_state,
                        direct_address_state=direct_address_state,
                    ),
                },
            ],
            model=model,
        )

        reply_text = _normalize_reply_content(completion.content or "")
        if not reply_text:
            return None

        logger.info("Generated group reply: %s", reply_text)
        return reply_text
    except Exception as exc:
        logger.exception("An error occurred in generate_group_reply: %s", exc)
        return None


async def should_reply_and_generate(
    message_history: list[str],
    *,
    rag_related_messages: Optional[list[str]] = None,
    additional_context: Optional[list[str]] = None,
    is_reply_to_bot: bool = False,
    is_mentioned: bool = False,
    runtime_state: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    directly_addressed = is_reply_to_bot or is_mentioned
    if not directly_addressed:
        should_reply = await should_activate_reply(
            message_history,
            rag_related_messages=rag_related_messages,
            additional_context=additional_context,
            is_reply_to_bot=is_reply_to_bot,
            is_mentioned=is_mentioned,
            runtime_state=runtime_state,
            model=model,
        )
        if not should_reply:
            return None

    return await generate_group_reply(
        message_history,
        rag_related_messages=rag_related_messages,
        additional_context=additional_context,
        is_reply_to_bot=is_reply_to_bot,
        is_mentioned=is_mentioned,
        runtime_state=runtime_state,
        model=model,
    )