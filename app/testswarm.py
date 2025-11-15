"""Example: Minimal multi-turn, tool-enabled chat loop using the shared LLM helper.

This sample mirrors the original Azure-only demo but now routes all model calls
through ``app.ai_model`` so that the active provider can be swapped via
configuration. The behaviour remains identical for Azure, while also working
with the Ark Volces API if that is the selected backend.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from app.ai_model import (
    LLMProvider,
    chat_completion,
    get_settings,
    stream_chat_completion,
)


# -------- Example local "tools" (Python functions) --------
def get_weather(location: str) -> str:
    """Dummy weather lookup.

    In production, call a real weather API.
    """

    fake_db = {
        "san francisco": "Sunny, 68F",
        "new york": "Cloudy, 75F",
        "london": "Light rain, 60F",
    }
    return fake_db.get(location.lower(), f"Weather data for '{location}' unavailable.")


def detect_language(text: str) -> str:
    """Very naive language detector (English/Chinese) for demo purposes only."""

    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "chinese"
    return "english"


LOCAL_TOOL_IMPLEMENTATIONS = {
    "get_weather": get_weather,
    "detect_language": detect_language,
}


TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather conditions for a given location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name or location."}
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_language",
            "description": "Detect whether the user input seems to be English or Chinese.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "User input to analyze."}
                },
                "required": ["text"],
            },
        },
    },
]


def _default_model(settings=None) -> Optional[str]:
    settings = settings or get_settings()
    if settings.provider == LLMProvider.AZURE:
        return settings.azure_deployment
    return settings.ark_model


async def run_tool_loop(
    *,
    system_instructions: str,
    user_message: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    max_turns: int = 5,
    existing_messages: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Repeatedly call the chat API until no tool calls are returned."""

    if existing_messages:
        messages = existing_messages
    else:
        messages = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_message},
        ]

    for _ in range(max_turns):
        completion = await chat_completion(
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            model=model,
            provider=provider,
        )
        choice = completion.raw["choices"][0]
        message = choice["message"]
        messages.append({k: v for k, v in message.items() if v is not None})

        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            break

        for tool_call in tool_calls:
            function_info = tool_call.get("function", {})
            name = function_info.get("name")
            args_json = function_info.get("arguments")
            try:
                args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                tool_output = f"Error: Could not parse arguments: {args_json}"
            else:
                impl = LOCAL_TOOL_IMPLEMENTATIONS.get(name or "")
                if impl is None:
                    tool_output = f"Error: No local implementation for tool '{name}'."
                else:
                    try:
                        tool_output = impl(**args)
                    except Exception as exc:  # noqa: BLE001
                        tool_output = f"Error during tool '{name}': {exc}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.get("id"),
                    "name": name,
                    "content": str(tool_output),
                }
            )

    return messages


async def streaming_example(
    *,
    user_message: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """Demonstrate streaming completions (Azure only)."""

    print("\n--- Streaming Example ---")
    stream = await stream_chat_completion(
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": user_message},
        ],
        model=model,
        provider=provider,
    )

    collected: List[str] = []
    async for event in stream:
        if getattr(event, "choices", None):
            delta = event.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                collected.append(token)
                print(token, end="", flush=True)
    print("\n--- End of Stream ---\nFull response:", "".join(collected))


async def agent_handoff_example(
    *,
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> None:
    """Demonstrate a crude handoff by swapping system instructions mid-run."""

    print("\n--- Agent Handoff Example ---")
    english_agent_sys = "You are an English support agent. If user speaks Chinese, call detect_language."
    chinese_agent_sys = "你是一名中文客服代理。请始终用简体中文回答。"

    msgs = await run_tool_loop(
        system_instructions=english_agent_sys,
        user_message="你好，我需要知道旧金山的天气。",
        model=model,
        provider=provider,
        max_turns=3,
    )

    detected_chinese = any(
        m.get("role") == "tool"
        and m.get("name") == "detect_language"
        and "chinese" in (m.get("content") or "").lower()
        for m in msgs
    )

    if detected_chinese:
        print("Handoff triggered: switching to Chinese agent.\n")
        msgs[0]["content"] = chinese_agent_sys
        msgs.append({"role": "user", "content": "你现在可以告诉我旧金山的天气吗？"})

        msgs = await run_tool_loop(
            system_instructions=chinese_agent_sys,
            user_message="",
            model=model,
            provider=provider,
            max_turns=3,
            existing_messages=msgs,
        )

    for entry in msgs:
        role = entry.get("role")
        content = entry.get("content")
        if role in {"system", "user", "assistant"}:
            print(f"{role.upper()}: {content}")
        elif role == "tool":
            print(f"TOOL ({entry.get('name')}): {content}")


async def main() -> None:
    settings = get_settings()
    provider = settings.provider.value
    model = os.getenv("LLM_DEMO_MODEL", _default_model(settings) or "") or None

    print("=== Basic Single Call ===")
    completion = await chat_completion(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a whimsical, one-line haiku about ocean fog."},
        ],
        model=model,
        provider=provider,
    )
    print(completion.content)

    print("\n=== Tool Loop (Weather) ===")
    messages = await run_tool_loop(
        system_instructions="You are a weather assistant. Use get_weather to answer location questions.",
        user_message="What's the weather in London?",
        model=model,
        provider=provider,
    )
    for msg in messages:
        if msg.get("role") in {"assistant", "tool"}:
            print(f"{msg['role']}: {msg.get('content')}")

    if settings.provider == LLMProvider.AZURE:
        await streaming_example(user_message="Explain recursion in one sentence.", model=model, provider=provider)
        await agent_handoff_example(model=model, provider=provider)
    else:
        print("\nStreaming and handoff examples are skipped for non-Azure providers.")


if __name__ == "__main__":
    asyncio.run(main())