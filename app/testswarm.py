"""
Example: Minimal multi-turn, tool-enabled chat loop using Azure OpenAI with AsyncAzureOpenAI.

This script manually reproduces (in a simplified way) the execution loop
described in Swarm's README:

1. Get a completion from the "current agent" (represented here by a system prompt).
2. Execute tool (function) calls returned by the model.
3. (Optional) Switch "agent" (we simulate handoff by swapping system instructions).
4. Repeat until the model stops requesting tools.

Prerequisites:
  pip install openai>=1.35.0

Environment Variables Required:
  AZURE_OPENAI_API_KEY         Your Azure OpenAI key
  AZURE_OPENAI_ENDPOINT        Your Azure OpenAI endpoint, e.g. https://my-resource.openai.azure.com
  (Optional) AZURE_OPENAI_API_VERSION  Defaults to 2024-02-15-preview if unset

Azure Note:
  For Azure, the value you pass in the 'model' field must match your deployment name,
  NOT necessarily the base model name (e.g. "gpt-4o-mini" might be deployed as "my-gpt4o-mini").

Run:
  python sample.py

What it shows:
  - Basic non-streaming single call
  - Simple iterative tool (function) execution loop
  - Streaming example
"""

import asyncio
import json
import os
from typing import Any, Dict, List
from openai import AsyncAzureOpenAI


# -------- Configuration helpers --------
def get_azure_client() -> AsyncAzureOpenAI:
    api_key = os.environ["AZURE_OPENAI_API_KEY"]
    azure_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]
    api_version = "2024-04-01-preview"
    return AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
    )


# -------- Example local "tools" (Python functions) --------
def get_weather(location: str) -> str:
    """
    Dummy weather lookup.
    In production, call a real weather API.
    """
    fake_db = {
        "san francisco": "Sunny, 68F",
        "new york": "Cloudy, 75F",
        "london": "Light rain, 60F",
    }
    return fake_db.get(location.lower(), f"Weather data for '{location}' unavailable.")


def detect_language(text: str) -> str:
    """
    Very naive language detector (English/Chinese) for demo purposes only.
    Returns 'chinese' if any CJK Unified Ideograph is present, else 'english'.
    """
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            return "chinese"
    return "english"


# Map tool name -> actual callable
LOCAL_TOOL_IMPLEMENTATIONS = {
    "get_weather": get_weather,
    "detect_language": detect_language,
}


# JSON schema tool definitions for the Chat Completions API
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


# -------- Core loop (simplified Swarm-like) --------
async def run_tool_loop(
    client: AsyncAzureOpenAI,
    deployment: str,
    system_instructions: str,
    user_message: str,
    max_turns: int = 5,
    existing_messages: List[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Repeatedly call the Chat Completions API until no further tool calls occur
    or max_turns is reached.
    """
    if existing_messages:
        messages = existing_messages
    else:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_instructions},
            {"role": "user", "content": user_message},
        ]

    for turn in range(1, max_turns + 1):
        response = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
        )

        choice = response.choices[0]
        msg = choice.message
        messages.append(msg.model_dump(exclude_none=True))

        tool_calls = msg.tool_calls or []
        if not tool_calls:
            # Model responded normally; done
            break

        # Execute each tool call and append a tool response
        for tool_call in tool_calls:
            name = tool_call.function.name
            args_json = tool_call.function.arguments
            try:
                args = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                tool_output = f"Error: Could not parse arguments: {args_json}"
            else:
                impl = LOCAL_TOOL_IMPLEMENTATIONS.get(name)
                if impl is None:
                    tool_output = f"Error: No local implementation for tool '{name}'."
                else:
                    try:
                        tool_output = impl(**args)
                    except Exception as e:  # noqa: BLE001
                        tool_output = f"Error during tool '{name}': {e}"

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": name,
                    "content": str(tool_output),
                }
            )

    return messages


# -------- Streaming example --------
async def streaming_example(
    client: AsyncAzureOpenAI,
    deployment: str,
    user_message: str,
) -> None:
    """
    Show how to stream a completion (no tools here for brevity).
    """
    print("\n--- Streaming Example ---")
    stream = await client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": user_message},
        ],
        stream=True,
    )

    collected = []
    async for event in stream:
        if event.choices and event.choices[0].delta and event.choices[0].delta.content:
            token = event.choices[0].delta.content
            collected.append(token)
            print(token, end="", flush=True)
    print("\n--- End of Stream ---\nFull response:", "".join(collected))


# -------- Simulated "Agent Handoff" --------
async def agent_handoff_example(client: AsyncAzureOpenAI, deployment: str) -> None:
    """
    Demonstrate a crude 'handoff' by swapping system instructions mid-conversation
    based on a tool call decision (detect_language).
    """
    print("\n--- Agent Handoff Example ---")

    english_agent_sys = "You are an English support agent. If user speaks Chinese, call detect_language."
    chinese_agent_sys = "你是一名中文客服代理。请始终用简体中文回答。"

    # First loop: start with English agent
    msgs = await run_tool_loop(
        client,
        deployment,
        system_instructions=english_agent_sys,
        user_message="你好，我需要知道旧金山的天气。",
        max_turns=3,
    )

    # Check if language was detected as Chinese
    detected_chinese = any(
        m.get("role") == "tool"
        and m.get("name") == "detect_language"
        and "chinese" in (m.get("content") or "").lower()
        for m in msgs
    )

    if detected_chinese:
        print("Handoff triggered: switching to Chinese agent.\n")
        # Replace first system message
        msgs[0]["content"] = chinese_agent_sys
        # Add new user message to existing conversation
        msgs.append({"role": "user", "content": "你现在可以告诉我旧金山的天气吗？"})
        # Continue with existing messages
        msgs = await run_tool_loop(
            client,
            deployment,
            system_instructions=chinese_agent_sys,
            user_message="",  # Not used when existing_messages is provided
            max_turns=3,
            existing_messages=msgs,
        )

    # Print final conversation
    for m in msgs:
        role = m.get("role")
        content = m.get("content")
        if role in ("system", "user", "assistant"):
            print(f"{role.upper()}: {content}")
        elif role == "tool":
            print(f"TOOL ({m.get('name')}): {content}")


# -------- Main demo --------
async def main():
    # UPDATE this to your Azure deployment name (not necessarily the base model name)
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-nano")

    client = get_azure_client()

    print("=== Basic Single Call ===")
    basic = await client.chat.completions.create(
        model=deployment_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Give me a whimsical, one-line haiku about ocean fog."},
        ],
    )
    print(basic.choices[0].message.content)

    print("\n=== Tool Loop (Weather) ===")
    messages = await run_tool_loop(
        client,
        deployment=deployment_name,
        system_instructions="You are a weather assistant. Use get_weather to answer location questions.",
        user_message="What's the weather in London?",
    )
    for m in messages:
        if m["role"] in ("assistant", "tool"):
            print(f"{m['role']}: {m.get('content')}")

    await streaming_example(client, deployment_name, "Explain recursion in one sentence.")

    await agent_handoff_example(client, deployment_name)


if __name__ == "__main__":
    asyncio.run(main())