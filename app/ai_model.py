"""Shared helpers for invoking configurable LLM backends.

This module provides a single entry point for chat completions that can talk to
either the Ark Volces endpoint or Azure OpenAI. Import and call
``chat_completion`` (or ``chat_completion_text``) everywhere in the project so
that switching providers only requires updating ``configure_llm`` or
environment variables.

Usage example:

```
from app.ai_model import chat_completion_text

markdown = await chat_completion_text(
  messages=[{"role": "system", "content": "..."}, {"role": "user", "content": prompt}],
  model="deepseek-r1-250528",
)
```

Set environment variables or call ``configure_llm`` once during program start
to select the provider and credentials.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

try:  # Optional dependency for Azure support
  from openai import AsyncAzureOpenAI
except ImportError:  # pragma: no cover - Azure SDK not installed
  AsyncAzureOpenAI = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
  """Supported chat completion providers."""

  ARK = "ark"
  AZURE = "azure"
  OLLAMA = "ollama"


@dataclass
class LLMSettings:
  """Configuration for the active LLM provider."""

  provider: LLMProvider
  azure_endpoint: Optional[str] = None
  azure_api_key: Optional[str] = None
  azure_api_version: Optional[str] = None
  azure_deployment: Optional[str] = None
  ark_endpoint: str = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
  ark_api_key: Optional[str] = None
  ark_model: Optional[str] = None
  ollama_endpoint: str = "http://100.69.97.8:11434"
  ollama_model: str = "gpt-oss:20b"
  request_timeout: float = 60.0


@dataclass
class ChatCompletionResult:
  """Normalized chat completion response."""

  content: str
  raw: Dict[str, Any]
  provider: LLMProvider


_settings: Optional[LLMSettings] = None
_azure_client_lock = asyncio.Lock()
_azure_client: Any = None


def configure_llm(
  *,
  provider: Optional[str | LLMProvider] = None,
  azure_endpoint: Optional[str] = None,
  azure_api_key: Optional[str] = None,
  azure_api_version: Optional[str] = None,
  azure_deployment: Optional[str] = None,
  ark_endpoint: Optional[str] = None,
  ark_api_key: Optional[str] = None,
  ark_model: Optional[str] = None,
  ollama_endpoint: Optional[str] = None,
  ollama_model: Optional[str] = None,
  request_timeout: Optional[float] = None,
) -> None:
  """Override LLM configuration at runtime."""

  global _settings

  settings = _load_settings_from_env()

  if provider is not None:
    settings.provider = _coerce_provider(provider)
  # When provider is omitted, keep the env-selected provider from
  # _load_settings_from_env() instead of overriding based on credentials.

  if azure_endpoint is not None:
    settings.azure_endpoint = azure_endpoint
  if azure_api_key is not None:
    settings.azure_api_key = azure_api_key
  if azure_api_version is not None:
    settings.azure_api_version = azure_api_version
  if azure_deployment is not None:
    settings.azure_deployment = azure_deployment
  if ark_endpoint is not None:
    settings.ark_endpoint = ark_endpoint
  if ark_api_key is not None:
    settings.ark_api_key = ark_api_key
  if ark_model is not None:
    settings.ark_model = ark_model
  if ollama_endpoint is not None:
    settings.ollama_endpoint = ollama_endpoint
  if ollama_model is not None:
    settings.ollama_model = ollama_model
  if request_timeout is not None:
    settings.request_timeout = request_timeout

  _settings = settings


def get_settings() -> LLMSettings:
  """Return cached LLM settings, loading from env if necessary."""

  global _settings
  if _settings is None:
    _settings = _load_settings_from_env()
  return _settings


def _load_settings_from_env() -> LLMSettings:
  provider_env = os.getenv("LLM_PROVIDER") or os.getenv("AI_PROVIDER")
  if provider_env:
    provider = _coerce_provider(provider_env)
  elif os.getenv("OLLAMA_ENDPOINT") or os.getenv("OLLAMA_MODEL"):
    provider = LLMProvider.OLLAMA
  elif os.getenv("ARK_API_KEY"):
    provider = LLMProvider.ARK
  elif os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"):
    provider = LLMProvider.AZURE
  else:
    provider = LLMProvider.ARK

  settings = LLMSettings(provider=provider)

  settings.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
  settings.azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
  settings.azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-04-01-preview")
  settings.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))

  settings.ark_endpoint = os.getenv(
    "ARK_API_ENDPOINT",
    settings.ark_endpoint,
  )
  settings.ark_api_key = os.getenv("ARK_API_KEY", settings.ark_api_key)
  settings.ark_model = os.getenv("ARK_MODEL", settings.ark_model)

  settings.ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", settings.ollama_endpoint)
  settings.ollama_model = os.getenv("OLLAMA_MODEL", settings.ollama_model)

  timeout_env = os.getenv("LLM_REQUEST_TIMEOUT")
  if timeout_env:
    try:
      settings.request_timeout = float(timeout_env)
    except ValueError:
      logger.warning("Invalid LLM_REQUEST_TIMEOUT=%s", timeout_env)

  return settings


def _coerce_provider(provider: str | LLMProvider) -> LLMProvider:
  if isinstance(provider, LLMProvider):
    return provider
  normalized = (provider or "").strip().lower()
  if normalized in {"azure_openai", "azure-openai", "azureopenai"}:
    normalized = LLMProvider.AZURE.value
  if normalized == LLMProvider.AZURE.value:
    return LLMProvider.AZURE
  if normalized == LLMProvider.OLLAMA.value:
    return LLMProvider.OLLAMA
  return LLMProvider.ARK


async def _get_azure_client(settings: LLMSettings) -> Any:
  if AsyncAzureOpenAI is None:  # pragma: no cover - azure optional
    raise RuntimeError("AsyncAzureOpenAI is not available; install openai>=1.35.0")

  global _azure_client
  if _azure_client is not None:
    return _azure_client

  async with _azure_client_lock:
    if _azure_client is None:
      if not settings.azure_api_key or not settings.azure_endpoint:
        raise RuntimeError("Azure OpenAI configuration is incomplete.")
      _azure_client = AsyncAzureOpenAI(
        api_key=settings.azure_api_key,
        api_version=settings.azure_api_version or "2024-04-01-preview",
        azure_endpoint=settings.azure_endpoint,
      )
  return _azure_client


def _clean_dict(payload: Dict[str, Any]) -> Dict[str, Any]:
  return {k: v for k, v in payload.items() if v is not None}


async def chat_completion(
  *,
  messages: List[Dict[str, Any]],
  model: Optional[str] = None,
  provider: Optional[str | LLMProvider] = None,
  response_format: Optional[Dict[str, Any]] = None,
  temperature: Optional[float] = None,
  max_tokens: Optional[int] = None,
  top_p: Optional[float] = None,
  tools: Optional[List[Dict[str, Any]]] = None,
  tool_choice: Optional[Any] = None,
  extra_body: Optional[Dict[str, Any]] = None,
) -> ChatCompletionResult:
  """Send a chat completion request to the configured provider."""

  settings = get_settings()
  active_provider = _coerce_provider(provider or settings.provider)

  if active_provider == LLMProvider.ARK:
    return await _chat_completion_ark(
      settings=settings,
      messages=messages,
      model=model,
      response_format=response_format,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      tools=tools,
      tool_choice=tool_choice,
      extra_body=extra_body,
    )

  if active_provider == LLMProvider.OLLAMA:
    return await _chat_completion_ollama(
      settings=settings,
      messages=messages,
      model=model,
      response_format=response_format,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=top_p,
      extra_body=extra_body,
    )

  return await _chat_completion_azure(
    settings=settings,
    messages=messages,
    model=model,
    response_format=response_format,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    tools=tools,
    tool_choice=tool_choice,
    extra_body=extra_body,
  )


async def _chat_completion_ollama(
  *,
  settings: LLMSettings,
  messages: List[Dict[str, Any]],
  model: Optional[str],
  response_format: Optional[Dict[str, Any]],
  temperature: Optional[float],
  max_tokens: Optional[int],
  top_p: Optional[float],
  extra_body: Optional[Dict[str, Any]],
) -> ChatCompletionResult:
  """Send chat completion to Ollama /api/chat.

  Docs: https://github.com/ollama/ollama/blob/main/docs/api.md
  """

  if response_format is not None:
    logger.info(
      "Ollama provider does not support response_format; dropping %s and relying on prompt instructions instead",
      response_format,
    )

  # Ollama expects messages in {role, content}. We accept and forward other keys.
  ollama_messages: List[Dict[str, Any]] = []
  for msg in messages:
    role = msg.get("role")
    content = msg.get("content")
    if not role or content is None:
      continue
    ollama_messages.append({"role": role, "content": str(content)})

  options: Dict[str, Any] = {}
  if temperature is not None:
    options["temperature"] = temperature
  if top_p is not None:
    options["top_p"] = top_p
  if max_tokens is not None:
    options["num_predict"] = max_tokens

  payload: Dict[str, Any] = {
    "model": model or settings.ollama_model,
    "messages": ollama_messages,
    "stream": False,
    "options": options or None,
  }

  if extra_body:
    payload.update(extra_body)

  cleaned_payload = _clean_dict(payload)

  base = (settings.ollama_endpoint or "").rstrip("/")
  url = f"{base}/api/chat"

  async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
    response = await client.post(url, json=cleaned_payload)

  response.raise_for_status()
  data = response.json()

  content = ""
  try:
    content = data.get("message", {}).get("content", "")
  except AttributeError:
    content = ""

  return ChatCompletionResult(content=content or "", raw=data, provider=LLMProvider.OLLAMA)


async def _chat_completion_ark(
  *,
  settings: LLMSettings,
  messages: List[Dict[str, Any]],
  model: Optional[str],
  response_format: Optional[Dict[str, Any]],
  temperature: Optional[float],
  max_tokens: Optional[int],
  top_p: Optional[float],
  tools: Optional[List[Dict[str, Any]]],
  tool_choice: Optional[Any],
  extra_body: Optional[Dict[str, Any]],
) -> ChatCompletionResult:
  if not settings.ark_api_key:
    raise RuntimeError("ARK_API_KEY is not configured.")

  ark_response_format: Optional[Dict[str, Any]] = None
  if response_format is not None:
    logger.info(
      "Ark provider does not support response_format; dropping %s and relying on prompt instructions instead",
      response_format,
    )

  payload: Dict[str, Any] = {
    "model": model or settings.ark_model,
    "messages": messages,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "top_p": top_p,
    "response_format": ark_response_format,
    "tools": tools,
    "tool_choice": tool_choice,
  }

  if extra_body:
    payload.update(extra_body)

  cleaned_payload = _clean_dict(payload)

  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {settings.ark_api_key}",
  }

  async with httpx.AsyncClient(timeout=settings.request_timeout) as client:
    response = await client.post(
      settings.ark_endpoint,
      headers=headers,
      json=cleaned_payload,
    )

  response.raise_for_status()
  data = response.json()
  content = ""
  try:
    content = data["choices"][0]["message"].get("content", "")
  except (KeyError, IndexError):
    logger.warning("Ark response missing content: %s", json.dumps(data))

  return ChatCompletionResult(content=content or "", raw=data, provider=LLMProvider.ARK)


async def _chat_completion_azure(
  *,
  settings: LLMSettings,
  messages: List[Dict[str, Any]],
  model: Optional[str],
  response_format: Optional[Dict[str, Any]],
  temperature: Optional[float],
  max_tokens: Optional[int],
  top_p: Optional[float],
  tools: Optional[List[Dict[str, Any]]],
  tool_choice: Optional[Any],
  extra_body: Optional[Dict[str, Any]],
) -> ChatCompletionResult:
  client = await _get_azure_client(settings)

  request_kwargs = _clean_dict(
    {
      "model": model or settings.azure_deployment,
      "messages": messages,
      "temperature": temperature,
      "max_tokens": max_tokens,
      "top_p": top_p,
      "response_format": response_format,
      "tools": tools,
      "tool_choice": tool_choice,
    }
  )

  if not request_kwargs.get("model"):
    raise RuntimeError("Azure deployment name is not configured.")

  if extra_body:
    request_kwargs.update(extra_body)

  response = await client.chat.completions.create(**request_kwargs)
  data = response.model_dump()

  content = ""
  try:
    content = data["choices"][0]["message"].get("content", "")
  except (KeyError, IndexError):
    logger.warning("Azure response missing content: %s", json.dumps(data))

  return ChatCompletionResult(content=content or "", raw=data, provider=LLMProvider.AZURE)


async def chat_completion_text(**kwargs: Any) -> str:
  """Return only the assistant message content for a chat completion."""

  result = await chat_completion(**kwargs)
  return result.content


async def stream_chat_completion(
  *,
  messages: List[Dict[str, Any]],
  model: Optional[str] = None,
  provider: Optional[str | LLMProvider] = None,
  response_format: Optional[Dict[str, Any]] = None,
  temperature: Optional[float] = None,
  max_tokens: Optional[int] = None,
  top_p: Optional[float] = None,
  tools: Optional[List[Dict[str, Any]]] = None,
  tool_choice: Optional[Any] = None,
  extra_body: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Any]:
  """Yield streaming events for providers that support it (currently Azure)."""

  settings = get_settings()
  active_provider = _coerce_provider(provider or settings.provider)

  if active_provider != LLMProvider.AZURE:
    raise NotImplementedError("Streaming is only supported for the Azure provider.")

  client = await _get_azure_client(settings)

  request_kwargs = _clean_dict(
    {
      "model": model or settings.azure_deployment,
      "messages": messages,
      "temperature": temperature,
      "max_tokens": max_tokens,
      "top_p": top_p,
      "response_format": response_format,
      "tools": tools,
      "tool_choice": tool_choice,
      "stream": True,
    }
  )

  if not request_kwargs.get("model"):
    raise RuntimeError("Azure deployment name is not configured.")

  if extra_body:
    request_kwargs.update(extra_body)

  stream = await client.chat.completions.create(**request_kwargs)

  async def _iterator() -> AsyncIterator[Any]:
    async for event in stream:
      yield event

  return _iterator()


def _build_test_messages(user_prompt: str, system_prompt: str) -> List[Dict[str, Any]]:
  return [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
  ]


async def _run_diagnostic(args: argparse.Namespace) -> None:
  if args.ark_api_key or args.ark_model:
    configure_llm(ark_api_key=args.ark_api_key, ark_model=args.ark_model)

  settings = get_settings()
  provider = _coerce_provider(args.provider or settings.provider)

  messages = _build_test_messages(args.user_prompt, args.system_prompt)

  logger.info(
    "Running diagnostic call",
    extra={
      "provider": provider.value,
      "model": args.model,
      "temperature": args.temperature,
      "max_tokens": args.max_tokens,
      "top_p": args.top_p,
    },
  )

  if args.dry_run:
    payload = _clean_dict(
      {
        "model": args.model or (settings.ark_model if provider == LLMProvider.ARK else settings.azure_deployment),
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "response_format": None,
      }
    )
    print("Dry run payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return

  try:
    result = await chat_completion(
      messages=messages,
      model=args.model,
      provider=provider,
      temperature=args.temperature,
      max_tokens=args.max_tokens,
      top_p=args.top_p,
    )
  except httpx.HTTPStatusError as exc:
    print("HTTP error while requesting chat completion:")
    print(f"Status: {exc.response.status_code} {exc.response.reason_phrase}")
    print(f"URL: {exc.request.url}")
    try:
      body = exc.response.json()
      print("Response JSON:")
      print(json.dumps(body, indent=2, ensure_ascii=False))
    except json.JSONDecodeError:
      print("Response Text:")
      print(exc.response.text)
    raise

  print("Completion succeeded. Provider:", result.provider.value)
  print("Content:\n", result.content)


def main() -> None:
  parser = argparse.ArgumentParser(description="Run a diagnostic chat completion call")
  parser.add_argument("--provider", choices=[p.value for p in LLMProvider], help="Force provider instead of env")
  parser.add_argument("--model", help="Override model/deployment name")
  parser.add_argument("--temperature", type=float, default=None)
  parser.add_argument("--top-p", dest="top_p", type=float, default=None)
  parser.add_argument("--max-tokens", dest="max_tokens", type=int, default=None)
  parser.add_argument(
    "--system-prompt",
    default="You are a helpful assistant that replies briefly.",
    help="System prompt for the diagnostic request",
  )
  parser.add_argument(
    "--user-prompt",
    default="Reply with the single word 'pong' if you received this message.",
    help="User message for the diagnostic request",
  )
  parser.add_argument("--dry-run", action="store_true", help="Print payload without sending the request")
  parser.add_argument("--ark-api-key", help="Override ARK_API_KEY for this run")
  parser.add_argument("--ark-model", help="Override ARK_MODEL for this run")

  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO)

  asyncio.run(_run_diagnostic(args))


if __name__ == "__main__":
  main()


