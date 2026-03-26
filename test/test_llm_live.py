import asyncio
import os

import pytest

from app import ai_model
from app.runtime_config import bootstrap_runtime_environment


def test_live_llm_smoke_current_provider_model(request: pytest.FixtureRequest) -> None:
    """Real network smoke test for the currently configured LLM.

    This test is opt-in to avoid flaky CI/network failures.
    Run manually with:
      RUN_LIVE_LLM_TEST=1 uv run pytest -q test/test_llm_live.py
    """
    #if os.getenv("RUN_LIVE_LLM_TEST") != "1":
    #    pytest.skip("Set RUN_LIVE_LLM_TEST=1 to run live LLM smoke test.")

    bootstrap_runtime_environment()

    provider = (os.getenv("LLM_PROVIDER") or "ark").strip().lower()

    if provider == "ark" and not os.getenv("ARK_API_KEY"):
        pytest.skip("ARK_API_KEY is missing for live ark test.")
    if provider == "azure" and (not os.getenv("AZURE_OPENAI_API_KEY") or not os.getenv("AZURE_OPENAI_ENDPOINT")):
        pytest.skip("Azure OpenAI credentials are missing for live azure test.")
    if provider == "ollama" and not os.getenv("OLLAMA_ENDPOINT"):
        pytest.skip("OLLAMA_ENDPOINT is missing for live ollama test.")

    ai_model.configure_llm(
        provider=provider,
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        ark_endpoint=os.getenv("ARK_API_ENDPOINT"),
        ark_api_key=os.getenv("ARK_API_KEY"),
        ark_model=os.getenv("ARK_MODEL"),
        ollama_endpoint=os.getenv("OLLAMA_ENDPOINT"),
        ollama_model=os.getenv("OLLAMA_MODEL"),
        request_timeout=45.0,
    )

    async def _call_model() -> str:
        return await ai_model.chat_completion_text(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": "Reply with a short greeting only."},
            ],
            provider=provider,
            temperature=0,
            max_tokens=32,
        )

    content = asyncio.run(_call_model())

    # Show model output only in non-quiet pytest runs.
    if getattr(request.config.option, "verbose", 0) >= 0:
        terminal_reporter = request.config.pluginmanager.get_plugin("terminalreporter")
        if terminal_reporter is not None:
            terminal_reporter.write_line(f"LLM ({provider}) response: {content}")

    assert isinstance(content, str)
    assert content.strip() != ""
