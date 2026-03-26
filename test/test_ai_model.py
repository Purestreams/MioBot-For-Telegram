from app.ai_model import LLMProvider, _clean_dict, _coerce_provider


def test_coerce_provider_alias_and_values():
    assert _coerce_provider("azure_openai") == LLMProvider.AZURE
    assert _coerce_provider("azure") == LLMProvider.AZURE
    assert _coerce_provider("ollama") == LLMProvider.OLLAMA
    assert _coerce_provider("ark") == LLMProvider.ARK
    # Unknown values currently fall back to ARK.
    assert _coerce_provider("unknown") == LLMProvider.ARK


def test_clean_dict_removes_none_values_only():
    payload = {"a": 1, "b": None, "c": "", "d": False}
    assert _clean_dict(payload) == {"a": 1, "c": "", "d": False}
