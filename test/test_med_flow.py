import asyncio
from pathlib import Path

import pytest

import app.med as med


class _Completion:
    def __init__(self, content: str):
        self.content = content


def test_generate_med_applies_defaults(monkeypatch):
    async def fake_chat_completion(*, messages, response_format, model=None):
        return _Completion(
            '{"hospital_name":"H","patient":{"name":"A","gender":"女","age":"20"},'
            '"medicines":[{"name":"M"}],"doctor":{},"watermark":""}'
        )

    monkeypatch.setattr(med, "chat_completion", fake_chat_completion)

    payload = asyncio.run(med.generate_med("make one"))

    assert payload["patient"]["department"] == "精神科综合门诊"
    assert len(payload["patient"]["id"]) == 10
    assert payload["patient"]["fee_type"] == "自费"
    assert payload["patient"]["date"]["year"] == "2025"
    assert payload["doctor"]["name"] == "孙致连"
    assert payload["medicines"][0]["quantity"] == ""


def test_generate_med_raises_on_invalid_json(monkeypatch):
    async def fake_chat_completion(*, messages, response_format, model=None):
        return _Completion("not-json")

    monkeypatch.setattr(med, "chat_completion", fake_chat_completion)

    with pytest.raises(ValueError):
        asyncio.run(med.generate_med("bad"))


def test_generate_med_raises_when_required_field_missing(monkeypatch):
    async def fake_chat_completion(*, messages, response_format, model=None):
        return _Completion('{"patient":{},"medicines":[{}],"doctor":{},"watermark":""}')

    monkeypatch.setattr(med, "chat_completion", fake_chat_completion)

    with pytest.raises(ValueError):
        asyncio.run(med.generate_med("missing hospital_name"))


def test_generate_jpg_returns_false_when_source_pdf_missing(tmp_path):
    out = asyncio.run(med.generate_jpg(str(tmp_path / "missing.pdf")))
    assert out is False


def test_generate_jpg_rejects_invalid_quality_and_ppi(tmp_path):
    src = tmp_path / "a.pdf"
    src.write_bytes(b"dummy")

    assert asyncio.run(med.generate_jpg(str(src), quality=0)) is False
    assert asyncio.run(med.generate_jpg(str(src), ppi=0)) is False


def test_generate_jpg_returns_false_when_pdfium_missing(tmp_path, monkeypatch):
    src = tmp_path / "a.pdf"
    src.write_bytes(b"dummy")
    monkeypatch.setattr(med, "pdfium", None)

    assert asyncio.run(med.generate_jpg(str(src))) is False


def test_generate_jpg_from_med_json_short_circuits_when_pdf_generation_fails(monkeypatch):
    async def fake_generate_pdf(json_input, output_pdf=None):
        return False

    async def fake_generate_jpg(pdf_path, jpg_output=None, quality=30, ppi=150):
        raise AssertionError("generate_jpg should not be called")

    monkeypatch.setattr(med, "generate_pdf", fake_generate_pdf)
    monkeypatch.setattr(med, "generate_jpg", fake_generate_jpg)

    out = asyncio.run(med.generate_jpg_from_med_json({"k": "v"}, "out.jpg"))
    assert out is False


def test_generate_jpg_from_med_json_can_raise_render_error(monkeypatch):
    async def fake_generate_pdf(json_input, output_pdf=None):
        return False

    monkeypatch.setattr(med, "generate_pdf", fake_generate_pdf)

    with pytest.raises(med.MedRenderError):
        asyncio.run(med.generate_jpg_from_med_json({"k": "v"}, "out.jpg", raise_on_failure=True))


def test_generate_jpg_from_med_json_returns_jpg_path(monkeypatch):
    async def fake_generate_pdf(json_input, output_pdf=None):
        return "x.pdf"

    async def fake_generate_jpg(pdf_path, jpg_output=None, quality=30, ppi=150):
        assert pdf_path == "x.pdf"
        return "x.jpg"

    monkeypatch.setattr(med, "generate_pdf", fake_generate_pdf)
    monkeypatch.setattr(med, "generate_jpg", fake_generate_jpg)

    out = asyncio.run(med.generate_jpg_from_med_json({"k": "v"}, "out.jpg"))
    assert out == "x.jpg"


def test_latex_resource_exists_returns_true_when_kpsewhich_missing(monkeypatch):
    async def fake_exec(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    assert asyncio.run(med._latex_resource_exists("ctexart.cls")) is True
