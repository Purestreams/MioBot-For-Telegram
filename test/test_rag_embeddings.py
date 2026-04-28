import asyncio

import numpy as np
import pytest

import app.rag_embeddings as rag


class _FakeEmbedder:
    def __init__(self, vector):
        self._vector = vector

    def embed(self, texts):
        for _ in texts:
            yield self._vector


def test_pack_unpack_roundtrip():
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    blob, dim = rag.pack_embedding(vec)
    out = rag.unpack_embedding(blob, dim)
    assert np.allclose(out, vec)


def test_get_embedder_raises_when_fastembed_is_missing(monkeypatch):
    monkeypatch.setattr(rag, "_FASTEMBED_AVAILABLE", False)

    with pytest.raises(RuntimeError, match="fastembed is required"):
        asyncio.run(rag.get_embedder())


def test_get_embedder_rejects_non_fastembed_backend(monkeypatch):
    monkeypatch.setenv("EMBED_BACKEND", "hash")
    monkeypatch.setattr(rag, "_FASTEMBED_AVAILABLE", True)

    with pytest.raises(RuntimeError, match="requires fastembed only"):
        asyncio.run(rag.get_embedder())


def test_embed_text_with_metadata_uses_fastembed(monkeypatch):
    async def fake_get_embedder(model_name=None):
        return _FakeEmbedder(np.array([0.4, 0.6], dtype=np.float32))

    monkeypatch.delenv("EMBED_BACKEND", raising=False)
    monkeypatch.setattr(rag, "get_embedder", fake_get_embedder)

    vector, metadata = asyncio.run(rag.embed_text_with_metadata("hello world", model_name="demo-model"))

    assert np.allclose(vector, np.array([0.4, 0.6], dtype=np.float32))
    assert metadata.backend == "fastembed"
    assert metadata.model == "demo-model"
    assert metadata.signature == "fastembed:demo-model"
