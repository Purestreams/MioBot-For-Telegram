import asyncio

import numpy as np

import app.rag_embeddings as rag


def test_hash_embed_empty_returns_zero_vector(monkeypatch):
    monkeypatch.setattr(rag, "_HASH_DIM", 16)
    vec = rag._hash_embed("")
    assert vec.shape == (16,)
    assert np.allclose(vec, np.zeros((16,), dtype=np.float32))


def test_pack_unpack_roundtrip():
    vec = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    blob, dim = rag.pack_embedding(vec)
    out = rag.unpack_embedding(blob, dim)
    assert np.allclose(out, vec)


def test_embed_text_uses_hash_backend_when_configured(monkeypatch):
    monkeypatch.setattr(rag, "_EMBED_BACKEND", "hash")
    monkeypatch.setattr(rag, "_HASH_DIM", 32)
    out = asyncio.run(rag.embed_text("hello world"))
    assert out.shape == (32,)
    assert np.linalg.norm(out) > 0
