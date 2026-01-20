import asyncio
import logging
import os
import importlib
import zlib
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_FASTEMBED_AVAILABLE: Optional[bool] = None


_DEFAULT_EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
_EMBED_BACKEND = os.getenv("EMBED_BACKEND", "fastembed")  # fastembed|hash
_HASH_DIM = int(os.getenv("EMBED_HASH_DIM", "512"))

_embedder = None
_embedder_model_name: Optional[str] = None
_embedder_lock = asyncio.Lock()


def _fastembed_is_available() -> bool:
    global _FASTEMBED_AVAILABLE
    if _FASTEMBED_AVAILABLE is not None:
        return _FASTEMBED_AVAILABLE
    try:
        importlib.import_module("fastembed")
        _FASTEMBED_AVAILABLE = True
    except Exception:
        _FASTEMBED_AVAILABLE = False
    return _FASTEMBED_AVAILABLE


async def get_embedder(model_name: Optional[str] = None):
    """Return a singleton FastEmbed embedder.

    Uses a single instance to avoid repeatedly loading the ONNX runtime + model.
    """
    if not _fastembed_is_available():
        raise RuntimeError(
            "fastembed is not available in this Python environment. "
            "(On Python 3.14, onnxruntime wheels may be unavailable.)"
        )

    chosen = model_name or _DEFAULT_EMBED_MODEL

    global _embedder, _embedder_model_name
    if _embedder is not None and _embedder_model_name == chosen:
        return _embedder

    async with _embedder_lock:
        if _embedder is not None and _embedder_model_name == chosen:
            return _embedder

        logger.info("Initializing fastembed model: %s", chosen)
        fastembed = importlib.import_module("fastembed")
        TextEmbedding = getattr(fastembed, "TextEmbedding")
        _embedder = TextEmbedding(model_name=chosen)
        _embedder_model_name = chosen
        return _embedder


async def embed_text(text: str, *, model_name: Optional[str] = None) -> np.ndarray:
    """Embed a single piece of text to a float32 numpy vector.

    Preferred backend: fastembed (local ONNX embeddings).
    Fallback backend: pure-numpy hashed char-ngram vector (still vector search, fully local).
    """
    backend = _EMBED_BACKEND
    if backend == "fastembed" and _fastembed_is_available():
        embedder = await get_embedder(model_name=model_name)

        def _embed_sync() -> np.ndarray:
            vec = next(embedder.embed([text]))
            return np.asarray(vec, dtype=np.float32)

        return await asyncio.to_thread(_embed_sync)

    return await asyncio.to_thread(_hash_embed, text)


def _hash_embed(text: str) -> np.ndarray:
    s = (text or "").strip().lower()
    if not s:
        return np.zeros((_HASH_DIM,), dtype=np.float32)

    vec = np.zeros((_HASH_DIM,), dtype=np.float32)
    b = s.encode("utf-8", errors="ignore")

    # Character n-grams (3..5) over bytes gives reasonable multilingual robustness.
    for n in (3, 4, 5):
        if len(b) < n:
            continue
        for i in range(0, len(b) - n + 1):
            ng = b[i : i + n]
            idx = zlib.crc32(ng) % _HASH_DIM
            vec[idx] += 1.0

    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


def pack_embedding(vec: np.ndarray) -> tuple[bytes, int]:
    """Serialize an embedding vector to bytes + dim for SQLite storage."""
    vec32 = np.asarray(vec, dtype=np.float32)
    return vec32.tobytes(), int(vec32.shape[0])


def unpack_embedding(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize bytes + dim to a float32 numpy vector."""
    arr = np.frombuffer(blob, dtype=np.float32)
    if dim and arr.shape[0] != dim:
        # If dim mismatches, trust actual buffer length.
        return arr.astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)
