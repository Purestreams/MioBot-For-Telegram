import asyncio
from dataclasses import dataclass
import logging
import importlib
from typing import Optional

import numpy as np
from app.runtime_config import get_runtime_value

logger = logging.getLogger(__name__)

_FASTEMBED_AVAILABLE: Optional[bool] = None


_DEFAULT_EMBED_MODEL = get_runtime_value("EMBED_MODEL")

_embedder = None
_embedder_model_name: Optional[str] = None
_embedder_lock = asyncio.Lock()


@dataclass(frozen=True)
class EmbeddingMetadata:
    backend: str
    model: str
    dim: int
    signature: str


def _configured_embed_backend() -> str:
    backend = (get_runtime_value("EMBED_BACKEND") or "fastembed").strip().lower()
    return backend or "fastembed"


def _validate_embed_backend_config() -> None:
    backend = _configured_embed_backend()
    if backend != "fastembed":
        raise RuntimeError(
            f"Unsupported EMBED_BACKEND={backend!r}. MioBot now requires fastembed only."
        )


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
    _validate_embed_backend_config()
    if not _fastembed_is_available():
        raise RuntimeError(
            "fastembed is required for MioBot embeddings but is not available in this Python environment. "
            "Install fastembed in the active environment before starting the bot."
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

    MioBot requires fastembed for semantic retrieval.
    """
    vector, _ = await embed_text_with_metadata(text, model_name=model_name)
    return vector


async def embed_text_with_metadata(
    text: str,
    *,
    model_name: Optional[str] = None,
) -> tuple[np.ndarray, EmbeddingMetadata]:
    """Embed text and return the vector plus the actual runtime embedding metadata."""
    embedder = await get_embedder(model_name=model_name)
    chosen_model = model_name or _DEFAULT_EMBED_MODEL

    def _embed_sync() -> np.ndarray:
        vec = next(embedder.embed([text]))
        return np.asarray(vec, dtype=np.float32)

    vector = await asyncio.to_thread(_embed_sync)
    metadata = EmbeddingMetadata(
        backend="fastembed",
        model=chosen_model,
        dim=int(vector.shape[0]),
        signature=f"fastembed:{chosen_model}",
    )
    return vector, metadata


async def get_runtime_embedding_metadata(*, model_name: Optional[str] = None) -> EmbeddingMetadata:
    """Return metadata describing the embedding backend currently used at runtime."""
    _, metadata = await embed_text_with_metadata("embedding healthcheck", model_name=model_name)
    return metadata


async def ensure_fastembed_ready(*, model_name: Optional[str] = None) -> EmbeddingMetadata:
    """Validate fastembed availability and model initialization for startup fail-fast checks."""
    return await get_runtime_embedding_metadata(model_name=model_name)


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
