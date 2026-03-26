import numpy as np

from app.database import MessageRow, _cosine_top_k, _format_message


def test_format_message_truncates_long_content():
    row = MessageRow(
        id=1,
        chat_id=10,
        username="alice",
        content="x" * 20,
        timestamp="2026-03-24 00:00:00",
    )
    text = _format_message(row, max_chars=10)
    assert text.startswith("[2026-03-24 00:00:00] alice: ")
    assert text.endswith("…")


def test_cosine_top_k_returns_highest_similarity_indices():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    idx = _cosine_top_k(query, matrix, top_k=2)
    # First vector should rank highest, second vector should be next.
    assert list(idx) == [0, 1]


def test_get_env_int_falls_back_for_invalid_value(monkeypatch):
    from app.database import _get_env_int

    monkeypatch.setenv("RAG_TOP_K", "not-an-int")
    assert _get_env_int("RAG_TOP_K", 12) == 12
