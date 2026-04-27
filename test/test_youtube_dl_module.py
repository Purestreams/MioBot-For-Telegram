import asyncio
from pathlib import Path

import app.youtube_dl as youtube_dl


class _Response:
    def __init__(self, url: str, headers=None):
        self.url = url
        self.headers = headers or {}

    def raise_for_status(self):
        return None


class _Client:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def head(self, url):
        return self._response


def test_get_bilibili_permanent_url_extracts_canonical_url(monkeypatch):
    response = _Response(
        "https://b23.tv/xyz",
        headers={"location": "https://www.bilibili.com/video/BV1abc1234?p=1"},
    )
    monkeypatch.setattr(youtube_dl.httpx, "AsyncClient", lambda **kwargs: _Client(response))

    result = asyncio.run(youtube_dl.get_bilibili_permanent_url("https://b23.tv/xyz"))
    assert result == "https://www.bilibili.com/video/BV1abc1234"


def test_get_bilibili_permanent_url_returns_none_when_invalid_input():
    assert asyncio.run(youtube_dl.get_bilibili_permanent_url("")) is None


def test_resolve_caption_url_returns_original_for_non_bilibili():
    result = asyncio.run(youtube_dl.resolve_caption_url("https://example.com/video"))
    assert result == "https://example.com/video"


def test_download_video_to_file_returns_fallback_title(monkeypatch):
    called = {}

    async def _fake_get_video_title(_url):
        return None

    async def _fake_download_video_720p_h264(url, output_path='output/%(title)s.%(ext)s'):
        called["url"] = url
        called["output_path"] = output_path

    monkeypatch.setattr(youtube_dl, "get_video_title", _fake_get_video_title)
    monkeypatch.setattr(youtube_dl, "download_video_720p_h264", _fake_download_video_720p_h264)

    title = asyncio.run(youtube_dl.download_video_to_file("https://example.com/v", "output/test.mp4"))

    assert title == "Video"
    assert called == {"url": "https://example.com/v", "output_path": "output/test.mp4"}


def test_normalize_output_path_truncates_overlong_filename(tmp_path: Path):
    long_name = "a" * 400 + ".mp4"

    normalized_path = youtube_dl._normalize_output_path(str(tmp_path / long_name))

    assert Path(normalized_path).parent == tmp_path
    assert len(Path(normalized_path).name.encode("utf-8")) <= youtube_dl.MAX_FILENAME_BYTES
    assert normalized_path.endswith(".mp4")


def test_build_compressed_output_path_keeps_filename_within_limit(tmp_path: Path):
    long_input = tmp_path / (("中" * 180) + ".mp4")

    compressed_path = youtube_dl._build_compressed_output_path(str(long_input))

    assert Path(compressed_path).parent == tmp_path
    assert len(Path(compressed_path).name.encode("utf-8")) <= youtube_dl.MAX_FILENAME_BYTES
    assert compressed_path.endswith("_compressed.mp4")
