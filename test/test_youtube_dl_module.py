import asyncio
from pathlib import Path

import app.youtube_dl as youtube_dl


class _Response:
    def __init__(self, url: str, headers=None, status_code: int = 200):
        self.url = url
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self):
        return None


class _Client:
    def __init__(self, response, get_response=None):
        self._response = response
        self._get_response = get_response or response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def head(self, url):
        return self._response

    async def get(self, url, follow_redirects=True):
        return self._get_response


def test_get_bilibili_permanent_url_extracts_canonical_url(monkeypatch):
    response = _Response(
        "https://b23.tv/xyz",
        status_code=302,
        headers={"location": "https://www.bilibili.com/video/BV1TuLA6ZEE7?p=1"},
    )
    monkeypatch.setattr(youtube_dl.httpx, "AsyncClient", lambda **kwargs: _Client(response))

    result = asyncio.run(youtube_dl.get_bilibili_permanent_url("https://b23.tv/xyz"))
    assert result == "https://www.bilibili.com/video/BV1TuLA6ZEE7/"


def test_get_bilibili_permanent_url_returns_none_when_invalid_input():
    assert asyncio.run(youtube_dl.get_bilibili_permanent_url("")) is None


def test_bilibili_canonicalizer_supports_av_and_query_style_links():
    assert youtube_dl._extract_bilibili_canonical_url(
        "https://www.bilibili.com/video/av170001?p=2"
    ) == "https://www.bilibili.com/video/av170001/"
    assert youtube_dl._extract_bilibili_canonical_url(
        "bilibili.com/watch?bvid=BV1TuLA6ZEE7&from=share"
    ) == "https://www.bilibili.com/video/BV1TuLA6ZEE7/"
    assert youtube_dl._extract_bilibili_canonical_url(
        "https://evil.example/video/BV1TuLA6ZEE7"
    ) is None


def test_bilibili_matcher_accepts_short_links_and_canonical_av_links():
    assert youtube_dl._is_bilibili_url("https://b23.tv/Enqggyo")
    assert youtube_dl._is_bilibili_url("https://www.bilibili.com/video/av170001")
    assert youtube_dl._is_bilibili_url("https://www.bilibili.com/watch?bvid=BV1TuLA6ZEE7")
    assert not youtube_dl._is_bilibili_url("https://foo.bilibili.com/video/BV1TuLA6ZEE7")


def test_get_bilibili_permanent_url_fallbacks_to_followed_get(monkeypatch):
    head_response = _Response("https://b23.tv/abc123", status_code=302, headers={})
    get_response = _Response("https://www.bilibili.com/video/BV1TuLA6ZEE7?from=share", status_code=200)
    monkeypatch.setattr(
        youtube_dl.httpx,
        "AsyncClient",
        lambda **kwargs: _Client(head_response, get_response=get_response),
    )

    result = asyncio.run(youtube_dl.get_bilibili_permanent_url("https://b23.tv/abc123"))
    assert result == "https://www.bilibili.com/video/BV1TuLA6ZEE7/"


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


def test_resolve_download_candidates_prefers_canonical_bilibili(monkeypatch):
    async def _fake_get_bilibili_permanent_url(_url):
        return "https://www.bilibili.com/video/BV1TuLA6ZEE7/"

    monkeypatch.setattr(youtube_dl, "get_bilibili_permanent_url", _fake_get_bilibili_permanent_url)

    candidates = asyncio.run(youtube_dl._resolve_download_candidates("https://b23.tv/abc123"))

    assert candidates == [
        "https://www.bilibili.com/video/BV1TuLA6ZEE7/",
        "https://b23.tv/abc123",
    ]


def test_download_video_to_file_retries_next_candidate_when_first_fails(monkeypatch, tmp_path: Path):
    attempted = []

    async def _fake_get_video_title(_url):
        return "Title"

    async def _fake_resolve_download_candidates(_url):
        return ["https://www.bilibili.com/video/BV1TuLA6ZEE7/", "https://b23.tv/abc123"]

    async def _fake_download_video_720p_h264(url, output_path='output/%(title)s.%(ext)s'):
        attempted.append((url, output_path))
        if len(attempted) == 1:
            # Create a partial file to make sure cleanup runs before retry.
            with open(output_path, "wb") as handle:
                handle.write(b"partial")
            raise RuntimeError("first candidate failed")

    output_path = tmp_path / "video.mp4"

    monkeypatch.setattr(youtube_dl, "get_video_title", _fake_get_video_title)
    monkeypatch.setattr(youtube_dl, "_resolve_download_candidates", _fake_resolve_download_candidates)
    monkeypatch.setattr(youtube_dl, "download_video_720p_h264", _fake_download_video_720p_h264)

    title = asyncio.run(youtube_dl.download_video_to_file("https://b23.tv/abc123", str(output_path)))

    assert title == "Title"
    assert attempted[0][0] == "https://www.bilibili.com/video/BV1TuLA6ZEE7/"
    assert attempted[1][0] == "https://b23.tv/abc123"


def test_resolve_caption_url_returns_clean_bilibili_canonical(monkeypatch):
    async def _fake_get_bilibili_permanent_url(_url):
        return "https://www.bilibili.com/video/BV1TuLA6ZEE7/"

    monkeypatch.setattr(youtube_dl, "get_bilibili_permanent_url", _fake_get_bilibili_permanent_url)

    result = asyncio.run(youtube_dl.resolve_caption_url("https://b23.tv/gv547yI"))

    assert result == "https://www.bilibili.com/video/BV1TuLA6ZEE7/"


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
