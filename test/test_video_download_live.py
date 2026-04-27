import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from app.twitter_downloader import TwitterDownloader
from app.youtube_dl import download_video_to_file


RUN_LIVE_DOWNLOAD_TESTS = os.getenv("RUN_LIVE_VIDEO_DOWNLOAD_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_LIVE_DOWNLOAD_TESTS,
    reason="set RUN_LIVE_VIDEO_DOWNLOAD_TESTS=1 to run live network download checks",
)


YOUTUBE_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
BILIBILI_URL = "https://b23.tv/Enqggyo"
TWITTER_VIDEO_URL = "https://x.com/NASA/status/2048166203735040057"


def _require_ffprobe() -> str:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        pytest.skip("ffprobe is required for live playable-video checks")
    return ffprobe_path


def _assert_playable_video_file(file_path: Path) -> None:
    ffprobe_path = _require_ffprobe()
    result = subprocess.run(
        [
            ffprobe_path,
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,codec_name:format=duration,size",
            "-of",
            "json",
            str(file_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout or "{}")
    streams = payload.get("streams") or []
    fmt = payload.get("format") or {}

    assert file_path.exists()
    assert file_path.stat().st_size > 0
    assert any(stream.get("codec_type") == "video" for stream in streams)
    assert float(fmt.get("duration") or 0.0) > 0.0


def test_live_youtube_download_is_playable(tmp_path: Path):
    output_path = tmp_path / "youtube.mp4"

    title = asyncio.run(download_video_to_file(YOUTUBE_URL, str(output_path)))

    assert title
    _assert_playable_video_file(output_path)


def test_live_bilibili_download_is_playable(tmp_path: Path):
    output_path = tmp_path / "bilibili.mp4"

    title = asyncio.run(download_video_to_file(BILIBILI_URL, str(output_path)))

    assert title
    _assert_playable_video_file(output_path)


def test_live_twitter_download_is_playable(tmp_path: Path):
    downloader = TwitterDownloader()
    media, text = downloader.extract_twitter_media(TWITTER_VIDEO_URL)

    assert text
    video_payloads = [payload for kind, payload in media if kind == "vid"]
    assert video_payloads, "expected at least one direct Twitter video payload"

    output_path = tmp_path / "twitter.mp4"
    output_path.write_bytes(video_payloads[0])
    _assert_playable_video_file(output_path)