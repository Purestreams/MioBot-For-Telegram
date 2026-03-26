import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import ModuleType, SimpleNamespace


yt_dlp_stub = ModuleType("yt_dlp")
yt_dlp_stub.YoutubeDL = object
httpx_stub = ModuleType("httpx")
httpx_stub.AsyncClient = object
sys.modules.setdefault("yt_dlp", yt_dlp_stub)
sys.modules.setdefault("httpx", httpx_stub)

from app.youtube_dl import compress_video_if_needed


class TestCompressVideoIfNeeded(unittest.IsolatedAsyncioTestCase):
    async def test_returns_original_path_when_within_limit(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(b"small-video")
            input_path = tmp.name

        try:
            result = await compress_video_if_needed(input_path, max_size_bytes=1024)
            self.assertEqual(result, input_path)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)

    async def test_compresses_and_returns_new_path_when_oversized(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(b"x" * 2048)
            input_path = tmp.name

        compressed_path = input_path.replace(".mp4", "_compressed.mp4")

        def _fake_subprocess_run(*args, **kwargs):
            with open(compressed_path, "wb") as f:
                f.write(b"y" * 128)

        try:
            with patch("app.youtube_dl.subprocess.run", side_effect=_fake_subprocess_run):
                result = await compress_video_if_needed(input_path, max_size_bytes=1024)
            self.assertEqual(result, compressed_path)
            self.assertTrue(os.path.exists(compressed_path))
            self.assertLessEqual(os.path.getsize(compressed_path), 1024)
        finally:
            if os.path.exists(input_path):
                os.remove(input_path)
            if os.path.exists(compressed_path):
                os.remove(compressed_path)


if __name__ == "__main__":
    unittest.main()
