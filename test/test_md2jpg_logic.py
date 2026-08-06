import asyncio
from pathlib import Path

import app.md2jpg as md2jpg


class _FakeImg:
    def __init__(self):
        self.mode = "RGB"
        self.saved = []

    def convert(self, mode):
        self.mode = mode
        return self

    def save(self, path, *args, **kwargs):
        self.saved.append((path, args, kwargs))
        Path(path).write_bytes(b"img")


class _FakePage:
    def __init__(self):
        self.routes = []

    async def route(self, pattern, handler):
        self.routes.append((pattern, handler))

    async def set_content(self, html):
        self.html = html

    async def set_viewport_size(self, vp):
        self.vp = vp

    async def screenshot(self, path, full_page=True):
        Path(path).write_bytes(b"png")


class _FakeBrowser:
    async def new_page(self, device_scale_factor=1):
        return _FakePage()

    async def close(self):
        return None


class _FakeChromium:
    async def launch(self):
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self):
        self.chromium = _FakeChromium()


class _FakeContextManager:
    async def __aenter__(self):
        return _FakePlaywright()

    async def __aexit__(self, exc_type, exc, tb):
        return False


def test_md_to_image_appends_jpg_when_no_extension(monkeypatch, tmp_path):
    monkeypatch.setattr(md2jpg, "async_playwright", lambda: _FakeContextManager())
    monkeypatch.setattr(md2jpg.Image, "open", lambda path: _FakeImg())

    out_base = tmp_path / "rendered"
    asyncio.run(md2jpg.md_to_image("# title", output_path=str(out_base), theme="formal_code"))

    assert (tmp_path / "rendered.jpg").exists()


def test_md_to_image_escapes_raw_html_and_blocks_network(monkeypatch, tmp_path):
    page = _FakePage()

    class _Browser(_FakeBrowser):
        async def new_page(self, device_scale_factor=1):
            return page

    class _Chromium(_FakeChromium):
        async def launch(self):
            return _Browser()

    class _Playwright(_FakePlaywright):
        def __init__(self):
            self.chromium = _Chromium()

    class _Context(_FakeContextManager):
        async def __aenter__(self):
            return _Playwright()

    monkeypatch.setattr(md2jpg, "async_playwright", lambda: _Context())
    monkeypatch.setattr(md2jpg.Image, "open", lambda path: _FakeImg())

    asyncio.run(
        md2jpg.md_to_image(
            '<img src="http://127.0.0.1/private">',
            output_path=str(tmp_path / "rendered"),
        )
    )

    assert "&lt;img" in page.html
    assert page.routes and page.routes[0][0] == "**/*"
