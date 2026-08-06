import socket

import app.zhihu_dl as zhihu_dl


def test_classify_link_distinguishes_answer_article_post_and_question():
    assert zhihu_dl.classify_link(
        "https://www.zhihu.com/question/123/answer/456"
    ) == "answer"
    assert zhihu_dl.classify_link("https://zhuanlan.zhihu.com/p/789") == "article"
    assert zhihu_dl.classify_link("https://www.zhihu.com/column/demo/p/789") == "article"
    assert zhihu_dl.classify_link("https://www.zhihu.com/pin/789") == "post"
    assert zhihu_dl.classify_link("https://www.zhihu.com/p/789") == "post"
    assert zhihu_dl.classify_link("https://www.zhihu.com/question/123") == "question"
    assert zhihu_dl.classify_link("https://notzhihu.com/question/123") is None
    assert zhihu_dl.classify_link("https://www.notzhihu.com/pin/789") is None


def test_question_resource_falls_back_to_first_answer_question_metadata(monkeypatch):
    class _Response:
        def __init__(self, payload=None, error=None):
            self.payload = payload
            self.error = error

        def raise_for_status(self):
            if self.error:
                raise self.error

        def json(self):
            return self.payload

    class _Session:
        def __init__(self):
            self.calls = []

        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if url.endswith("/questions/123"):
                return _Response(error=zhihu_dl.requests.HTTPError("403"))
            return _Response({
                "data": [{"question": {"id": "123", "title": "Question title", "detail": "Question detail"}}]
            })

        def close(self):
            return None

    session = _Session()
    monkeypatch.setattr(zhihu_dl, "build_session", lambda **kwargs: session)
    monkeypatch.setattr(
        zhihu_dl,
        "_fetch_page_fallback",
        lambda *args: (_ for _ in ()).throw(AssertionError("page fallback should not be needed")),
    )

    result = zhihu_dl.fetch_resource("https://www.zhihu.com/question/123", "question")

    assert result["content_type"] == "question"
    assert result["title"] == "Question title"
    assert result["content"] == "Question detail"
    assert session.calls[1][0].endswith("/questions/123/answers")
    assert session.calls[1][1]["params"]["include"] == zhihu_dl.QUESTION_ANSWERS_INCLUDE


def test_parse_link_dispatches_non_answer_content_to_resource_fetch(monkeypatch):
    calls = []

    def fake_fetch_resource(link, content_type, **kwargs):
        calls.append((link, content_type))
        return {
            "title": "An article",
            "author": {"name": "Author", "url_token": "author"},
            "content": "<p>Article body</p>",
        }

    monkeypatch.setattr(zhihu_dl, "fetch_resource", fake_fetch_resource)

    result = zhihu_dl.parse_link("https://zhuanlan.zhihu.com/p/789")

    assert calls == [("https://zhuanlan.zhihu.com/p/789", "article")]
    assert result["content_type"] == "article"
    assert result["title"] == "An article"
    assert result["content"] == "Article body"


def test_parse_link_keeps_answer_format_compatible(monkeypatch):
    monkeypatch.setattr(
        zhihu_dl,
        "fetch",
        lambda link, **kwargs: {
            "question": {"title": "Question"},
            "author": {"name": "Answerer", "url_token": "answerer"},
            "content": "<p>Answer body</p>",
        },
    )

    result = zhihu_dl.parse_link("https://www.zhihu.com/question/123/answer/456")

    assert result["content_type"] == "answer"
    assert result["question"] == "Question"
    assert result["content"] == "Answer body"


def test_fmt_extracts_image_urls_from_rich_and_plain_zhihu_content():
    result = zhihu_dl.fmt(
        {
            "question": {"title": "Question"},
            "author": {
                "name": "Answerer",
                "avatar_url": "https://picx.zhimg.com/author-avatar",
            },
            "content": (
                '<p>Text before</p>'
                '<p><img src="https://cdn.example.com/image/no-extension" /></p>'
                '<p>https://picx.zhimg.com/abc123</p>'
            ),
            "images": [{"url": "https://images.example.com/cover.png"}],
        }
    )

    assert result["image_urls"] == [
        "https://cdn.example.com/image/no-extension",
        "https://picx.zhimg.com/abc123",
        "https://images.example.com/cover.png",
    ]
    assert result["content"] == "Text before"


def test_fmt_does_not_treat_regular_body_links_as_images():
    result = zhihu_dl.fmt({"content": "Read more: https://example.com/article"})

    assert result["image_urls"] == []
    assert result["content"] == "Read more: https://example.com/article"


def test_fmt_extracts_srcset_and_poster_but_not_video_or_audio_sources():
    result = zhihu_dl.fmt(
        {
            "content": (
                '<img src="https://cdn.example.com/no-extension" '
                'srcset="https://cdn.example.com/one 1x, https://cdn.example.com/two 2x">'
                '<video src="https://cdn.example.com/video-no-extension" '
                'poster="https://cdn.example.com/poster-no-extension"></video>'
                '<audio src="https://cdn.example.com/audio-no-extension"></audio>'
            )
        }
    )

    assert result["image_urls"] == [
        "https://cdn.example.com/no-extension",
        "https://cdn.example.com/one",
        "https://cdn.example.com/two",
        "https://cdn.example.com/poster-no-extension",
    ]


def test_fmt_extracts_og_image_when_meta_attributes_are_reversed():
    result = zhihu_dl.fmt(
        {
            "html": '<meta content="https://cdn.example.com/og-image" property="og:image">'
        },
        content_type="article",
    )

    assert result["content_type"] == "article"
    assert result["image_urls"] == ["https://cdn.example.com/og-image"]
    assert result["content"] == "（无内容）"


def test_image_target_validation_rejects_private_and_credential_urls():
    assert zhihu_dl._is_safe_image_url("http://127.0.0.1/internal") is False
    assert zhihu_dl._is_safe_image_url("http://localhost/internal") is False
    assert zhihu_dl._is_safe_image_url("https://user:secret@example.com/image.jpg") is False
    assert zhihu_dl._is_safe_image_url("https://[broken/image.jpg") is False


def test_image_target_validation_accepts_public_dns_results(monkeypatch):
    monkeypatch.setattr(
        zhihu_dl.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("93.184.216.34", 443))
        ],
    )

    assert zhihu_dl._is_safe_image_url("https://cdn.example.com/image") is True


def test_download_image_media_uses_cookie_free_session_and_skips_non_images(monkeypatch):
    class _Response:
        def __init__(self, content_type, payload):
            self.headers = {"Content-Type": content_type}
            self.payload = payload
            self.closed = False

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            return [self.payload]

        def close(self):
            self.closed = True

    class _Session:
        def __init__(self):
            self.requests = []
            self.headers = {}

        def get(self, url, **kwargs):
            self.requests.append((url, kwargs))
            if "not-image" in url:
                return _Response("text/html", b"<html>blocked</html>")
            return _Response("image/png", b"\x89PNG\r\n\x1a\nbytes")

        def close(self):
            return None

    session = _Session()
    monkeypatch.setattr(zhihu_dl.requests, "Session", lambda: session)
    monkeypatch.setattr(zhihu_dl, "_is_safe_image_url", lambda url: True)

    result = zhihu_dl.download_image_media(
        [
            "https://picx.zhimg.com/image.png",
            "https://picx.zhimg.com/not-image",
            "https://picx.zhimg.com/image.png",
        ]
    )

    assert len(session.requests) == 2
    assert "Referer" not in session.headers
    assert len(result) == 1
    assert result[0]["filename"] == "zhihu_image_1.png"
    assert result[0]["content"] == b"\x89PNG\r\n\x1a\nbytes"
