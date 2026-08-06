import asyncio
from types import SimpleNamespace

import main
import pytest
from app.main_helpers import classify_zhihu_url, extract_supported_links_from_message, is_zhihu_answer_url
from app.twitter_downloader import format_tweet_text_for_reply, summarize_tweet_text


def test_extract_video_url_prefers_youtube_when_present():
    message = "check this https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    extracted = main._extract_video_url(message)
    assert extracted is not None
    assert "youtube.com" in extracted


def test_extract_video_url_supports_zhihu_answer_links():
    message = "看看这个 https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097"
    extracted = main._extract_video_url(message)
    assert extracted == "https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097"


def test_zhihu_url_classification_does_not_treat_questions_as_answers():
    assert classify_zhihu_url("https://www.zhihu.com/question/123/answer/456") == "answer"
    assert classify_zhihu_url("https://zhuanlan.zhihu.com/p/789") == "article"
    assert classify_zhihu_url("https://www.zhihu.com/pin/789") == "post"
    assert classify_zhihu_url("https://www.zhihu.com/question/123") == "question"
    assert is_zhihu_answer_url("https://www.zhihu.com/question/123") is False
    assert main.extract_supported_links(
        "https://zhuanlan.zhihu.com/p/789 https://www.zhihu.com/pin/987"
    ) == [
        "https://zhuanlan.zhihu.com/p/789",
        "https://www.zhihu.com/pin/987",
    ]


def test_extract_supported_links_keeps_all_mixed_links_in_source_order():
    message = (
        "text https://www.youtube.com/watch?v=dQw4w9WgXcQ. "
        "post x.com/user/status/123, article "
        "https://www.zhihu.com/question/1/answer/2; video b23.tv/BV1xx411c7mD!"
    )

    assert main.extract_supported_links(message) == [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://x.com/user/status/123",
        "https://www.zhihu.com/question/1/answer/2",
        "https://b23.tv/BV1xx411c7mD",
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("youtu.be/dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ"),
        (
            "youtube.com/shorts/dQw4w9WgXcQ?t=30&si=abc&feature=share",
            "https://youtube.com/shorts/dQw4w9WgXcQ?t=30",
        ),
        (
            "www.bilibili.com/watch?bvid=BV1TuLA6ZEE7&p=2&spm_id_from=333.999",
            "https://www.bilibili.com/watch?bvid=BV1TuLA6ZEE7&p=2",
        ),
        (
            "x.com/user/status/123?t=abc&s=20",
            "https://x.com/user/status/123",
        ),
        (
            "zhuanlan.zhihu.com/p/789?utm_source=share",
            "https://zhuanlan.zhihu.com/p/789",
        ),
        (
            "www.zhihu.com/people/example/pins/987?share_code=x",
            "https://www.zhihu.com/people/example/pins/987",
        ),
        (
            "www.zhihu.com/question/123?utm_id=1",
            "https://www.zhihu.com/question/123",
        ),
    ],
)
def test_extract_supported_links_normalizes_variants_and_removes_tracking(source, expected):
    assert main.extract_supported_links(source) == [expected]


def test_extract_supported_links_deduplicates_urls_after_tracking_cleanup():
    assert main.extract_supported_links(
        "https://www.zhihu.com/pin/789?share_code=alice "
        "https://www.zhihu.com/pin/789?share_code=bob"
    ) == ["https://www.zhihu.com/pin/789"]


def test_extract_supported_links_rejects_provider_names_inside_other_hosts():
    assert main.extract_supported_links(
        "notyoutube.com/watch?v=dQw4w9WgXcQ foo.x.com/user/status/123 "
        "foo.bilibili.com/video/BV1xx411c7mD foo.zhihu.com/pin/789"
    ) == []
    assert main.extract_supported_links(
        "https://www.youtube.com/watch?v=dQw4w9WgXcQx "
        "https://x.com/user/status/1234x"
    ) == []


def test_extract_supported_links_from_message_includes_caption_text_links():
    message = SimpleNamespace(
        text=None,
        caption="look at this image",
        entities=None,
        caption_entities=[SimpleNamespace(url="https://x.com/user/status/123")],
    )

    assert extract_supported_links_from_message(message) == ["https://x.com/user/status/123"]


def test_extract_search_keywords_deduplicates_and_filters_stopwords():
    text = "This is this a test test for bot bot context retrieval"
    keywords = main._extract_search_keywords(text)
    assert "this" not in keywords
    assert "is" not in keywords
    assert keywords.count("test") == 1


def test_match_command_payload_reads_wrapped_content():
    content = main._match_command_payload("/md2jpg ,,,# title,,,", main.MD2JPG_REGEX)
    assert content == "# title"


def test_build_rag_query_uses_keywords():
    query = main._build_rag_query_from_message("hello hello world and world")
    assert query == "hello world"


def test_build_rag_query_includes_message_specific_context():
    query = main._build_rag_query_from_message(
        "mioo 看一下",
        additional_context=[
            "replied_to_content: previous deployment failed on sqlite lock",
            "sticker_description: annoyed face with error text",
            "user_personal_memory:\nshould not become retrieval query",
        ],
        sender_display="Alice @alice",
    )

    assert "mioo" in query
    assert "previous deployment failed on sqlite lock" in query
    assert "annoyed face with error text" in query
    assert "Alice @alice" in query
    assert "should not become retrieval query" not in query


def test_classify_group_reply_trigger_detects_username_mention():
    trigger = main._classify_group_reply_trigger("hey @MioooooooooBot look here", "MioooooooooBot")
    assert trigger == "username_mention"


def test_classify_group_reply_trigger_detects_alias_mention():
    trigger = main._classify_group_reply_trigger("mioo look here", "MioooooooooBot")
    assert trigger == "alias_mention"


def test_classify_group_reply_trigger_ignores_embedded_alias_text():
    trigger = main._classify_group_reply_trigger("amiooops should stay ambient", "MioooooooooBot")
    assert trigger == "ambient"


def test_is_reply_to_this_bot_matches_username_case_insensitively():
    update = SimpleNamespace(
        message=SimpleNamespace(
            reply_to_message=SimpleNamespace(
                from_user=SimpleNamespace(is_bot=True, username="MioBot", id=777)
            )
        )
    )

    assert main._is_reply_to_this_bot(update, "miobot") is True


def test_is_reply_to_this_bot_requires_known_bot_identity():
    update = SimpleNamespace(
        message=SimpleNamespace(
            reply_to_message=SimpleNamespace(
                from_user=SimpleNamespace(is_bot=True, username=None, id=777)
            )
        )
    )

    assert main._is_reply_to_this_bot(update, None) is False


def test_telegram_user_key_from_user_uses_stable_telegram_user_id():
    user = SimpleNamespace(id=123456789, username="alice")
    assert main._telegram_user_key_from_user(user) == "tg_user:123456789"


def test_tweet_text_summary_prefers_first_value():
    summary = summarize_tweet_text({"1": "hello twitter"})
    assert summary == "hello twitter"


def test_truncate_caption_text_caps_telegram_caption_length():
    caption = "x" * (main.TELEGRAM_CAPTION_LIMIT + 50)

    truncated = main._truncate_caption_text(caption)

    assert len(truncated) == main.TELEGRAM_CAPTION_LIMIT
    assert truncated.endswith("...")


def test_format_tweet_text_for_reply_keeps_tweet_body_only():
    formatted = format_tweet_text_for_reply(
        "Manus CEO没有政治敏感度，说让你开会你真的去开会…… 这下好了，这辈子出不去了 — 面包🍞 (@himself65) March 25, 2026",
        "https://x.com/himself65/status/2036933945200406781?s=46&t=6C5C8msOW1klCHHbUUlASA",
    )

    assert formatted == "Manus CEO没有政治敏感度，说让你开会你真的去开会…… 这下好了，这辈子出不去了"


def test_build_help_text_lists_current_features():
    help_text = main._build_help_text()

    for expected in (
        "/start",
        "/help",
        "/md2jpg",
        "/text2jpg",
        ".txt or .md",
        "Zhihu",
        "text/photo/sticker",
        "/med2jpg",
        "/crypto",
        "/memory_help",
        "/memory_refresh",
        "TELEGRAM_ADMIN_USER_IDS",
    ):
        assert expected in help_text


def test_handle_help_replies_with_feature_list():
    class _FakeMessage:
        def __init__(self):
            self.replies = []

        async def reply_text(self, text, **kwargs):
            self.replies.append(text)

    message = _FakeMessage()
    update = SimpleNamespace(message=message)

    asyncio.run(main.handle_help(update, SimpleNamespace()))

    assert len(message.replies) == 1
    assert "MioBot help" in message.replies[0]
    assert "/crypto" in message.replies[0]


def test_start_points_to_help_command():
    class _FakeMessage:
        def __init__(self):
            self.replies = []

        async def reply_text(self, text, **kwargs):
            self.replies.append(text)

    message = _FakeMessage()
    update = SimpleNamespace(message=message)

    asyncio.run(main.start(update, SimpleNamespace()))

    assert message.replies == [
        "Hi! I can render text to images, download media links, and join group chats with contextual replies. Send /help to see all features."
    ]


def test_handle_twitter_media_message_sends_images_and_text(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.message_id = 123
            self.replies = []
            self.deleted = False

        async def reply_text(self, text, **kwargs):
            self.replies.append({"text": text, "kwargs": kwargs})

        async def delete(self):
            self.deleted = True

    class _FakeBot:
        def __init__(self):
            self.photos = []
            self.media_groups = []
            self.videos = []
            self.documents = []
            self.messages = []

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_media_group(self, **kwargs):
            self.media_groups.append(kwargs)

        async def send_video(self, **kwargs):
            self.videos.append(kwargs)

        async def send_document(self, **kwargs):
            self.documents.append(kwargs)

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

    class _FakeStatus:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class _FakeTwitterDownloader:
        def extract_twitter_media(self, url):
            return [("pic", b"img1"), ("pic", b"img2")], {"1": "tweet body"}

    async def fake_add_message(**kwargs):
        return 1

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())
    monkeypatch.setattr(main, "add_message", fake_add_message)

    handled = asyncio.run(
        main._handle_twitter_media_message(
            update=update,
            context=context,
            video_url="https://x.com/u/status/1",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert handled is True
    assert message.replies == []
    assert bot.photos == []
    assert len(bot.media_groups) == 1
    group_payload = bot.media_groups[0]
    assert group_payload["chat_id"] == 1
    media_items = group_payload["media"]
    assert len(media_items) == 2
    assert media_items[0].caption == (
        'tweet body\n'
        '-- Posted by <a href="https://x.com/u/status/1">@u</a>\n\n'
        'Requested by Tester @tester'
    )
    assert media_items[0].parse_mode == main.ParseMode.HTML
    assert media_items[1].caption is None
    assert len(bot.documents) == 0
    assert bot.messages == []
    assert bot.videos == []
    assert status.deleted is True
    assert message.deleted is True


def test_handle_zhihu_link_message_sends_text_and_persists_content(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.text = "帮我总结 https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097"
            self.message_id = 808
            self.replies = []
            self.deleted = False
            self.reply_to_message = None

        async def reply_text(self, text, **kwargs):
            self.replies.append({"text": text, "kwargs": kwargs})

        async def delete(self):
            self.deleted = True

    class _FakeBot:
        def __init__(self):
            self.messages = []

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

    class _FakeStatus:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    captured = {}

    def fake_parse_zhihu_link(url):
        assert url == "https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097"
        return {
            "question": "00后是否会更加认可自由主义？",
            "author": "Allen",
            "author_url": "chen-shi-xuan-44",
            "content": "目前的趋势是00后要用一生的代价来认可自由主义。",
            "time": "2026-04-28 05:07",
        }

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        captured["chat_id"] = chat_id
        captured["username"] = username
        captured["content"] = content
        captured["kwargs"] = kwargs
        return 1

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1),
        effective_user=SimpleNamespace(full_name="Tester", username="tester", id=42),
    )
    context = SimpleNamespace(bot=bot)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(main, "parse_zhihu_link", fake_parse_zhihu_link)
    monkeypatch.setattr(main, "add_message", fake_add_message)
    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)

    handled = asyncio.run(
        main._handle_zhihu_link_message(
            update=update,
            context=context,
            video_url="https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert handled is True
    assert captured["chat_id"] == 1
    assert captured["username"] == "Tester @tester"
    assert "shared_zhihu_link: https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097" in captured["content"]
    assert "user_comment: 帮我总结" in captured["content"]
    assert "zhihu_question: 00后是否会更加认可自由主义？" in captured["content"]
    assert "zhihu_answer: 目前的趋势是00后要用一生的代价来认可自由主义。" in captured["content"]
    assert captured["kwargs"]["telegram_user_key"] == "tg_user:42"
    assert bot.messages == [
        {
            "chat_id": 1,
            "text": (
                "00后是否会更加认可自由主义？\n\n"
                "目前的趋势是00后要用一生的代价来认可自由主义。\n"
                "-- Allen (@chen-shi-xuan-44) · 2026-04-28 05:07\n\n"
                "https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097\n"
                "Requested by: Tester @tester"
            ),
            "disable_web_page_preview": True,
        }
    ]
    assert status.deleted is True
    assert message.deleted is True


def test_handle_zhihu_link_message_sends_extracted_images_after_text(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.text = "see https://www.zhihu.com/question/1/answer/2"
            self.message_id = 809
            self.reply_to_message = None
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class _FakeBot:
        def __init__(self):
            self.messages = []
            self.photos = []
            self.media_groups = []
            self.documents = []

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_media_group(self, **kwargs):
            self.media_groups.append(kwargs)

        async def send_document(self, **kwargs):
            self.documents.append(kwargs)

    class _FakeStatus:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    async def fake_add_message(**kwargs):
        return 1

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(
        main,
        "parse_zhihu_link",
        lambda url: {
            "content_type": "answer",
            "question": "Question",
            "author": "Answerer",
            "content": "Answer body",
            "time": "2026-08-05 10:00",
            "image_urls": [
                "https://picx.zhimg.com/one",
                "https://picx.zhimg.com/two",
            ],
        },
    )
    monkeypatch.setattr(
        main,
        "download_zhihu_image_media",
        lambda urls: [
            {"content": b"image-one", "filename": "one.jpg"},
            {"content": b"image-two", "filename": "two.jpg"},
        ],
    )
    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(main, "add_message", fake_add_message)

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1),
        effective_user=SimpleNamespace(full_name="Tester", username="tester", id=42),
    )
    context = SimpleNamespace(bot=bot)

    handled = asyncio.run(
        main._handle_zhihu_link_message(
            update=update,
            context=context,
            video_url="https://www.zhihu.com/question/1/answer/2",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert handled is True
    assert len(bot.messages) == 1
    assert bot.messages[0]["disable_web_page_preview"] is True
    assert len(bot.media_groups) == 1
    assert len(bot.media_groups[0]["media"]) == 2
    assert bot.photos == []
    assert bot.documents == []
    assert status.deleted is True
    assert message.deleted is True


def test_send_zhihu_image_media_chunks_albums_and_falls_back_for_large_images():
    class _FakeBot:
        def __init__(self):
            self.photos = []
            self.media_groups = []
            self.documents = []

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_media_group(self, **kwargs):
            self.media_groups.append(kwargs)

        async def send_document(self, **kwargs):
            self.documents.append(kwargs)

    bot = _FakeBot()
    context = SimpleNamespace(bot=bot)
    image_media = [
        {"content": f"image-{index}".encode(), "filename": f"image-{index}.jpg"}
        for index in range(11)
    ]
    image_media.append(
        {
            "content": b"x" * (10 * 1024 * 1024 + 1),
            "filename": "large.jpg",
        }
    )

    asyncio.run(
        main._send_zhihu_image_media(
            context=context,
            chat_id=1,
            image_media=image_media,
        )
    )

    assert len(bot.media_groups) == 1
    assert len(bot.media_groups[0]["media"]) == 10
    assert len(bot.photos) == 1
    assert len(bot.documents) == 1
    assert bot.documents[0]["document"].name == "large.jpg"


def test_handle_twitter_media_message_persists_semantic_tweet_content(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.text = "check this out https://x.com/u/status/1"
            self.message_id = 123
            self.replies = []
            self.deleted = False
            self.reply_to_message = None

        async def reply_text(self, text, **kwargs):
            self.replies.append({"text": text, "kwargs": kwargs})

        async def delete(self):
            self.deleted = True

    class _FakeBot:
        def __init__(self):
            self.photos = []
            self.media_groups = []
            self.videos = []
            self.documents = []
            self.messages = []

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_media_group(self, **kwargs):
            self.media_groups.append(kwargs)

        async def send_video(self, **kwargs):
            self.videos.append(kwargs)

        async def send_document(self, **kwargs):
            self.documents.append(kwargs)

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

    class _FakeStatus:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class _FakeTwitterDownloader:
        def extract_twitter_media(self, url):
            return [
                ("pic", b"img1"),
                ("gif", b"gif1"),
                ("vid", b"vid1"),
            ], {"1": "tweet body with sqlite lock context"}

    captured = {}

    async def fake_add_message(*, chat_id, username, content, **kwargs):
        captured["chat_id"] = chat_id
        captured["username"] = username
        captured["content"] = content
        captured["kwargs"] = kwargs
        return 1

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1),
        effective_user=SimpleNamespace(full_name="Tester", username="tester", id=42),
    )
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())
    monkeypatch.setattr(main, "add_message", fake_add_message)

    handled = asyncio.run(
        main._handle_twitter_media_message(
            update=update,
            context=context,
            video_url="https://x.com/u/status/1",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert handled is True
    assert captured["chat_id"] == 1
    assert captured["username"] == "Tester @tester"
    assert "shared_twitter_link: https://x.com/u/status/1" in captured["content"]
    assert "shared_twitter_media: 1 image(s), 1 video(s), 1 gif(s)" in captured["content"]
    assert "user_comment: check this out" in captured["content"]
    assert "tweet_text: tweet body with sqlite lock context" in captured["content"]
    assert captured["kwargs"]["telegram_user_key"] == "tg_user:42"
    assert captured["kwargs"]["telegram_message_id"] == 123
    assert len(bot.videos) == 1
    assert bot.videos[0]["caption"].startswith("tweet body with sqlite lock context")
    assert status.deleted is True
    assert message.deleted is True


def test_handle_medjpg_reports_render_failure_details(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.message_id = 777
            self.text = "/med2jpg patient A_B"
            self.replies = []

        async def reply_text(self, text, **kwargs):
            self.replies.append(text)
            return SimpleNamespace(delete=lambda: None)

    class _FakeBot:
        async def send_document(self, **kwargs):
            raise AssertionError("send_document should not run after render failure")

    async def fake_generate_med(prompt):
        return {"hospital_name": "H", "patient": {}, "medicines": [{}], "doctor": {}, "watermark": ""}

    async def fake_generate_jpg_from_med_json(json_input, output_jpg, *, raise_on_failure=False):
        raise main.MedRenderError("PDF generation failed. Check xelatex.")

    monkeypatch.setattr(main, "generate_med", fake_generate_med)
    monkeypatch.setattr(main, "generate_jpg_from_med_json", fake_generate_jpg_from_med_json)

    message = _FakeMessage()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=_FakeBot())

    asyncio.run(main.handle_medjpg(update, context))

    assert any("MED image rendering failed." in reply for reply in message.replies)
    assert any("PDF generation failed" in reply for reply in message.replies)


def test_handle_twitter_media_message_offloads_sync_extraction(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.message_id = 124
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class _FakeBot:
        def __init__(self):
            self.messages = []
            self.photos = []
            self.media_groups = []
            self.videos = []
            self.documents = []

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_media_group(self, **kwargs):
            self.media_groups.append(kwargs)

        async def send_video(self, **kwargs):
            self.videos.append(kwargs)

        async def send_document(self, **kwargs):
            self.documents.append(kwargs)

    class _FakeStatus:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class _FakeTwitterDownloader:
        def extract_twitter_media(self, url):
            return [], {"1": "text only tweet"}

    async def fake_add_message(**kwargs):
        return 1

    captured = {"to_thread": False, "callable_name": None}

    async def fake_to_thread(func, *args, **kwargs):
        captured["to_thread"] = True
        captured["callable_name"] = getattr(func, "__name__", None)
        return func(*args, **kwargs)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())
    monkeypatch.setattr(main.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(main, "add_message", fake_add_message)

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    handled = asyncio.run(
        main._handle_twitter_media_message(
            update=update,
            context=context,
            video_url="https://x.com/u/status/1",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert handled is True
    assert captured == {"to_thread": True, "callable_name": "extract_twitter_media"}
    assert bot.messages
    assert status.deleted is True
    assert message.deleted is True


def test_delete_message_if_exists_swallows_cleanup_errors():
    class _BadMessage:
        async def delete(self):
            raise RuntimeError("delete failed")

    asyncio.run(main._delete_message_if_exists(_BadMessage()))


def test_handle_twitter_media_message_handles_text_only_tweet(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.message_id = 456
            self.replies = []
            self.deleted = False

        async def reply_text(self, text, **kwargs):
            self.replies.append({"text": text, "kwargs": kwargs})

        async def delete(self):
            self.deleted = True

    class _FakeBot:
        def __init__(self):
            self.photos = []
            self.media_groups = []
            self.videos = []
            self.documents = []
            self.messages = []

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_media_group(self, **kwargs):
            self.media_groups.append(kwargs)

        async def send_video(self, **kwargs):
            self.videos.append(kwargs)

        async def send_document(self, **kwargs):
            self.documents.append(kwargs)

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

    class _FakeStatus:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class _FakeTwitterDownloader:
        def extract_twitter_media(self, url):
            return [], {"1": "text only tweet"}

    async def fake_add_message(**kwargs):
        return 1

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())
    monkeypatch.setattr(main, "add_message", fake_add_message)

    handled = asyncio.run(
        main._handle_twitter_media_message(
            update=update,
            context=context,
            video_url="https://x.com/u/status/1",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert handled is True
    assert message.replies == []
    assert bot.photos == []
    assert bot.media_groups == []
    assert bot.videos == []
    assert bot.documents == []
    assert bot.messages == [
        {
            "chat_id": 1,
            "text": (
                'text only tweet\n'
                '-- Posted by <a href="https://x.com/u/status/1">@u</a>\n\n'
                'Requested by Tester @tester'
            ),
            "parse_mode": main.ParseMode.HTML,
            "disable_web_page_preview": True,
        }
    ]
    assert status.deleted is True
    assert message.deleted is True


def test_handle_twitter_media_message_sends_video_with_caption(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.message_id = 999
            self.replies = []
            self.deleted = False

        async def reply_text(self, text, **kwargs):
            self.replies.append({"text": text, "kwargs": kwargs})

        async def delete(self):
            self.deleted = True

    class _FakeBot:
        def __init__(self):
            self.photos = []
            self.media_groups = []
            self.videos = []
            self.documents = []
            self.messages = []

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_media_group(self, **kwargs):
            self.media_groups.append(kwargs)

        async def send_video(self, **kwargs):
            self.videos.append(kwargs)

        async def send_document(self, **kwargs):
            self.documents.append(kwargs)

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

    class _FakeStatus:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class _FakeTwitterDownloader:
        def extract_twitter_media(self, url):
            return [("vid", b"video-data")], {"1": "video tweet body"}

    async def fake_add_message(**kwargs):
        return 1

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())
    monkeypatch.setattr(main, "add_message", fake_add_message)

    handled = asyncio.run(
        main._handle_twitter_media_message(
            update=update,
            context=context,
            video_url="https://x.com/u/status/1",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert handled is True
    assert message.replies == []
    assert bot.photos == []
    assert bot.media_groups == []
    assert len(bot.videos) == 1
    assert bot.videos[0]["caption"] == (
        'video tweet body\n'
        '-- Posted by <a href="https://x.com/u/status/1">@u</a>\n\n'
        'Requested by Tester @tester'
    )
    assert bot.videos[0]["parse_mode"] == main.ParseMode.HTML
    assert bot.documents == []
    assert bot.messages == []
    assert status.deleted is True
    assert message.deleted is True


def test_handle_twitter_media_message_sends_video_when_text_empty(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.message_id = 1001
            self.replies = []
            self.deleted = False

        async def reply_text(self, text, **kwargs):
            self.replies.append({"text": text, "kwargs": kwargs})

        async def delete(self):
            self.deleted = True

    class _FakeBot:
        def __init__(self):
            self.photos = []
            self.media_groups = []
            self.videos = []
            self.documents = []
            self.messages = []

        async def send_photo(self, **kwargs):
            self.photos.append(kwargs)

        async def send_media_group(self, **kwargs):
            self.media_groups.append(kwargs)

        async def send_video(self, **kwargs):
            self.videos.append(kwargs)

        async def send_document(self, **kwargs):
            self.documents.append(kwargs)

        async def send_message(self, **kwargs):
            self.messages.append(kwargs)

    class _FakeStatus:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    class _FakeTwitterDownloader:
        def extract_twitter_media(self, url):
            return [("vid", b"video-data")], {"1": ""}

    async def fake_add_message(**kwargs):
        return 1

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())
    monkeypatch.setattr(main, "add_message", fake_add_message)

    handled = asyncio.run(
        main._handle_twitter_media_message(
            update=update,
            context=context,
            video_url="https://x.com/u/status/1",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert handled is True
    assert message.replies == []
    assert bot.photos == []
    assert bot.media_groups == []
    assert len(bot.videos) == 1
    assert bot.videos[0]["caption"] == (
        '-- Posted by <a href="https://x.com/u/status/1">@u</a>\n\n'
        'Requested by Tester @tester'
    )
    assert bot.videos[0]["parse_mode"] == main.ParseMode.HTML
    assert bot.documents == []
    assert bot.messages == []
    assert status.deleted is True
    assert message.deleted is True


def test_handle_text_for_youtube_or_group_sends_detailed_error(monkeypatch):
    class _FakeStatusMessage:
        def __init__(self, text):
            self.text = text
            self.deleted = False
            self.edits = []

        async def edit_text(self, text):
            self.edits.append(text)

        async def delete(self):
            self.deleted = True

    class _FakeMessage:
        def __init__(self, text):
            self.text = text
            self.message_id = 321
            self.deleted = False
            self.reply_calls = []
            self.status_messages = []

        async def reply_text(self, text, **kwargs):
            self.reply_calls.append({"text": text, "kwargs": kwargs})
            status = _FakeStatusMessage(text)
            self.status_messages.append(status)
            return status

        async def delete(self):
            self.deleted = True

    message = _FakeMessage("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1, type="group"),
        effective_user=SimpleNamespace(full_name="Tester", username="tester", id=1),
    )
    tasks = []

    class _FakeApplication:
        def create_task(self, coro):
            task = asyncio.create_task(coro)
            tasks.append(task)
            return task

    context = SimpleNamespace(bot=SimpleNamespace(), application=_FakeApplication())

    async def fake_download_video_to_file(url, output_path):
        raise ValueError("yt-dlp failed: HTTP Error 403: Forbidden")

    monkeypatch.setattr(main, "_extract_video_url", lambda text: "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    monkeypatch.setattr(main, "is_twitter_status_url", lambda url: False)
    monkeypatch.setattr(main, "download_video_to_file", fake_download_video_to_file)

    async def _run():
        await main.handle_text_for_youtube_or_group(update, context)
        assert len(tasks) == 1
        await tasks[0]

    asyncio.run(_run())

    assert len(message.reply_calls) == 2
    assert message.reply_calls[0]["text"] == "Downloading your video, please wait a moment..."
    assert message.reply_calls[1]["text"].startswith("Media link processing failed.\n")
    assert "ValueError: yt-dlp failed: HTTP Error 403: Forbidden" in message.reply_calls[1]["text"]
    assert message.status_messages[0].deleted is True


def test_process_video_link_request_sends_video_with_preview(monkeypatch, tmp_path):
    class _FakeStatusMessage:
        def __init__(self):
            self.deleted = False
            self.edits = []

        async def edit_text(self, text):
            self.edits.append(text)

        async def delete(self):
            self.deleted = True

    class _FakeMessage:
        def __init__(self):
            self.message_id = 999
            self.deleted = False

        async def delete(self):
            self.deleted = True

        async def reply_text(self, text, **kwargs):
            raise AssertionError(f"unexpected error reply: {text}")

    class _FakeBot:
        def __init__(self):
            self.video_calls = []
            self.document_calls = []

        async def send_video(self, **kwargs):
            self.video_calls.append(kwargs)

        async def send_document(self, **kwargs):
            self.document_calls.append(kwargs)

    message = _FakeMessage()
    status = _FakeStatusMessage()
    bot = _FakeBot()
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1),
    )
    context = SimpleNamespace(bot=bot)

    created_path = tmp_path / "video.mp4"
    created_path.write_bytes(b"fake-video")

    async def _fake_download_video_to_file(url, output_path):
        return "video title"

    async def _fake_compress_video_if_needed(_path):
        return str(created_path)

    async def _fake_resolve_caption_url(_url):
        return "https://example.com/original"

    monkeypatch.setattr(main, "is_zhihu_answer_url", lambda url: False)
    monkeypatch.setattr(main, "is_twitter_status_url", lambda url: False)
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main, "download_video_to_file", _fake_download_video_to_file)
    monkeypatch.setattr(main, "compress_video_if_needed", _fake_compress_video_if_needed)
    monkeypatch.setattr(main, "resolve_caption_url", _fake_resolve_caption_url)

    asyncio.run(
        main._process_video_link_request(
            update=update,
            context=context,
            video_url="https://b23.tv/xyz",
            sender_display="Tester @tester",
            status_message=status,
        )
    )

    assert len(bot.video_calls) == 1
    assert bot.document_calls == []
    assert bot.video_calls[0]["supports_streaming"] is True
    assert bot.video_calls[0]["chat_id"] == 1
    assert bot.video_calls[0]["reply_to_message_id"] == 999
    assert status.deleted is True
    assert message.deleted is True


def test_handle_text_for_youtube_or_group_schedules_video_processing_in_background(monkeypatch):
    class _FakeStatusMessage:
        def __init__(self, text):
            self.text = text

        async def edit_text(self, text):
            return None

        async def delete(self):
            return None

    class _FakeMessage:
        def __init__(self, text):
            self.text = text
            self.message_id = 654
            self.reply_calls = []

        async def reply_text(self, text, **kwargs):
            self.reply_calls.append({"text": text, "kwargs": kwargs})
            return _FakeStatusMessage(text)

    scheduled = {}
    message = _FakeMessage("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1, type="group"),
        effective_user=SimpleNamespace(full_name="Tester", username="tester", id=1),
    )
    context = SimpleNamespace(bot=SimpleNamespace())

    def fake_schedule_background_task(context_arg, coro):
        scheduled["context"] = context_arg
        scheduled["coro"] = coro
        coro.close()

    monkeypatch.setattr(main, "_extract_video_url", lambda text: "https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    monkeypatch.setattr(main, "_schedule_background_task", fake_schedule_background_task)

    asyncio.run(main.handle_text_for_youtube_or_group(update, context))

    assert len(message.reply_calls) == 1
    assert message.reply_calls[0]["text"] == "Downloading your video, please wait a moment..."
    assert scheduled["context"] is context
    assert scheduled["coro"] is not None


def test_handle_text_for_youtube_or_group_schedules_every_supported_link(monkeypatch):
    class _FakeStatusMessage:
        async def delete(self):
            return None

    class _FakeMessage:
        def __init__(self):
            self.text = (
                "first https://www.youtube.com/watch?v=dQw4w9WgXcQ and "
                "second x.com/user/status/123"
            )
            self.message_id = 656
            self.reply_calls = []

        async def reply_text(self, text, **kwargs):
            self.reply_calls.append(text)
            return _FakeStatusMessage()

    scheduled = {}
    message = _FakeMessage()
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1, type="group"),
        effective_user=SimpleNamespace(full_name="Tester", username="tester", id=1),
    )
    context = SimpleNamespace(bot=SimpleNamespace())

    def fake_process_video_link_batch(**kwargs):
        scheduled.update(kwargs)
        async def no_op():
            return None

        return no_op()

    def fake_schedule_background_task(context_arg, coro):
        assert context_arg is context
        coro.close()

    monkeypatch.setattr(main, "_process_video_link_batch", fake_process_video_link_batch)
    monkeypatch.setattr(main, "_schedule_background_task", fake_schedule_background_task)

    asyncio.run(main.handle_text_for_youtube_or_group(update, context))

    assert scheduled["video_urls"] == [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://x.com/user/status/123",
    ]
    assert scheduled["delete_source_message"] is True
    assert len(scheduled["status_messages"]) == 2
    assert len(message.reply_calls) == 2


def test_handle_text_for_youtube_or_group_limits_links_per_message(monkeypatch):
    class _FakeStatusMessage:
        async def delete(self):
            return None

    class _FakeMessage:
        text = "https://x.com/user/status/123 https://x.com/user/status/456"
        message_id = 657

        def __init__(self):
            self.replies = []

        async def reply_text(self, text, **kwargs):
            self.replies.append(text)
            return _FakeStatusMessage()

    scheduled = {}
    message = _FakeMessage()
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1, type="group"),
        effective_user=SimpleNamespace(full_name="Tester", username="tester", id=1),
    )

    def fake_process_video_link_batch(**kwargs):
        scheduled.update(kwargs)

        async def no_op():
            return None

        return no_op()

    def fake_schedule_background_task(_context, coro):
        coro.close()

    monkeypatch.setattr(main, "MAX_MEDIA_LINKS_PER_MESSAGE", 1)
    monkeypatch.setattr(main, "_process_video_link_batch", fake_process_video_link_batch)
    monkeypatch.setattr(main, "_schedule_background_task", fake_schedule_background_task)

    asyncio.run(main.handle_text_for_youtube_or_group(update, SimpleNamespace(bot=SimpleNamespace())))

    assert scheduled["video_urls"] == ["https://x.com/user/status/123"]
    assert message.replies[0] == "Processing the first 1 of 2 supported links."


def test_document_renderer_uses_generated_safe_download_path(monkeypatch, tmp_path):
    class _FakeDownloadedFile:
        def __init__(self):
            self.path = None

        async def download_to_drive(self, custom_path):
            self.path = custom_path
            with open(custom_path, "w", encoding="utf-8") as output:
                output.write("# title")
            return custom_path

    class _FakeDocument:
        file_name = "../../runtime.env.md"
        file_size = 16

        def __init__(self, file):
            self.file = file

        async def get_file(self):
            return self.file

    class _FakeStatus:
        async def edit_text(self, _text):
            return None

        async def delete(self):
            return None

    class _FakeMessage:
        message_id = 123

        def __init__(self, document):
            self.document = document
            self.replies = []

        async def reply_text(self, text, **kwargs):
            self.replies.append(text)
            return _FakeStatus()

    file = _FakeDownloadedFile()
    message = _FakeMessage(_FakeDocument(file))
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))

    monkeypatch.setattr(
        main,
        "_build_output_path",
        lambda prefix, _message_id, extension="jpg": str(tmp_path / f"{prefix}.{extension}"),
    )

    async def fake_render(*_args, **_kwargs):
        return None

    monkeypatch.setattr(main, "_render_and_send_image_from_markdown", fake_render)

    asyncio.run(main.handle_text_or_markdown_document(update, SimpleNamespace()))

    assert file.path == str(tmp_path / "source.md")
    assert not (tmp_path / "source.md").exists()


def test_process_video_link_batch_deletes_source_after_all_links_finish(monkeypatch):
    events = []

    class _FakeMessage:
        async def delete(self):
            events.append("source-deleted")

    async def fake_process_video_link_request(**kwargs):
        events.append(f"start:{kwargs['video_url']}")
        await asyncio.sleep(0)
        events.append(f"done:{kwargs['video_url']}")
        return True

    monkeypatch.setattr(main, "_process_video_link_request", fake_process_video_link_request)
    update = SimpleNamespace(message=_FakeMessage())

    asyncio.run(
        main._process_video_link_batch(
            update=update,
            context=SimpleNamespace(),
            video_urls=["youtube", "twitter"],
            sender_display="Tester",
            status_messages=[SimpleNamespace(), SimpleNamespace()],
        )
    )

    assert events == [
        "start:youtube",
        "start:twitter",
        "done:youtube",
        "done:twitter",
        "source-deleted",
    ]


def test_process_video_link_batch_keeps_source_when_a_link_fails(monkeypatch):
    class _FakeMessage:
        def __init__(self):
            self.deleted = False

        async def delete(self):
            self.deleted = True

    async def fake_process_video_link_request(**kwargs):
        return kwargs["video_url"] != "failed"

    monkeypatch.setattr(main, "_process_video_link_request", fake_process_video_link_request)
    message = _FakeMessage()

    asyncio.run(
        main._process_video_link_batch(
            update=SimpleNamespace(message=message),
            context=SimpleNamespace(),
            video_urls=["ok", "failed"],
            sender_display="Tester",
            status_messages=[SimpleNamespace(), SimpleNamespace()],
        )
    )

    assert message.deleted is False


def test_handle_text_for_youtube_or_group_uses_zhihu_status_text(monkeypatch):
    class _FakeStatusMessage:
        def __init__(self, text):
            self.text = text

        async def edit_text(self, text):
            return None

        async def delete(self):
            return None

    class _FakeMessage:
        def __init__(self, text):
            self.text = text
            self.message_id = 655
            self.reply_calls = []

        async def reply_text(self, text, **kwargs):
            self.reply_calls.append({"text": text, "kwargs": kwargs})
            return _FakeStatusMessage(text)

    scheduled = {}
    message = _FakeMessage("https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097")
    update = SimpleNamespace(
        message=message,
        effective_chat=SimpleNamespace(id=1, type="group"),
        effective_user=SimpleNamespace(full_name="Tester", username="tester", id=1),
    )
    context = SimpleNamespace(bot=SimpleNamespace())

    def fake_schedule_background_task(context_arg, coro):
        scheduled["context"] = context_arg
        scheduled["coro"] = coro
        coro.close()

    monkeypatch.setattr(
        main,
        "_extract_video_url",
        lambda text: "https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097",
    )
    monkeypatch.setattr(main, "_schedule_background_task", fake_schedule_background_task)

    asyncio.run(main.handle_text_for_youtube_or_group(update, context))

    assert len(message.reply_calls) == 1
    assert message.reply_calls[0]["text"] == "Parsing your Zhihu link, please wait a moment..."
    assert scheduled["context"] is context
    assert scheduled["coro"] is not None


def test_schedule_background_task_tracks_fallback_tasks():
    async def quick_task():
        return None

    async def _run() -> None:
        main._BACKGROUND_TASKS.clear()
        main._schedule_background_task(SimpleNamespace(), quick_task())
        assert len(main._BACKGROUND_TASKS) == 1
        await asyncio.gather(*list(main._BACKGROUND_TASKS))
        await asyncio.sleep(0)
        assert main._BACKGROUND_TASKS == set()

    asyncio.run(_run())
