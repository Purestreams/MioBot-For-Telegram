import asyncio
from types import SimpleNamespace

import main
from app.twitter_downloader import format_tweet_text_for_reply, summarize_tweet_text


def test_extract_video_url_prefers_youtube_when_present():
    message = "check this https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    extracted = main._extract_video_url(message)
    assert extracted is not None
    assert "youtube.com" in extracted


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


def test_classify_group_reply_trigger_detects_username_mention():
    trigger = main._classify_group_reply_trigger("hey @MioooooooooBot look here", "MioooooooooBot")
    assert trigger == "username_mention"


def test_classify_group_reply_trigger_detects_alias_mention():
    trigger = main._classify_group_reply_trigger("mioo look here", "MioooooooooBot")
    assert trigger == "alias_mention"


def test_classify_group_reply_trigger_ignores_embedded_alias_text():
    trigger = main._classify_group_reply_trigger("amiooops should stay ambient", "MioooooooooBot")
    assert trigger == "ambient"


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

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())

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

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())

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

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())

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

    message = _FakeMessage()
    bot = _FakeBot()
    status = _FakeStatus()
    update = SimpleNamespace(message=message, effective_chat=SimpleNamespace(id=1))
    context = SimpleNamespace(bot=bot)

    monkeypatch.setattr(main, "TwitterDownloader", lambda: _FakeTwitterDownloader())

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
