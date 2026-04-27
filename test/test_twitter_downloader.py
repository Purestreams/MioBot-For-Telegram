import unittest
from unittest import mock

from yt_dlp.utils import DownloadError

from app.twitter_downloader import (
    P_TWT_LINK,
    TwitterDownloader,
    build_twitter_caption,
    format_tweet_text_for_reply,
    is_twitter_status_url,
)


class TwitterDownloaderRegexTests(unittest.TestCase):
    def test_detects_twitter_status_url(self):
        self.assertTrue(is_twitter_status_url("https://x.com/minokakay/status/2035538892175405329?s=46&t=abc"))
        self.assertTrue(is_twitter_status_url("https://twitter.com/user/status/1234567890"))

    def test_detects_html_escaped_twitter_status_url(self):
        escaped = "https://x.com/minokakay/status/2035538892175405329?s=46&amp;t=6C5C8msOW1klCHHbUUlASA"
        self.assertTrue(is_twitter_status_url(escaped))
        match = P_TWT_LINK.findall(escaped.replace("&amp;", "&"))
        self.assertEqual(match[0][1], "2035538892175405329")


class TwitterDownloaderComponentTests(unittest.TestCase):
    def setUp(self):
        self.downloader = TwitterDownloader()

    def test_extract_csrf_token(self):
        token = TwitterDownloader._extract_csrf_token("foo=bar; ct0=mycsrf; baz=1")
        self.assertEqual(token, "mycsrf")
        self.assertIsNone(TwitterDownloader._extract_csrf_token(""))

    def test_strip_html_text_converts_blockquote_html(self):
        html_block = "<blockquote><p>Line1<br>Line2</p>&mdash; user</blockquote>"
        text = TwitterDownloader._strip_html_text(html_block)
        self.assertIn("Line1", text)
        self.assertIn("Line2", text)

    def test_handle_url_extracts_tweet_id_after_html_unescape(self):
        with mock.patch.object(self.downloader, "get_single_tweet_data", return_value={"ok": True}) as mocked:
            result = self.downloader.handle_url(
                "https://x.com/minokakay/status/2035538892175405329?s=46&amp;t=6C5C8msOW1klCHHbUUlASA"
            )
            self.assertEqual(result, {"ok": True})
            mocked.assert_called_once_with(
                "2035538892175405329",
                "https://x.com/minokakay/status/2035538892175405329?s=46&t=6C5C8msOW1klCHHbUUlASA",
            )

    def test_parse_ydlp_info_selects_highest_resolution_video(self):
        info = {
            "id": "111",
            "description": "hello world",
            "formats": [
                {
                    "url": "https://video.twimg.com/ext_tw_video/111/pu/vid/640x360/low.mp4",
                    "ext": "mp4",
                    "vcodec": "h264",
                    "acodec": "aac",
                    "width": 640,
                    "height": 360,
                },
                {
                    "url": "https://video.twimg.com/ext_tw_video/111/pu/vid/1280x720/high.mp4",
                    "ext": "mp4",
                    "vcodec": "h264",
                    "acodec": "aac",
                    "width": 1280,
                    "height": 720,
                },
            ],
            "thumbnails": [
                {"url": "https://pbs.twimg.com/media/abc123.jpg?name=small"},
            ],
        }
        pic, gif, vid, text = self.downloader._parse_ydlp_info(info, "111")
        self.assertIn("abc123.jpg", pic)
        self.assertEqual(gif, {})
        self.assertIn("high.mp4", vid)
        self.assertNotIn("low.mp4", vid)
        self.assertEqual(text.get("111"), "hello world")

    def test_parse_ydlp_info_prefers_direct_mp4_over_hls_manifest(self):
        info = {
            "id": "111",
            "description": "hello world",
            "formats": [
                {
                    "format_id": "http-2176",
                    "url": "https://video.twimg.com/ext_tw_video/111/pu/vid/1280x720/high.mp4",
                    "ext": "mp4",
                    "protocol": "https",
                    "width": 1280,
                    "height": 720,
                },
                {
                    "format_id": "hls-6017",
                    "url": "https://video.twimg.com/ext_tw_video/111/pu/pl/1280x720/high.m3u8",
                    "ext": "mp4",
                    "protocol": "m3u8_native",
                    "vcodec": "avc1.640033",
                    "acodec": "none",
                    "width": 1280,
                    "height": 720,
                },
                {
                    "format_id": "hls-audio-128000-Audio",
                    "url": "https://video.twimg.com/ext_tw_video/111/pu/pl/audio.m3u8",
                    "ext": "mp4",
                    "protocol": "m3u8_native",
                    "vcodec": "none",
                    "width": None,
                    "height": None,
                },
            ],
        }

        pic, gif, vid, text = self.downloader._parse_ydlp_info(info, "111")
        self.assertEqual(pic, {})
        self.assertEqual(gif, {})
        self.assertIn("high.mp4", vid)
        self.assertEqual(vid["high.mp4"]["url"], "https://video.twimg.com/ext_tw_video/111/pu/vid/1280x720/high.mp4")
        self.assertEqual(text.get("111"), "hello world")

    def test_get_single_tweet_data_uses_yt_dlp(self):
        fake_info = {
            "id": "2035538892175405329",
            "description": "tweet text",
            "formats": [
                {
                    "url": "https://video.twimg.com/ext_tw_video/2035/pu/vid/1280x720/high.mp4",
                    "ext": "mp4",
                    "vcodec": "h264",
                    "acodec": "aac",
                    "width": 1280,
                    "height": 720,
                }
            ],
        }

        with mock.patch("app.twitter_downloader.yt_dlp.YoutubeDL") as ydl_cls:
            ydl = ydl_cls.return_value.__enter__.return_value
            ydl.extract_info.return_value = fake_info

            data = self.downloader.get_single_tweet_data("2035538892175405329")

        self.assertIsNotNone(data)
        self.assertIn("vidList", data)
        self.assertEqual(list(data["textList"].values())[0], "tweet text")
        ydl.extract_info.assert_called_once_with("https://x.com/i/status/2035538892175405329", download=False)

    def test_get_single_tweet_data_returns_none_on_yt_dlp_error(self):
        with mock.patch("app.twitter_downloader.yt_dlp.YoutubeDL") as ydl_cls:
            ydl = ydl_cls.return_value.__enter__.return_value
            ydl.extract_info.side_effect = Exception("boom")
            self.assertIsNone(self.downloader.get_single_tweet_data("123"))

    def test_get_single_tweet_data_uses_syndication_fallback_for_no_video(self):
        with (
            mock.patch("app.twitter_downloader.yt_dlp.YoutubeDL") as ydl_cls,
            mock.patch.object(
                self.downloader,
                "_fetch_syndication_fallback",
                return_value={
                    "picList": {"img.jpg": {"url": "https://pbs.twimg.com/media/img.jpg", "twtId": "123"}},
                    "gifList": {},
                    "vidList": {},
                    "textList": {"123": "tweet text"},
                },
            ) as fallback,
        ):
            ydl = ydl_cls.return_value.__enter__.return_value
            ydl.extract_info.side_effect = DownloadError("ERROR: [twitter] 123: No video could be found in this tweet")

            data = self.downloader.get_single_tweet_data("123")

        self.assertIsNotNone(data)
        assert data is not None
        self.assertIn("img.jpg", data["picList"])
        self.assertEqual(data["textList"].get("123"), "tweet text")
        fallback.assert_called_once_with("123")

    def test_fetch_syndication_fallback_supports_full_text_and_media_details(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "full_text": "hello from fallback",
            "mediaDetails": [
                {"type": "photo", "media_url_https": "https://pbs.twimg.com/media/fallback.jpg"}
            ],
        }

        with mock.patch.object(self.downloader.session, "get", return_value=response):
            data = self.downloader._fetch_syndication_fallback("123")

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["textList"].get("123"), "hello from fallback")
        self.assertIn("fallback.jpg", data["picList"])

    def test_get_single_tweet_data_uses_oembed_when_syndication_is_empty(self):
        with (
            mock.patch("app.twitter_downloader.yt_dlp.YoutubeDL") as ydl_cls,
            mock.patch.object(self.downloader, "_fetch_syndication_fallback", return_value={
                "picList": {},
                "gifList": {},
                "vidList": {},
                "textList": {},
            }) as syndication,
            mock.patch.object(
                self.downloader,
                "_fetch_oembed_text_fallback",
                return_value={
                    "picList": {},
                    "gifList": {},
                    "vidList": {},
                    "textList": {"123": "oembed text"},
                },
            ) as oembed,
            mock.patch.object(self.downloader, "_fetch_fxtwitter_fallback", return_value=None) as fxt,
        ):
            ydl = ydl_cls.return_value.__enter__.return_value
            ydl.extract_info.side_effect = DownloadError("ERROR: [twitter] 123: No video could be found in this tweet")

            data = self.downloader.get_single_tweet_data("123")

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["textList"].get("123"), "oembed text")
        syndication.assert_called_once_with("123")
        oembed.assert_called_once()
        fxt.assert_called_once_with("123")

    def test_fetch_oembed_text_fallback_extracts_pbs_image_from_html(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "html": '<blockquote><p>hello</p><img src="https://pbs.twimg.com/media/abc123.jpg?format=jpg&amp;name=small"></blockquote>'
        }

        with mock.patch.object(self.downloader.session, "get", return_value=response):
            data = self.downloader._fetch_oembed_text_fallback("https://x.com/i/status/123", "123")

        self.assertIsNotNone(data)
        assert data is not None
        self.assertIn("abc123.jpg", data["picList"])

    def test_resolve_pic_twitter_to_image_url_supports_content_before_property(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.url = "https://x.com/some/redirect"
        response.text = (
            '<html><head><meta content="https://pbs.twimg.com/media/xyz456.jpg?format=jpg&name=small" '
            'property="og:image"></head></html>'
        )

        with mock.patch.object(self.downloader.session, "get", return_value=response):
            image_url = self.downloader._resolve_pic_twitter_to_image_url("pic.twitter.com/AbCdEf12")

        self.assertEqual(image_url, "https://pbs.twimg.com/media/xyz456.jpg?format=jpg&name=small")

    def test_get_single_tweet_data_uses_fallback_on_internal_server_error(self):
        with (
            mock.patch("app.twitter_downloader.yt_dlp.YoutubeDL") as ydl_cls,
            mock.patch.object(
                self.downloader,
                "_fetch_syndication_fallback",
                return_value={
                    "picList": {"img.jpg": {"url": "https://pbs.twimg.com/media/img.jpg", "twtId": "123"}},
                    "gifList": {},
                    "vidList": {},
                    "textList": {"123": "tweet text"},
                },
            ) as syndication,
            mock.patch.object(self.downloader, "_fetch_oembed_text_fallback") as oembed,
        ):
            ydl = ydl_cls.return_value.__enter__.return_value
            ydl.extract_info.side_effect = DownloadError(
                "ERROR: [twitter] 123: Error(s) while querying API: Internal server error"
            )

            data = self.downloader.get_single_tweet_data("123")

        self.assertIsNotNone(data)
        assert data is not None
        self.assertIn("img.jpg", data["picList"])
        self.assertEqual(data["textList"].get("123"), "tweet text")
        syndication.assert_called_once_with("123")
        oembed.assert_not_called()

    def test_get_single_tweet_data_prefers_fxtwitter_when_oembed_has_no_media(self):
        with (
            mock.patch("app.twitter_downloader.yt_dlp.YoutubeDL") as ydl_cls,
            mock.patch.object(self.downloader, "_fetch_syndication_fallback", return_value={
                "picList": {},
                "gifList": {},
                "vidList": {},
                "textList": {},
            }) as syndication,
            mock.patch.object(
                self.downloader,
                "_fetch_oembed_text_fallback",
                return_value={
                    "picList": {},
                    "gifList": {},
                    "vidList": {},
                    "textList": {"123": "oembed text"},
                },
            ) as oembed,
            mock.patch.object(
                self.downloader,
                "_fetch_fxtwitter_fallback",
                return_value={
                    "picList": {"img.jpg": {"url": "https://pbs.twimg.com/media/img.jpg", "twtId": "123"}},
                    "gifList": {},
                    "vidList": {},
                    "textList": {"123": "fx text"},
                },
            ) as fxt,
            mock.patch.object(self.downloader, "_fetch_vxtwitter_fallback") as vxt,
        ):
            ydl = ydl_cls.return_value.__enter__.return_value
            ydl.extract_info.side_effect = DownloadError("ERROR: [twitter] 123: Error(s) while querying API: Internal server error")

            data = self.downloader.get_single_tweet_data("123")

        self.assertIsNotNone(data)
        assert data is not None
        self.assertIn("img.jpg", data["picList"])
        syndication.assert_called_once_with("123")
        oembed.assert_called_once()
        fxt.assert_called_once_with("123")
        vxt.assert_not_called()

    def test_fetch_oembed_text_fallback_normalizes_to_status_url(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "html": '<blockquote><p>hello</p></blockquote>'
        }

        with mock.patch.object(self.downloader.session, "get", return_value=response) as getter:
            data = self.downloader._fetch_oembed_text_fallback(
                "https://x.com/NASA/status/2048166203735040057?s=46&t=abc",
                "2048166203735040057",
            )

        self.assertIsNotNone(data)
        called_url = getter.call_args.args[0]
        self.assertIn("twitter.com%2FNASA%2Fstatus%2F2048166203735040057", called_url)

    def test_fetch_vxtwitter_fallback_parses_image_media(self):
        response = mock.Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "text": "tweet text",
            "mediaURLs": ["https://pbs.twimg.com/media/sample.jpg"],
            "media_extended": [
                {"type": "image", "url": "https://pbs.twimg.com/media/sample2.jpg"}
            ],
        }

        with mock.patch.object(self.downloader.session, "get", return_value=response):
            data = self.downloader._fetch_vxtwitter_fallback("123")

        self.assertIsNotNone(data)
        assert data is not None
        self.assertIn("sample.jpg", data["picList"])
        self.assertIn("sample2.jpg", data["picList"])
        self.assertEqual(data["textList"].get("123"), "tweet text")

    def test_extract_twitter_media_returns_empty_when_unhandled_url(self):
        with mock.patch.object(self.downloader, "_activate_guest_token") as activate:
            media, text = self.downloader.extract_twitter_media("https://example.com/not-twitter")
            activate.assert_called_once()
            self.assertEqual(media, [])
            self.assertEqual(text, {})

    def test_extract_twitter_media_downloads_each_media_type(self):
        mocked_data = {
            "picList": {"a.jpg": {"url": "https://example.com/a.jpg"}},
            "gifList": {"b.mp4": {"url": "https://example.com/b.mp4"}},
            "vidList": {"c.mp4": {"url": "https://example.com/c.mp4"}},
            "textList": {"123": "tweet text"},
        }
        with (
            mock.patch.object(self.downloader, "_activate_guest_token"),
            mock.patch.object(self.downloader, "handle_url", return_value=mocked_data),
            mock.patch.object(self.downloader, "download_media_bytes", side_effect=[b"p", b"g", b"v"]),
        ):
            media, text = self.downloader.extract_twitter_media("https://x.com/user/status/123")
        self.assertEqual(media, [("pic", b"p"), ("gif", b"g"), ("vid", b"v")])
        self.assertEqual(text, {"123": "tweet text"})

    def test_extract_twitter_media_resolves_pic_twitter_short_link_to_image(self):
        mocked_data = {
            "picList": {},
            "gifList": {},
            "vidList": {},
            "textList": {"123": "look pic.twitter.com/AbCdEf12"},
        }
        with (
            mock.patch.object(self.downloader, "_activate_guest_token"),
            mock.patch.object(self.downloader, "handle_url", return_value=mocked_data),
            mock.patch.object(
                self.downloader,
                "_resolve_pic_twitter_to_image_url",
                return_value="https://pbs.twimg.com/media/resolved.jpg",
            ) as resolver,
            mock.patch.object(self.downloader, "download_media_bytes", return_value=b"img-bytes") as downloader,
        ):
            media, text = self.downloader.extract_twitter_media("https://x.com/user/status/123")

        self.assertEqual(media, [("pic", b"img-bytes")])
        self.assertEqual(text, {"123": "look pic.twitter.com/AbCdEf12"})
        resolver.assert_called_once_with("pic.twitter.com/AbCdEf12")
        downloader.assert_called_once_with("https://pbs.twimg.com/media/resolved.jpg")


class TwitterDownloaderFormattingTests(unittest.TestCase):
    def setUp(self):
        self.downloader = TwitterDownloader()

    def test_format_tweet_text_for_reply_removes_pic_twitter_short_link(self):
        formatted = format_tweet_text_for_reply(
            "自相矛盾这一块 pic.twitter.com/Ly9lXqeti0 — 沫柠 (@Moningmeng) March 25, 2026",
            "https://x.com/moningmeng/status/2036721497700434404",
        )
        self.assertIn("自相矛盾这一块", formatted)
        self.assertNotIn("pic.twitter.com", formatted)
        self.assertNotIn("Moningmeng", formatted)

    def test_build_twitter_caption_matches_required_template(self):
        caption = build_twitter_caption(
            "tweet body",
            "Tester @tester",
            "https://x.com/Moningmeng/status/2036721497700434404",
        )
        self.assertEqual(
            caption,
            'tweet body\n'
            '-- Posted by <a href="https://x.com/Moningmeng/status/2036721497700434404">@Moningmeng</a>\n\n'
            'Requested by Tester @tester',
        )

    def test_extract_twitter_media_logs_diagnostics_when_fallback_is_empty(self):
        mocked_data = {
            "picList": {},
            "gifList": {},
            "vidList": {},
            "textList": {},
        }
        with (
            mock.patch.object(self.downloader, "_activate_guest_token"),
            mock.patch.object(self.downloader, "handle_url", return_value=mocked_data),
            self.assertLogs("app.twitter_downloader", level="ERROR") as logs,
        ):
            media, text = self.downloader.extract_twitter_media(
                "https://x.com/himself65/status/2036933945200406781?s=46&t=6C5C8msOW1klCHHbUUlASA"
            )

        self.assertEqual(media, [])
        self.assertEqual(text, {})
        joined = "\n".join(logs.output)
        self.assertIn("Twitter extraction produced empty result", joined)
        self.assertIn("payload_keys=['gifList', 'picList', 'textList', 'vidList']", joined)
        self.assertIn("pic=0 gif=0 vid=0 text=0", joined)

    def test_build_ydl_opts_uses_cookiefile_when_present(self):
        downloader = TwitterDownloader(cookie_file="config/x.com_cookies.txt")
        with mock.patch("app.twitter_downloader.os.path.isfile", return_value=True):
            opts = downloader._build_ydl_opts()
        self.assertEqual(opts.get("cookiefile"), "config/x.com_cookies.txt")

    def test_build_ydl_opts_falls_back_to_cookie_header(self):
        downloader = TwitterDownloader(cookie="auth_token=a; ct0=b", cookie_file="missing.txt")
        with mock.patch("app.twitter_downloader.os.path.isfile", return_value=False):
            opts = downloader._build_ydl_opts()
        headers = opts.get("http_headers", {})
        self.assertIn("Cookie", headers)
        self.assertEqual(headers.get("x-csrf-token"), "b")


if __name__ == "__main__":
    unittest.main()
