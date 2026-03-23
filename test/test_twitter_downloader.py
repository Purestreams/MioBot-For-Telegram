import unittest
from unittest import mock

from app.twitter_downloader import P_TWT_LINK, TwitterDownloader, is_twitter_status_url


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

    def test_handle_url_extracts_tweet_id_after_html_unescape(self):
        with mock.patch.object(self.downloader, "get_single_tweet_data", return_value={"ok": True}) as mocked:
            result = self.downloader.handle_url(
                "https://x.com/minokakay/status/2035538892175405329?s=46&amp;t=6C5C8msOW1klCHHbUUlASA"
            )
            self.assertEqual(result, {"ok": True})
            mocked.assert_called_once_with("2035538892175405329")

    def test_parse_tweet_content_selects_highest_resolution_video(self):
        tw_content = {
            "itemContent": {
                "tweet_results": {
                    "result": {
                        "legacy": {"full_text": "hello world"},
                        "video_urls": [
                            "https://video.twimg.com/ext_tw_video/111/pu/vid/avc1/640x360/low.mp4",
                            "https://video.twimg.com/ext_tw_video/111/pu/vid/avc1/1280x720/high.mp4",
                        ],
                    }
                }
            }
        }
        pic, gif, vid, text = self.downloader._parse_tweet_content(tw_content, "111")
        self.assertEqual(pic, {})
        self.assertEqual(gif, {})
        self.assertIn("high.mp4", vid)
        self.assertNotIn("low.mp4", vid)
        self.assertEqual(text.get("111"), "hello world")

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


if __name__ == "__main__":
    unittest.main()
