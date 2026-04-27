import html
import logging
import os
import re
import urllib.parse
from typing import Any, Dict, List, Mapping, Optional, Tuple, cast

import requests
import yt_dlp
from yt_dlp.utils import DownloadError
from app.runtime_config import get_runtime_value

logger = logging.getLogger(__name__)


P_TWT_LINK = re.compile(r'https://(?:x|twitter)\.com/(.+?)/status/(\d+)')
P_CSRF_TOKEN = re.compile(r'ct0=(.+?)(?:;|$)')
P_PIC_TWITTER_LINK = re.compile(r'(?:https?://)?pic\.twitter\.com/[A-Za-z0-9]+')
P_META_IMAGE = re.compile(
    r'<meta\b(?=[^>]*(?:property|name)=["\'](?:og:image|twitter:image)["\'])(?=[^>]*content=["\']([^"\']+)["\'])[^>]*>',
    re.IGNORECASE,
)
P_PBS_IMAGE_LINK = re.compile(r'https?://pbs\.twimg\.com/[^"\'\s<]+', re.IGNORECASE)

DEFAULT_HTTP_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/135.0.0.0 Safari/537.36'
    ),
    'Accept-Language': 'en-US,en;q=0.9',
}


def is_twitter_status_url(url: str) -> bool:
    return bool(P_TWT_LINK.search(html.unescape(url or '')))


def summarize_tweet_text(text_dict: Mapping[str, str]) -> str:
    tweet_text = next(iter(text_dict.values()), "")
    return tweet_text.strip()


def format_tweet_text_for_reply(tweet_text: str, original_url: str) -> str:
    text = (tweet_text or "").strip()
    text = P_PIC_TWITTER_LINK.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    # oEmbed text often appends author/date after " — "; keep only tweet content body.
    if " — " in text:
        text = text.split(" — ", 1)[0].strip()

    return html.escape(text)


def _extract_twitter_username_from_url(original_url: str) -> str:
    decoded_url = html.unescape(original_url or "")
    match = P_TWT_LINK.search(decoded_url)
    if not match:
        return "twitter_user"
    return match.group(1)


def build_twitter_caption(tweet_caption_html: str, sender_display: str, original_url: str) -> str:
    safe_url = html.escape(original_url, quote=True)
    twitter_username = html.escape(_extract_twitter_username_from_url(original_url))
    posted_by_line = f'-- Posted by <a href="{safe_url}">@{twitter_username}</a>'
    sender_line = f"Requested by {html.escape(sender_display)}"

    if tweet_caption_html:
        return f"{tweet_caption_html}\n{posted_by_line}\n\n{sender_line}"
    return f"{posted_by_line}\n\n{sender_line}"


class TwitterDownloader:
    def __init__(
        self,
        cookie: Optional[str] = None,
        proxies: Optional[dict] = None,
        cookie_file: Optional[str] = None,
    ):
        self.cookie = cookie or get_runtime_value('TWITTER_COOKIE')
        self.proxies = proxies or {}
        default_cookie_file = os.path.join('config', 'x.com_cookies.txt')
        self.cookie_file = cookie_file or get_runtime_value('TWITTER_COOKIE_FILE') or default_cookie_file

        self.session = requests.Session()
        if self.proxies:
            self.session.proxies.update(self.proxies)

    def _build_ydl_opts(self) -> Dict[str, Any]:
        opts: Dict[str, Any] = {
            'quiet': True,
            'no_warnings': True,
            'skip_download': True,
            'noplaylist': True,
            'extract_flat': False,
            'retries': 3,
            'fragment_retries': 3,
            'socket_timeout': 30,
        }

        headers: Dict[str, str] = dict(DEFAULT_HTTP_HEADERS)
        if self.cookie_file and os.path.isfile(self.cookie_file):
            # Prefer cookie file format exported from browser for stable auth.
            opts['cookiefile'] = self.cookie_file
        elif self.cookie:
            headers['Cookie'] = self.cookie
            csrf_token = self._extract_csrf_token(self.cookie)
            if csrf_token:
                headers['x-csrf-token'] = csrf_token
        opts['http_headers'] = headers

        if self.proxies:
            proxy = next((p for p in self.proxies.values() if p), None)
            if proxy:
                opts['proxy'] = proxy

        return opts

    @staticmethod
    def _extract_csrf_token(cookie_str: str) -> Optional[str]:
        if not cookie_str:
            return None
        match = P_CSRF_TOKEN.findall(cookie_str)
        return match[0] if match else None

    def _activate_guest_token(self) -> None:
        # yt-dlp handles Twitter/X extraction without manual guest-token bootstrapping.
        return

    @staticmethod
    def _extract_filename(url: str, fallback: str) -> str:
        file_name = url.split('?', 1)[0].rsplit('/', 1)[-1]
        return file_name or fallback

    @staticmethod
    def _looks_like_hls_manifest(url: str, protocol: str) -> bool:
        lowered_url = str(url or '').lower()
        lowered_protocol = str(protocol or '').lower()
        return lowered_url.endswith('.m3u8') or 'm3u8' in lowered_protocol

    @classmethod
    def _select_best_direct_video_format(cls, formats: List[Mapping[str, Any]]) -> Optional[Dict[str, str]]:
        best_video: Optional[Dict[str, str]] = None
        best_score: Tuple[int, int, int, int] = (-1, -1, -1, -1)

        for fmt in formats:
            if not isinstance(fmt, dict):
                continue

            fmt_url = str(fmt.get('url') or '').strip()
            if not fmt_url:
                continue

            protocol = str(fmt.get('protocol') or '')
            ext = str(fmt.get('ext') or '')
            width = int(fmt.get('width') or 0)
            height = int(fmt.get('height') or 0)
            format_id = str(fmt.get('format_id') or '')
            vcodec = fmt.get('vcodec')
            is_audio_only = (
                vcodec == 'none'
                or (
                    vcodec is None
                    and width == 0
                    and height == 0
                    and 'audio' in format_id.lower()
                )
            )
            if is_audio_only:
                continue

            is_direct_http = protocol in {'http', 'https'}
            is_hls = cls._looks_like_hls_manifest(fmt_url, protocol)
            is_mp4 = ext == 'mp4' or fmt_url.lower().split('?', 1)[0].endswith('.mp4')
            bitrate = int(fmt.get('tbr') or fmt.get('abr') or 0)
            resolution = width * height
            score = (1 if is_direct_http else 0, 1 if is_mp4 else 0, 0 if is_hls else 1, resolution + bitrate)
            if score <= best_score:
                continue

            best_score = score
            best_video = {
                'url': fmt_url,
                'file_name': cls._extract_filename(fmt_url, 'video.mp4'),
                'acodec': str(fmt.get('acodec') or ''),
            }

        return best_video

    @staticmethod
    def _has_audio_only_formats(formats: List[Mapping[str, Any]]) -> bool:
        for fmt in formats:
            if not isinstance(fmt, dict):
                continue
            format_id = str(fmt.get('format_id') or '').lower()
            width = int(fmt.get('width') or 0)
            height = int(fmt.get('height') or 0)
            vcodec = fmt.get('vcodec')
            if vcodec == 'none' or ('audio' in format_id and width == 0 and height == 0):
                return True
        return False

    @staticmethod
    def _normalize_status_url_for_oembed(tweet_url: str, twt_id: str) -> str:
        decoded_url = html.unescape(tweet_url or '').strip()
        match = P_TWT_LINK.search(decoded_url)
        if match:
            username = match.group(1)
            return f'https://twitter.com/{username}/status/{match.group(2)}'
        return f'https://twitter.com/i/status/{twt_id}'

    def _parse_ydlp_info(self, info: Mapping[str, Any], twt_id: str) -> Tuple[Dict, Dict, Dict, Dict]:
        pic_dict: Dict[str, Dict[str, str]] = {}
        gif_dict: Dict[str, Dict[str, str]] = {}
        vid_dict: Dict[str, Dict[str, str]] = {}
        text_dict: Dict[str, str] = {}

        raw_entries = info.get('entries')
        entries: List[Mapping[str, Any]]
        if isinstance(raw_entries, list):
            entries = [entry for entry in raw_entries if isinstance(entry, dict)]
            if not entries:
                entries = [info]
        else:
            entries = [info]

        for entry in entries:
            if not isinstance(entry, dict):
                continue

            entry_id = str(entry.get('id') or twt_id)
            text_value = entry.get('description') or entry.get('title')
            if text_value:
                text_dict[entry_id] = text_value

            # Twitter images appear as pbs.twimg.com thumbnails in yt-dlp metadata.
            for thumb in entry.get('thumbnails') or []:
                thumb_url = thumb.get('url') if isinstance(thumb, dict) else None
                if not thumb_url or 'pbs.twimg.com/media/' not in thumb_url:
                    continue
                file_name = self._extract_filename(thumb_url, f'{entry_id}.jpg')
                pic_dict[file_name] = {'url': thumb_url, 'twtId': entry_id}

            entry_formats = [fmt for fmt in (entry.get('formats') or []) if isinstance(fmt, dict)]
            best_video = self._select_best_direct_video_format(entry_formats)

            if best_video:
                media_key = 'gif'
                if best_video.get('acodec') not in ('', 'none') or self._has_audio_only_formats(entry_formats):
                    media_key = 'vid'
                target = gif_dict if media_key == 'gif' else vid_dict
                target[str(best_video['file_name'])] = {'url': str(best_video['url']), 'twtId': entry_id}

        return pic_dict, gif_dict, vid_dict, text_dict

    def _fetch_syndication_fallback(self, twt_id: str) -> Optional[Dict[str, Any]]:
        """Fallback to Twitter syndication API for text/images when yt-dlp cannot extract media."""
        url = f"https://cdn.syndication.twimg.com/tweet-result?id={twt_id}&lang=en"
        try:
            response = self.session.get(url, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return None

            text_value = str(
                payload.get("text")
                or payload.get("full_text")
                or payload.get("display_text")
                or ""
            ).strip()

            pic_list: Dict[str, Dict[str, str]] = {}
            photos = payload.get("photos")
            if isinstance(photos, list):
                for idx, photo in enumerate(photos, start=1):
                    if not isinstance(photo, dict):
                        continue
                    photo_url = photo.get("url")
                    if not photo_url:
                        continue
                    file_name = self._extract_filename(str(photo_url), f"{twt_id}_{idx}.jpg")
                    pic_list[file_name] = {"url": str(photo_url), "twtId": twt_id}

            media_details = payload.get("mediaDetails")
            if isinstance(media_details, list):
                for idx, media in enumerate(media_details, start=1):
                    if not isinstance(media, dict):
                        continue
                    if str(media.get("type") or "").lower() != "photo":
                        continue
                    media_url = media.get("media_url_https") or media.get("media_url")
                    if not media_url:
                        continue
                    file_name = self._extract_filename(str(media_url), f"{twt_id}_m_{idx}.jpg")
                    pic_list[file_name] = {"url": str(media_url), "twtId": twt_id}

            return {
                "picList": pic_list,
                "gifList": {},
                "vidList": {},
                "textList": ({twt_id: text_value} if text_value else {}),
            }
        except Exception as e:
            logger.error(f"Syndication fallback failed for tweet {twt_id}: {e}")
            return None

    @staticmethod
    def _strip_html_text(raw_html: str) -> str:
        text = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _fetch_oembed_text_fallback(self, tweet_url: str, twt_id: str) -> Optional[Dict[str, Any]]:
        normalized_tweet_url = self._normalize_status_url_for_oembed(tweet_url, twt_id)
        encoded_url = urllib.parse.quote(normalized_tweet_url, safe="")
        oembed_url = f"https://publish.twitter.com/oembed?omit_script=1&url={encoded_url}"
        try:
            response = self.session.get(oembed_url, timeout=20, headers=DEFAULT_HTTP_HEADERS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return None

            html_block = str(payload.get("html") or "")
            text_value = self._strip_html_text(html_block)
            pic_list = self._extract_oembed_pic_list(html_block, twt_id)
            if not text_value and not pic_list:
                return None

            return {
                "picList": pic_list,
                "gifList": {},
                "vidList": {},
                "textList": ({twt_id: text_value} if text_value else {}),
            }
        except Exception as e:
            logger.error(f"oEmbed fallback failed for tweet {twt_id}: {e}")
            return None

    def _fetch_fxtwitter_fallback(self, twt_id: str) -> Optional[Dict[str, Any]]:
        api_url = f"https://api.fxtwitter.com/2/status/{twt_id}"
        try:
            response = self.session.get(api_url, timeout=20, headers=DEFAULT_HTTP_HEADERS)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or int(payload.get('code') or 0) != 200:
                return None

            status = payload.get('status')
            if not isinstance(status, dict):
                return None

            text_value = str(status.get('raw_text') or status.get('text') or '').strip()
            media_payload = status.get('media')
            if not isinstance(media_payload, dict):
                media_payload = {}

            pic_list: Dict[str, Dict[str, str]] = {}
            gif_list: Dict[str, Dict[str, str]] = {}
            vid_list: Dict[str, Dict[str, str]] = {}

            for index, item in enumerate(media_payload.get('all') or [], start=1):
                if not isinstance(item, dict):
                    continue

                media_type = str(item.get('type') or '').lower()
                media_url = str(item.get('url') or '').strip()
                formats = item.get('formats')
                if isinstance(formats, list):
                    direct_formats = [fmt for fmt in formats if isinstance(fmt, dict)]
                    best_direct = None
                    best_bitrate = -1
                    for fmt in direct_formats:
                        fmt_url = str(fmt.get('url') or '').strip()
                        if not fmt_url:
                            continue
                        container = str(fmt.get('container') or '').lower()
                        if container != 'mp4' or self._looks_like_hls_manifest(fmt_url, ''):
                            continue
                        bitrate = int(fmt.get('bitrate') or 0)
                        if bitrate >= best_bitrate:
                            best_bitrate = bitrate
                            best_direct = fmt_url
                    if best_direct:
                        media_url = best_direct

                if not media_url:
                    continue

                file_name = self._extract_filename(media_url, f'{twt_id}_fx_{index}.bin')
                if media_type == 'image':
                    pic_list[file_name] = {'url': media_url, 'twtId': twt_id}
                elif media_type == 'gif':
                    gif_list[file_name] = {'url': media_url, 'twtId': twt_id}
                elif media_type == 'video':
                    vid_list[file_name] = {'url': media_url, 'twtId': twt_id}

            if not pic_list and not gif_list and not vid_list and not text_value:
                return None

            return {
                'picList': pic_list,
                'gifList': gif_list,
                'vidList': vid_list,
                'textList': ({twt_id: text_value} if text_value else {}),
            }
        except Exception as e:
            logger.error("fxtwitter fallback failed for tweet %s: %s", twt_id, e)
            return None

    def _fetch_vxtwitter_fallback(self, twt_id: str) -> Optional[Dict[str, Any]]:
        api_url = f"https://api.vxtwitter.com/Twitter/status/{twt_id}"
        try:
            response = self.session.get(api_url, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return None

            text_value = str(payload.get("text") or "").strip()
            pic_list: Dict[str, Dict[str, str]] = {}
            gif_list: Dict[str, Dict[str, str]] = {}
            vid_list: Dict[str, Dict[str, str]] = {}

            media_urls = payload.get("mediaURLs")
            if isinstance(media_urls, list):
                for index, media_url in enumerate(media_urls, start=1):
                    if not isinstance(media_url, str) or not media_url:
                        continue
                    file_name = self._extract_filename(media_url, f"{twt_id}_vx_{index}.jpg")
                    if media_url.lower().endswith(".mp4"):
                        vid_list[file_name] = {"url": media_url, "twtId": twt_id}
                    else:
                        pic_list[file_name] = {"url": media_url, "twtId": twt_id}

            media_extended = payload.get("media_extended")
            if isinstance(media_extended, list):
                for index, media in enumerate(media_extended, start=1):
                    if not isinstance(media, dict):
                        continue
                    media_type = str(media.get("type") or "").lower()
                    media_url = str(media.get("url") or media.get("thumbnail_url") or "").strip()
                    if not media_url:
                        continue
                    file_name = self._extract_filename(media_url, f"{twt_id}_vx_ext_{index}.bin")
                    if media_type == "image":
                        pic_list[file_name] = {"url": media_url, "twtId": twt_id}
                    elif media_type == "gif":
                        gif_list[file_name] = {"url": media_url, "twtId": twt_id}
                    elif media_type == "video":
                        vid_list[file_name] = {"url": media_url, "twtId": twt_id}

            if not pic_list and not gif_list and not vid_list and not text_value:
                return None

            return {
                "picList": pic_list,
                "gifList": gif_list,
                "vidList": vid_list,
                "textList": ({twt_id: text_value} if text_value else {}),
            }
        except Exception as e:
            logger.error("vxtwitter fallback failed for tweet %s: %s", twt_id, e)
            return None

    def _extract_oembed_pic_list(self, html_block: str, twt_id: str) -> Dict[str, Dict[str, str]]:
        pic_list: Dict[str, Dict[str, str]] = {}
        if not html_block:
            return pic_list

        html_text = html.unescape(html_block)
        for image_url in P_PBS_IMAGE_LINK.findall(html_text):
            normalized = image_url.replace("&amp;", "&")
            file_name = self._extract_filename(normalized, f"{twt_id}_oembed.jpg")
            pic_list[file_name] = {"url": normalized, "twtId": twt_id}

        for index, short_url in enumerate(P_PIC_TWITTER_LINK.findall(html_text), start=1):
            resolved_url = self._resolve_pic_twitter_to_image_url(short_url)
            if not resolved_url:
                continue
            file_name = self._extract_filename(resolved_url, f"{twt_id}_oembed_{index}.jpg")
            pic_list[file_name] = {"url": resolved_url, "twtId": twt_id}

        return pic_list

    def _resolve_pic_twitter_to_image_url(self, short_url: str) -> Optional[str]:
        try:
            normalized_short_url = short_url if short_url.startswith("http") else f"https://{short_url}"
            response = self.session.get(normalized_short_url, timeout=20, allow_redirects=True)
            response.raise_for_status()
            final_url = str(response.url or "")
            if "pbs.twimg.com/media/" in final_url:
                return final_url

            html_text = response.text or ""
            match = P_META_IMAGE.search(html_text)
            if not match:
                return None

            image_url = html.unescape(match.group(1)).strip()
            if image_url.startswith("//"):
                image_url = f"https:{image_url}"
            if image_url.startswith("http") and "pbs.twimg.com/" in image_url:
                return image_url
        except Exception as e:
            logger.warning("Failed to resolve pic.twitter short link %s: %s", short_url, e)
        return None

    def _enrich_piclist_from_text_short_links(self, data_dict: Dict[str, Any]) -> None:
        pic_list = data_dict.setdefault("picList", {})
        text_list = data_dict.get("textList", {})
        if not isinstance(text_list, dict):
            return

        matches_found = 0
        resolved_found = 0
        for twt_id, text_value in text_list.items():
            text = str(text_value or "")
            for index, short_url in enumerate(P_PIC_TWITTER_LINK.findall(text), start=1):
                matches_found += 1
                image_url = self._resolve_pic_twitter_to_image_url(short_url)
                if not image_url:
                    continue
                resolved_found += 1
                file_name = self._extract_filename(image_url, f"{twt_id}_pic_{index}.jpg")
                pic_list[file_name] = {"url": image_url, "twtId": str(twt_id)}

        if matches_found:
            logger.info(
                "pic.twitter short-link enrichment: matches=%d resolved=%d total_images=%d",
                matches_found,
                resolved_found,
                len(pic_list),
            )

    @staticmethod
    def _has_payload_content(data: Optional[Dict[str, Any]]) -> bool:
        if not data:
            return False
        return bool(data.get("picList") or data.get("gifList") or data.get("vidList") or data.get("textList"))

    def get_single_tweet_data(self, twt_id: str, tweet_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        extraction_url = f'https://x.com/i/status/{twt_id}'
        fallback_url = tweet_url or extraction_url
        try:
            with yt_dlp.YoutubeDL(cast(Any, self._build_ydl_opts())) as ydl:
                info = ydl.extract_info(extraction_url, download=False)

            if not isinstance(info, dict):
                return None

            pic_list, gif_list, vid_list, text_list = self._parse_ydlp_info(info, twt_id)
            return {'picList': pic_list, 'gifList': gif_list, 'vidList': vid_list, 'textList': text_list}
        except DownloadError as e:
            logger.error(f'yt-dlp extraction error for tweet {twt_id}: {e}')
            fallback = self._fetch_syndication_fallback(twt_id)
            if self._has_payload_content(fallback):
                logger.info("Using syndication fallback for tweet %s after yt-dlp error", twt_id)
                return fallback

            logger.info("Syndication fallback empty for tweet %s, trying oEmbed fallback", twt_id)
            oembed_data = self._fetch_oembed_text_fallback(fallback_url, twt_id)
            if self._has_payload_content(oembed_data):
                logger.info("Using oEmbed text fallback for tweet %s after yt-dlp error", twt_id)
                if oembed_data and oembed_data.get("picList"):
                    return oembed_data

            logger.info("oEmbed fallback has no media for tweet %s, trying fxtwitter fallback", twt_id)
            fx_data = self._fetch_fxtwitter_fallback(twt_id)
            if self._has_payload_content(fx_data):
                logger.info("Using fxtwitter fallback for tweet %s after yt-dlp error", twt_id)
                return fx_data

            logger.info("fxtwitter fallback empty for tweet %s, trying vxtwitter fallback", twt_id)
            vx_data = self._fetch_vxtwitter_fallback(twt_id)
            if self._has_payload_content(vx_data):
                logger.info("Using vxtwitter fallback for tweet %s after yt-dlp error", twt_id)
                return vx_data

            if self._has_payload_content(oembed_data):
                logger.info("Using oEmbed text-only fallback for tweet %s after yt-dlp error", twt_id)
                return oembed_data
            return None
        except Exception as e:
            logger.error(f'Unexpected extraction error for tweet {twt_id}: {e}')
            return None

    def handle_url(self, url: str) -> Optional[Dict[str, Any]]:
        decoded_url = html.unescape(url)
        twt_match = P_TWT_LINK.findall(decoded_url)
        if twt_match:
            twt_id = twt_match[0][1]
            logger.info(f'Identified tweet URL. Tweet ID: {twt_id}')
            return self.get_single_tweet_data(twt_id, decoded_url)
        logger.warning(f'Unsupported or unrecognized Twitter URL: {decoded_url}')
        return None

    def download_media_bytes(self, url: str) -> bytes:
        response = self.session.get(url, stream=True, timeout=20)
        response.raise_for_status()
        return response.content

    def extract_twitter_media(self, url: str) -> Tuple[List[Tuple[str, bytes]], Dict[str, str]]:
        self._activate_guest_token()
        data_dict = self.handle_url(url)
        if not data_dict:
            logger.error(
                "Twitter extraction returned no payload for url=%s. Likely deleted/protected/region-restricted tweet or auth issue.",
                url,
            )
            return [], {}

        self._enrich_piclist_from_text_short_links(data_dict)

        media_list: List[Tuple[str, bytes]] = []
        for item in data_dict.get('picList', {}).values():
            try:
                media_list.append(('pic', self.download_media_bytes(item['url'])))
            except requests.RequestException as e:
                logger.warning(f"Failed to download Twitter/X image media {item.get('url')}: {e}")
        for item in data_dict.get('gifList', {}).values():
            try:
                media_list.append(('gif', self.download_media_bytes(item['url'])))
            except requests.RequestException as e:
                logger.warning(f"Failed to download Twitter/X GIF media {item.get('url')}: {e}")
        for item in data_dict.get('vidList', {}).values():
            try:
                media_list.append(('vid', self.download_media_bytes(item['url'])))
            except requests.RequestException as e:
                logger.warning(f"Failed to download Twitter/X video media {item.get('url')}: {e}")
        text_list = data_dict.get('textList', {})
        if not media_list and not text_list:
            logger.error(
                "Twitter extraction produced empty result for url=%s. payload_keys=%s pic=%d gif=%d vid=%d text=%d",
                url,
                sorted(list(data_dict.keys())),
                len(data_dict.get('picList', {})),
                len(data_dict.get('gifList', {})),
                len(data_dict.get('vidList', {})),
                len(text_list),
            )
        return media_list, text_list
