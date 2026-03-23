import html
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)


P_TWT_LINK = re.compile(r'https://(?:x|twitter)\.com/(.+?)/status/(\d+)')
P_CSRF_TOKEN = re.compile(r'ct0=(.+?)(?:;|$)')
P_PIC_LINK = re.compile(r'''(https://pbs\.twimg\.com/media/(.+?))['"]''')
P_GIF_LINK = re.compile(r'(https://video\.twimg\.com/tweet_video/(.+?\.mp4))')
P_VID_LINK = re.compile(r'(https://video\.twimg\.com/ext_tw_video/(\d+)/(?:pu|pr)/vid/(avc1/)?(\d+x\d+)/(.+?\.mp4))')
P_VID_LINK1 = re.compile(r'(https://video\.twimg\.com/amplify_video/(\d+)/vid/(avc1/)?(\d+x\d+)/(.+?\.mp4))')

SINGLE_PAGE_API = 'https://x.com/i/api/graphql/BbmLpxKh8rX8LNe2LhVujA/TweetDetail'
HOST_URL = 'https://api.twitter.com/1.1/guest/activate.json'

SINGLE_PAGE_API_PAR = '{{"focalTweetId":"{}","with_rux_injections":false,"includePromotedContent":false,"withCommunity":false,"withQuickPromoteEligibilityTweetFields":false,"withBirdwatchNotes":false,"withSuperFollowsUserFields":true,"withDownvotePerspective":false,"withReactionsMetadata":false,"withReactionsPerspective":false,"withSuperFollowsTweetFields":true,"withVoice":true,"withV2Timeline":true}}'
SINGLE_PAGE_API_PAR2 = '{"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"responsive_web_home_pinned_timelines_enabled":true,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"c9s_tweet_anatomy_moderator_badge_enabled":true,"tweetypie_unmention_optimization_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":false,"tweet_awards_web_tipping_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_media_download_video_enabled":false,"responsive_web_enhance_cards_enabled":false}'
SINGLE_PAGE_API_PAR3 = '{"withArticleRichContentState":false}'


def is_twitter_status_url(url: str) -> bool:
    return bool(P_TWT_LINK.search(html.unescape(url or '')))


class TwitterDownloader:
    def __init__(self, cookie: Optional[str] = None, proxies: Optional[dict] = None):
        self.auth_token = 'Bearer AAAAAAAAAAAAAAAAAAAAANRILgAAAAAAnNwIzUejRCOuH5E6I8xnZz4puTs%3D1Zv7ttfk8LF81IUq16cHjhLTvJu4FA33AGWWjCpTnA'
        self.cookie = cookie or os.getenv('TWITTER_COOKIE', '')
        self.proxies = proxies or {}

        self.session = requests.Session()
        if self.proxies:
            self.session.proxies.update(self.proxies)

        self.headers = {
            'authorization': self.auth_token,
            'Cookie': self.cookie,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
        }
        csrf_token = self._extract_csrf_token(self.cookie)
        if csrf_token:
            self.headers['x-csrf-token'] = csrf_token
        self.session.headers.update(self.headers)

    @staticmethod
    def _extract_csrf_token(cookie_str: str) -> Optional[str]:
        if not cookie_str:
            return None
        match = P_CSRF_TOKEN.findall(cookie_str)
        return match[0] if match else None

    def _activate_guest_token(self) -> None:
        if self.cookie and 'x-csrf-token' in self.headers:
            return
        try:
            response = self.session.post(HOST_URL, timeout=5)
            response.raise_for_status()
            guest_token = response.json().get('guest_token')
            if guest_token:
                self.session.headers.update({'x-guest-token': guest_token})
        except Exception as e:
            logger.error(f'Error fetching guest token: {e}')

    def _parse_tweet_content(self, tw_content: dict, twt_id: str) -> Tuple[Dict, Dict, Dict, Dict]:
        pic_dict, gif_dict, vid_dict, text_dict = {}, {}, {}, {}
        str_content = json.dumps(tw_content)

        pic_links = P_PIC_LINK.findall(str_content)
        for link, filename in pic_links:
            pic_dict[filename] = {'url': f'{link}?name=orig', 'twtId': twt_id}

        gif_links = P_GIF_LINK.findall(str_content)
        for link, filename in gif_links:
            gif_dict[filename] = {'url': link, 'twtId': twt_id}

        vid_links = P_VID_LINK.findall(str_content) + P_VID_LINK1.findall(str_content)
        if vid_links:
            best_choices: Dict[str, Dict[str, Any]] = {}
            for link_match in vid_links:
                url, file_id, _, resolution_str, filename = link_match[:5]
                best_choices.setdefault(file_id, {'resolution': 0, 'file_name': None, 'url': None})
                try:
                    width, height = map(int, resolution_str.split('x'))
                    res_val = width * height
                    if res_val > best_choices[file_id]['resolution']:
                        best_choices[file_id].update({'resolution': res_val, 'file_name': filename, 'url': url})
                except Exception:
                    continue
            for choice in best_choices.values():
                if choice['file_name']:
                    vid_dict[choice['file_name']] = {'url': choice['url'], 'twtId': twt_id}

        try:
            result_node = tw_content.get('itemContent', {}).get('tweet_results', {}).get('result', {})
            if 'note_tweet' in result_node:
                text_dict[twt_id] = result_node['note_tweet']['note_tweet_results']['result']['text']
            elif 'tweet' in result_node:
                text_dict[twt_id] = result_node['tweet']['legacy']['full_text']
            elif 'legacy' in result_node:
                text_dict[twt_id] = result_node['legacy']['full_text']
        except Exception as e:
            logger.debug(f'Failed to parse text for {twt_id}: {e}')

        return pic_dict, gif_dict, vid_dict, text_dict

    def get_single_tweet_data(self, twt_id: str) -> Optional[Dict[str, Any]]:
        params = {
            'variables': SINGLE_PAGE_API_PAR.format(twt_id),
            'features': SINGLE_PAGE_API_PAR2,
            'fieldToggles': SINGLE_PAGE_API_PAR3,
        }
        try:
            response = self.session.get(SINGLE_PAGE_API, params=params, timeout=10)
            if response.status_code != 200:
                logger.warning(f'Failed to fetch content for tweet {twt_id}. Status: {response.status_code}')
                return None
            page_content = response.text
            if 'Age-restricted adult content' in page_content or 'Sorry, that page does not exist' in page_content:
                return None
            if f'"tweet-{twt_id}"' not in page_content:
                return None

            data = response.json()
            instructions = data.get('data', {}).get('threaded_conversation_with_injections_v2', {}).get('instructions', [])
            entries = []
            for instruction in instructions:
                entries.extend(instruction.get('entries', []))

            tw_content = None
            for entry in entries:
                if f'tweet-{twt_id}' in entry.get('entryId', ''):
                    tw_content = entry.get('content')
                    break
            if not tw_content and entries:
                tw_content = entries[0].get('content')
            if not tw_content:
                return None

            pic_list, gif_list, vid_list, text_list = self._parse_tweet_content(tw_content, twt_id)
            return {'picList': pic_list, 'gifList': gif_list, 'vidList': vid_list, 'textList': text_list}
        except requests.RequestException as e:
            logger.error(f'Request error fetching tweet {twt_id}: {e}')
            return None
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            logger.error(f'Data parse error for tweet {twt_id}: {e}')
            return None

    def handle_url(self, url: str) -> Optional[Dict[str, Any]]:
        decoded_url = html.unescape(url)
        twt_match = P_TWT_LINK.findall(decoded_url)
        if twt_match:
            twt_id = twt_match[0][1]
            logger.info(f'Identified tweet URL. Tweet ID: {twt_id}')
            return self.get_single_tweet_data(twt_id)
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
            return [], {}

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
        return media_list, data_dict.get('textList', {})
