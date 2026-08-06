#!/usr/bin/env python3
"""
Zhihu 解析器

参考 zhihu-cli 的稳定抓取路径：
- 使用 requests.Session 维持会话
- 使用统一 Chrome 浏览器指纹请求头
- 支持登录 Cookie 或 cookie 文件
- 匿名访问被 403 拦截时快速失败，而不是静默长时间重试

用法:
  python3 zhihu_parser.py <知乎链接>
  python3 zhihu_parser.py --cookie-file config/zhihu_cookies.txt <知乎链接>
  python3 zhihu_parser.py --patience <链接>
  python3 zhihu_parser.py --test <链接> [次数]
"""

import html
import ipaddress
import json
import mimetypes
import os
import re
import socket
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.error, http.cookiejar
from urllib.parse import unquote, urljoin, urlsplit

import requests

try:
    from app.runtime_config import get_runtime_value
except ImportError:  # pragma: no cover - direct execution from app/ still works
    from runtime_config import get_runtime_value

CHROME_VERSION = '145'
DEFAULT_USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    f'Chrome/{CHROME_VERSION}.0.0.0 Safari/537.36'
)
DEFAULT_HEADERS = {
    'User-Agent': DEFAULT_USER_AGENT,
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.zhihu.com/',
    'sec-ch-ua': (
        f'"Not:A-Brand";v="99", '
        f'"Google Chrome";v="{CHROME_VERSION}", '
        f'"Chromium";v="{CHROME_VERSION}"'
    ),
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
}
IMAGE_DOWNLOAD_HEADERS = {
    'User-Agent': DEFAULT_USER_AGENT,
    'Accept-Language': DEFAULT_HEADERS['Accept-Language'],
}
DEFAULT_COOKIE_FILE = os.path.join('config', 'zhihu_cookies.txt')
ANSWER_INCLUDE = 'content,voteup_count,comment_count,created_time,updated_time,author,question'
QUESTION_ANSWERS_INCLUDE = 'data[*].question'
LOGIN_COOKIE_NAMES = frozenset({'z_c0'})

ZHIHU_ANSWER_LINK = re.compile(
    r'(?i)(?<![\w@.])(?:https?://)?(?:www\.)?zhihu\.com/'
    r'(?:question/(?P<question_id>\d+(?![A-Za-z0-9_]))/answer/(?P<answer_id>\d+(?![A-Za-z0-9_]))|answer/(?P<direct_answer_id>\d+(?![A-Za-z0-9_])))'
)
ZHIHU_ARTICLE_LINK = re.compile(
    r'(?i)(?<![\w@.])(?:https?://)?(?:www\.)?zhihu\.com/'
    r'(?:article/(?P<article_id>\d+)(?![A-Za-z0-9_])|'
    r'column/[^/\s]+/p/(?P<column_article_id>\d+)(?![A-Za-z0-9_]))'
    r'|(?<![\w@.])(?:https?://)?zhuanlan\.zhihu\.com/p/'
    r'(?P<zhuanlan_article_id>\d+)(?![A-Za-z0-9_])'
)
ZHIHU_POST_LINK = re.compile(
    r'(?i)(?<![\w@.])(?:https?://)?(?:www\.)?zhihu\.com/'
    r'(?:(?:pin|p)/(?P<post_id>\d+(?![A-Za-z0-9_]))|people/[^/\s]+/(?:(?:pins|posts)/(?P<people_post_id>\d+(?![A-Za-z0-9_]))))'
)
ZHIHU_QUESTION_LINK = re.compile(
    r'(?i)(?<![\w@.])(?:https?://)?(?:www\.)?zhihu\.com/question/(?P<bare_question_id>\d+(?![A-Za-z0-9_]))(?!/answer/)'
)

MAX_ZHIHU_IMAGE_BYTES = 20 * 1024 * 1024
MAX_ZHIHU_IMAGES = 20
MAX_ZHIHU_IMAGE_REDIRECTS = 3
_IMAGE_URL_TOKEN = re.compile(r'(?i)(?:https?:)?//[^\s<>"\']+')
_IMAGE_ATTRIBUTE_URL = re.compile(
    r'(?i)\b(src|data-src|data-original|data-actualsrc|data-url|'
    r'image-url|image_url|thumbnail|thumbnail_url|original|original_url)\s*=\s*["\']([^"\']+)["\']'
)
_IMAGE_UNQUOTED_ATTRIBUTE_URL = re.compile(
    r'(?i)\b(src|data-src|data-original|data-actualsrc|data-url|'
    r'image-url|image_url|thumbnail|thumbnail_url|original|original_url)\s*=\s*((?:https?:)?//[^\s>]+)'
)
_CSS_IMAGE_URL = re.compile(r'(?i)url\(\s*["\']?((?:https?:)?//[^)\s"\']+)["\']?\s*\)')
_META_TAG = re.compile(r'(?is)<meta\b[^>]*>')
_MEDIA_TAG = re.compile(r'(?is)<(img|picture|source|video|audio)\b[^>]*>')
_SRCSET_ATTRIBUTE = re.compile(r'(?i)\bsrcset\s*=\s*["\']([^"\']+)["\']')
_HTML_ATTRIBUTE = re.compile(r'(?i)([\w:-]+)\s*=\s*["\']([^"\']*)["\']')
_IMAGE_EXTENSION = re.compile(r'(?i)\.(?:jpe?g|png|gif|webp|bmp|avif|heic)(?:[?#&]|$)')
_IMAGE_HOST = re.compile(r'(?i)(?:^|\.)zhimg\.com$')
_IMAGE_MIME_EXTENSION = {
    'image/jpeg': '.jpg',
    'image/png': '.png',
    'image/gif': '.gif',
    'image/webp': '.webp',
    'image/bmp': '.bmp',
    'image/avif': '.avif',
    'image/heic': '.heic',
}


def debug_log(message):
    now = datetime.now().strftime('%H:%M:%S')
    print(f"[zhihu_dl {now}] {message}", flush=True)


def cookie_str_to_dict(cookie_str):
    cookie_dict = {}
    for item in str(cookie_str or '').split(';'):
        item = item.strip()
        if '=' not in item:
            continue
        key, value = item.split('=', 1)
        key = key.strip()
        value = value.strip()
        if key:
            cookie_dict[key] = value
    return cookie_dict


def load_cookie_dict_from_file(cookie_file):
    cookie_path = Path(cookie_file)
    jar = http.cookiejar.MozillaCookieJar()
    try:
        jar.load(str(cookie_path), ignore_discard=True, ignore_expires=True)
    except (OSError, http.cookiejar.LoadError):
        return cookie_str_to_dict(cookie_path.read_text(encoding='utf-8').strip())
    return {cookie.name: cookie.value for cookie in jar}


def resolve_cookie_config(cookie=None, cookie_file=None):
    configured_cookie = cookie or get_runtime_value('ZHIHU_COOKIE')
    configured_file = cookie_file or get_runtime_value('ZHIHU_COOKIE_FILE') or DEFAULT_COOKIE_FILE
    if configured_file and os.path.isfile(configured_file):
        return load_cookie_dict_from_file(configured_file), f'file:{configured_file}'
    if configured_cookie:
        return cookie_str_to_dict(configured_cookie), 'inline'
    return {}, f'missing:{configured_file}'


def build_session(cookie=None, cookie_file=None):
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)

    cookie_dict, cookie_source = resolve_cookie_config(cookie=cookie, cookie_file=cookie_file)
    if cookie_dict:
        for name, value in cookie_dict.items():
            if value is None:
                continue
            session.cookies.set(str(name), str(value), domain='.zhihu.com')
        debug_log(f"cookie source={cookie_source} keys={','.join(sorted(cookie_dict))}")
    else:
        debug_log(f"cookie source={cookie_source}")

    try:
        response = session.get('https://www.zhihu.com/signin', timeout=15)
        debug_log(f"preflight signin status={response.status_code}")
    except requests.RequestException as e:
        debug_log(f"preflight signin failed: {e}")

    xsrf = session.cookies.get('_xsrf')
    if xsrf:
        session.headers['x-xsrftoken'] = xsrf

    cookie_keys = ','.join(sorted(session.cookies.keys())) or '(none)'
    debug_log(f"session cookies={cookie_keys}")
    return session


def has_login_cookie(session):
    return any(session.cookies.get(name) for name in LOGIN_COOKIE_NAMES)


def classify_link(url):
    """Classify a Zhihu URL without confusing answers, articles, or posts."""
    value = str(url or '').strip()
    if ZHIHU_ANSWER_LINK.search(value):
        return 'answer'
    if ZHIHU_ARTICLE_LINK.search(value):
        return 'article'
    if ZHIHU_POST_LINK.search(value):
        return 'post'
    if ZHIHU_QUESTION_LINK.search(value):
        return 'question'
    return None


def _extract_resource_id(url, content_type):
    patterns = {
        'article': ZHIHU_ARTICLE_LINK,
        'post': ZHIHU_POST_LINK,
        'question': ZHIHU_QUESTION_LINK,
    }
    pattern = patterns.get(content_type)
    match = pattern.search(str(url or '')) if pattern else None
    if not match:
        raise ValueError(f'无法解析知乎{content_type or "内容"}链接: {str(url or "")[:80]}')
    for key, value in match.groupdict().items():
        if value:
            return value
    raise ValueError(f'无法解析知乎{content_type or "内容"}链接: {str(url or "")[:80]}')


def extract_ids(url):
    m = re.search(r'/question/(\d+)/answer/(\d+)', url)
    if m: return m.group(1), m.group(2)
    m = re.search(r'/answer/(\d+)', url)
    if m: return None, m.group(1)
    m = re.search(r'answer_id=(\d+)', url)
    if m: return None, m.group(1)
    if url.isdigit(): return None, url
    raise ValueError(f"无法解析链接: {url[:80]}")


def _normalize_image_url(value):
    """Normalize image references found in HTML, JSON, or escaped API text."""
    normalized = html.unescape(str(value or '')).strip()
    normalized = normalized.replace('\\/', '/').replace('\\u002F', '/')
    if normalized.startswith('//'):
        normalized = f'https:{normalized}'
    if not re.match(r'https?://', normalized, flags=re.IGNORECASE):
        return ''
    return normalized.rstrip(".,!?;:'\"”’)]}>，。！？；：")


def _is_safe_image_url(value):
    """Allow only public HTTP(S) image targets and reject SSRF destinations."""
    normalized = _normalize_image_url(value)
    if not normalized:
        return False
    try:
        parsed = urlsplit(normalized)
    except ValueError:
        return False
    if parsed.scheme.lower() not in {'http', 'https'} or parsed.username or parsed.password:
        return False
    hostname = (parsed.hostname or '').rstrip('.').lower()
    if not hostname or hostname in {'localhost', 'localhost.localdomain'} or hostname.endswith('.localhost'):
        return False

    try:
        literal_ip = ipaddress.ip_address(hostname)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        return literal_ip.is_global

    try:
        addresses = {
            sockaddr[0]
            for _family, _socktype, _proto, _canonname, sockaddr in socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme.lower() == 'https' else 80),
            )
        }
    except (OSError, socket.gaierror, ValueError):
        return False
    try:
        return bool(addresses) and all(ipaddress.ip_address(address).is_global for address in addresses)
    except ValueError:
        return False


def _safe_image_log_url(value):
    try:
        parsed = urlsplit(str(value or ''))
    except ValueError:
        return '<invalid-image-url>'
    host = parsed.hostname or ''
    path = (parsed.path or '/')[:100]
    return f'{parsed.scheme}://{host}{path}' if host else '<invalid-image-url>'


def _is_zhihu_image_host(value):
    try:
        host = (urlsplit(str(value or '')).hostname or '').lower().rstrip('.')
    except ValueError:
        return False
    return host == 'zhimg.com' or host.endswith('.zhimg.com') or host == 'zhihu.com' or host.endswith('.zhihu.com')


def _is_image_url(value, *, allow_generic=False):
    normalized = _normalize_image_url(value)
    if not normalized:
        return False
    parsed = urlsplit(normalized)
    host = (parsed.hostname or '').lower()
    if _IMAGE_HOST.search(host):
        return True
    if _IMAGE_EXTENSION.search(parsed.path or ''):
        return True
    return bool(allow_generic)


def _append_image_url(urls, seen, value, *, allow_generic=False):
    normalized = _normalize_image_url(value)
    if not normalized or not _is_image_url(normalized, allow_generic=allow_generic):
        return
    key = normalized.casefold()
    if key in seen:
        return
    seen.add(key)
    urls.append(normalized)


def _extract_image_urls_from_text(value, *, allow_generic=False):
    """Extract image URLs from rich HTML and Zhihu's plain image-link format."""
    urls = []
    seen = set()
    text = str(value or '')

    # Generic ``src`` is only considered an image inside an image-capable tag;
    # otherwise <video src=...> and <audio src=...> can be misclassified.
    attribute_text = re.sub(r'(?is)<(?:video|audio)\b[^>]*>', '', text)
    for attribute_name, candidate in _IMAGE_ATTRIBUTE_URL.findall(attribute_text):
        _append_image_url(
            urls,
            seen,
            candidate,
            allow_generic=attribute_name.lower() != 'src',
        )
    for attribute_name, candidate in _IMAGE_UNQUOTED_ATTRIBUTE_URL.findall(attribute_text):
        _append_image_url(
            urls,
            seen,
            candidate,
            allow_generic=attribute_name.lower() != 'src',
        )
    for tag_match in _MEDIA_TAG.finditer(text):
        tag_name = tag_match.group(1).lower()
        tag = tag_match.group(0)
        if tag_name in {'video', 'audio'}:
            for poster in re.findall(r'(?i)\bposter\s*=\s*["\']([^"\']+)["\']', tag):
                _append_image_url(urls, seen, poster, allow_generic=True)
            continue
        source_attributes = {
            key.lower(): value for key, value in _HTML_ATTRIBUTE.findall(tag)
        }
        source_type = source_attributes.get('type', '')
        if tag_name == 'source' and source_type.lower().startswith(('video/', 'audio/')):
            continue
        for _attribute_name, candidate in _IMAGE_ATTRIBUTE_URL.findall(tag):
            _append_image_url(urls, seen, candidate, allow_generic=True)
        for _attribute_name, candidate in _IMAGE_UNQUOTED_ATTRIBUTE_URL.findall(tag):
            _append_image_url(urls, seen, candidate, allow_generic=True)
        for srcset in _SRCSET_ATTRIBUTE.findall(tag):
            for candidate in srcset.split(','):
                image_candidate = candidate.strip().split(None, 1)[0]
                _append_image_url(urls, seen, image_candidate, allow_generic=True)
    for candidate in _CSS_IMAGE_URL.findall(text):
        _append_image_url(urls, seen, candidate, allow_generic=True)
    for tag in _META_TAG.findall(text):
        attributes = {key.lower(): value for key, value in _HTML_ATTRIBUTE.findall(tag)}
        image_name = attributes.get('property') or attributes.get('name') or ''
        if image_name.lower() in {'og:image', 'og:image:url', 'twitter:image', 'twitter:image:src', 'image'}:
            _append_image_url(urls, seen, attributes.get('content'), allow_generic=True)
    for candidate in _IMAGE_URL_TOKEN.findall(text):
        _append_image_url(urls, seen, candidate, allow_generic=allow_generic)
    return urls


def extract_image_urls(data):
    """Return embedded image URLs from any supported Zhihu payload shape."""
    urls = []
    seen = set()
    media_key_markers = (
        'content', 'body', 'html', 'detail', 'description', 'text', 'image',
        'photo', 'picture', 'thumbnail', 'cover', 'media', 'src',
    )
    metadata_keys = {'author', 'question', 'creator', 'user', 'owner', 'relationship'}

    def visit(value, key_hint='', media_context=False):
        if isinstance(value, dict):
            for key, nested in value.items():
                key_name = str(key).lower()
                child_context = False if key_name in metadata_keys else (
                    media_context or any(marker in key_name for marker in media_key_markers)
                )
                visit(nested, f'{key_hint} {key_name}'.strip(), child_context)
            return
        if isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested, key_hint, media_context)
            return
        if not isinstance(value, str):
            return
        if not media_context:
            return

        allow_generic = any(
            marker in key_hint
            for marker in ('image', 'photo', 'picture', 'thumbnail', 'cover', 'media', 'src')
        )
        for candidate in _extract_image_urls_from_text(value, allow_generic=allow_generic):
            _append_image_url(urls, seen, candidate, allow_generic=True)

    visit(data, media_context=isinstance(data, str))
    return urls


def strip_html(text, image_urls=None):
    text = html.unescape(text)
    text = text.replace('\\/', '/')
    # Remove image/source tags before generic tag stripping so image URLs do
    # not leak into the text response.
    text = re.sub(r'<(?:img|source)\b[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<p[^>]*>', '\n', text)
    text = re.sub(r'</p>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    image_urls = image_urls if image_urls is not None else extract_image_urls(text)
    for image_url in image_urls:
        text = text.replace(image_url, '')
    text = _IMAGE_URL_TOKEN.sub(
        lambda match: '' if _is_image_url(match.group(0)) else match.group(0),
        text,
    )
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text.strip() or '（无内容）'


def make_request(session, url):
    response = session.get(url, params={'include': ANSWER_INCLUDE}, timeout=15)
    debug_log(f"response status={response.status_code} bytes={len(response.content)}")
    response.raise_for_status()
    return response.json()


def fetch(link, patient=False, cookie=None, cookie_file=None):
    clean = link.split('?')[0] if '?' in link else link
    _qid, aid = extract_ids(clean)
    api_url = f'https://www.zhihu.com/api/v4/answers/{aid}'

    max_retries = 30 if patient else 10
    base_delay = 3 if patient else 2
    session = build_session(cookie=cookie, cookie_file=cookie_file)

    debug_log(
        f"start fetch answer={aid} patient={patient} max_retries={max_retries} base_delay={base_delay}"
    )
    if not has_login_cookie(session):
        debug_log('no z_c0 cookie present; anonymous access may be blocked')

    try:
        for attempt in range(1, max_retries + 1):
            debug_log(f"attempt {attempt}/{max_retries} request {api_url}")
            try:
                data = make_request(session, api_url)
                if 'error' not in data:
                    debug_log(f"attempt {attempt} success keys={','.join(sorted(data.keys())[:6])}")
                    return data

                error = data.get('error') or {}
                error_code = error.get('code', 'unknown')
                error_message = str(error.get('message') or 'unknown api error')
                if error_code == 40362 and not has_login_cookie(session):
                    raise RuntimeError(
                        '知乎接口拒绝匿名访问；请提供 z_c0 登录 cookie '
                        '（--cookie / --cookie-file / ZHIHU_COOKIE / ZHIHU_COOKIE_FILE）。'
                    )

                delay = min(base_delay ** attempt, 180)
                debug_log(
                    f"attempt {attempt} api error code={error_code} sleep {delay:.1f}s message={error_message[:80]}"
                )
                if attempt < max_retries:
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"重试 {max_retries} 次仍失败: {error_message}")
            except requests.HTTPError as e:
                status = e.response.status_code if e.response is not None else 'unknown'
                body_excerpt = ''
                if e.response is not None:
                    body_excerpt = ' '.join(e.response.text.split())[:160]
                debug_log(f"attempt {attempt} HTTPError code={status} body={body_excerpt}")
                if status == 403 and not has_login_cookie(session):
                    raise RuntimeError(
                        '知乎接口返回 403；当前会话没有 z_c0 登录 cookie。'
                        '请提供 --cookie、--cookie-file、ZHIHU_COOKIE 或 ZHIHU_COOKIE_FILE。'
                    )
                if status == 403 and attempt < max_retries:
                    delay = min(base_delay ** attempt, 180)
                    debug_log(f"attempt {attempt} hit 403, sleep {delay:.1f}s then retry")
                    time.sleep(delay)
                    continue
                raise RuntimeError(f"HTTP {status}")
            except (requests.RequestException, OSError) as e:
                debug_log(f"attempt {attempt} network error: {e}")
                if attempt < max_retries:
                    debug_log('sleep 5.0s then retry')
                    time.sleep(5)
                    continue
                raise RuntimeError(f"网络: {e}")
    finally:
        session.close()

    raise RuntimeError("未知错误")


def _rich_text(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ('text', 'content', 'body', 'description', 'detail', 'excerpt', 'title'):
            text = _rich_text(value.get(key))
            if text:
                return text
        return ''
    if isinstance(value, list):
        return '\n'.join(text for text in (_rich_text(item) for item in value) if text).strip()
    return str(value)


def _extract_html_field(raw_html, pattern):
    match = re.search(pattern, raw_html or '', flags=re.IGNORECASE | re.DOTALL)
    return html.unescape(match.group(1)).strip() if match else ''


def _extract_meta_content(raw_html, names):
    wanted = {str(name).lower() for name in names}
    for tag in _META_TAG.findall(raw_html or ''):
        attributes = {key.lower(): html.unescape(value).strip() for key, value in _HTML_ATTRIBUTE.findall(tag)}
        name = (attributes.get('property') or attributes.get('name') or '').lower()
        if name in wanted and attributes.get('content'):
            return attributes['content']
    return ''


def _fetch_page_fallback(session, link, content_type):
    response = session.get(link, timeout=15)
    response.raise_for_status()
    raw_html = response.text or ''
    article_html = _extract_html_field(raw_html, r'<article\b[^>]*>(.*?)</article>')
    return {
        'content_type': content_type,
        'title': _extract_meta_content(raw_html, ('og:title', 'twitter:title'))
        or _extract_html_field(raw_html, r'<title[^>]*>(.*?)</title>'),
        'content': article_html or _extract_html_field(
            raw_html,
            r'<meta\b[^>]*(?:property|name)=["\'](?:og:description|description|twitter:description)["\'][^>]*content=["\']([^"\']+)',
        ) or _extract_meta_content(raw_html, ('og:description', 'description', 'twitter:description')),
        'html': raw_html,
    }


def _fetch_question_via_answers(session, resource_id):
    """Recover public question metadata when the question endpoint is 403."""
    endpoint = f'https://www.zhihu.com/api/v4/questions/{resource_id}/answers'
    response = session.get(
        endpoint,
        params={'limit': 1, 'offset': 0, 'include': QUESTION_ANSWERS_INCLUDE},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    for answer in (payload.get('data') or []) if isinstance(payload, dict) else []:
        if not isinstance(answer, dict):
            continue
        question = answer.get('question')
        if isinstance(question, dict):
            return {
                'content_type': 'question',
                'question': question,
                'title': question.get('title'),
                'content': question.get('detail') or '',
                'created_time': question.get('created') or question.get('created_time') or 0,
            }
    return None


def fetch_resource(link, content_type, cookie=None, cookie_file=None):
    """Fetch an article/post/question using its resource API with HTML fallback."""
    resource_id = _extract_resource_id(link, content_type)
    endpoint = {
        'article': f'https://www.zhihu.com/api/v4/articles/{resource_id}',
        'post': f'https://www.zhihu.com/api/v4/pins/{resource_id}',
        'question': f'https://www.zhihu.com/api/v4/questions/{resource_id}',
    }[content_type]
    session = build_session(cookie=cookie, cookie_file=cookie_file)
    try:
        try:
            response = session.get(endpoint, timeout=15)
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and 'error' not in payload:
                payload.setdefault('content_type', content_type)
                return payload
        except (requests.RequestException, ValueError, OSError) as exc:
            debug_log(f'{content_type} API fetch failed; trying page fallback: {exc}')

        if content_type == 'question':
            try:
                question_payload = _fetch_question_via_answers(session, resource_id)
                if question_payload:
                    return question_payload
            except (requests.RequestException, ValueError, OSError) as exc:
                debug_log(f'question answers fallback failed; trying page fallback: {exc}')

        return _fetch_page_fallback(session, link, content_type)
    finally:
        session.close()


def _looks_like_image_bytes(payload):
    if not payload:
        return False
    return (
        payload.startswith(b'\xff\xd8\xff')  # JPEG
        or payload.startswith(b'\x89PNG\r\n\x1a\n')
        or payload.startswith((b'GIF87a', b'GIF89a'))
        or (payload.startswith(b'RIFF') and payload[8:12] == b'WEBP')
        or payload.startswith(b'\x00\x00\x00\x0cJXL \r\n\x87\n')
    )


def _image_filename(url, content_type, index, payload):
    suffix = os.path.splitext(unquote(urlsplit(url).path).rsplit('/', 1)[-1])[1].lower()
    allowed_suffixes = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.avif', '.heic'}
    if suffix not in allowed_suffixes:
        mime = (content_type or '').split(';', 1)[0].strip().lower()
        suffix = _IMAGE_MIME_EXTENSION.get(mime) or mimetypes.guess_extension(mime) or ''
    if not suffix:
        if payload.startswith(b'\xff\xd8\xff'):
            suffix = '.jpg'
        elif payload.startswith(b'\x89PNG'):
            suffix = '.png'
        elif payload.startswith((b'GIF87a', b'GIF89a')):
            suffix = '.gif'
        elif payload.startswith(b'RIFF') and payload[8:12] == b'WEBP':
            suffix = '.webp'
        else:
            suffix = '.jpg'
    return f'zhihu_image_{index}{suffix}'


def _request_image_response(session, image_url):
    current_url = _normalize_image_url(image_url)
    for _ in range(MAX_ZHIHU_IMAGE_REDIRECTS + 1):
        if not _is_safe_image_url(current_url):
            raise ValueError(f'unsafe image URL: {_safe_image_log_url(current_url)}')

        headers = {
            'Accept': 'image/avif,image/webp,image/apng,image/*,*/*;q=0.8',
        }
        if _is_zhihu_image_host(current_url):
            headers['Referer'] = 'https://www.zhihu.com/'
        response = session.get(
            current_url,
            headers=headers,
            timeout=20,
            stream=True,
            allow_redirects=False,
        )
        status_code = int(getattr(response, 'status_code', 200) or 0)
        if 300 <= status_code < 400:
            location = response.headers.get('Location') or response.headers.get('location')
            response.close()
            if not location:
                raise ValueError(f'image redirect missing location: {_safe_image_log_url(current_url)}')
            current_url = urljoin(current_url, str(location))
            continue
        response.raise_for_status()
        return response, current_url
    raise ValueError(f'too many image redirects: {_safe_image_log_url(image_url)}')


def download_image_media(image_urls):
    """Download extracted Zhihu images without sending API credentials.

    Each result contains ``content`` bytes plus a safe filename and MIME type,
    allowing the Telegram layer to choose photo or document delivery.  A
    single inaccessible image does not discard the rest of the answer.
    """
    urls = []
    seen = set()
    for image_url in image_urls or []:
        normalized = _normalize_image_url(image_url)
        if normalized and normalized.casefold() not in seen:
            seen.add(normalized.casefold())
            urls.append(normalized)
    urls = urls[:MAX_ZHIHU_IMAGES]
    if not urls:
        return []

    # The API/page fetch uses the configured cookie.  Image URLs can point at
    # Zhihu's CDN or another host, so never attach the authenticated cookie jar
    # to these requests.  Do not inherit the API session's Zhihu Referer for
    # third-party image hosts; _request_image_response adds it only for Zhihu.
    session = requests.Session()
    session.headers.update(IMAGE_DOWNLOAD_HEADERS)
    media = []
    try:
        for index, image_url in enumerate(urls, start=1):
            response = None
            try:
                response, resolved_url = _request_image_response(session, image_url)
                content_length = int(response.headers.get('Content-Length') or 0)
                if content_length > MAX_ZHIHU_IMAGE_BYTES:
                    debug_log(f'skip image over size limit bytes={content_length} url={_safe_image_log_url(resolved_url)}')
                    continue

                chunks = []
                total = 0
                iterator = response.iter_content(chunk_size=64 * 1024)
                for chunk in iterator:
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_ZHIHU_IMAGE_BYTES:
                        raise ValueError(f'image exceeds {MAX_ZHIHU_IMAGE_BYTES} bytes')
                    chunks.append(chunk)
                payload = b''.join(chunks)
                content_type = str(response.headers.get('Content-Type') or '').split(';', 1)[0].strip().lower()
                if not content_type.startswith('image/') and not _looks_like_image_bytes(payload):
                    debug_log(f'skip non-image response type={content_type or "unknown"} url={_safe_image_log_url(resolved_url)}')
                    continue
                media.append({
                    'url': resolved_url,
                    'content': payload,
                    'content_type': content_type or 'image/jpeg',
                    'filename': _image_filename(resolved_url, content_type, index, payload),
                })
            except (requests.RequestException, OSError, ValueError) as exc:
                debug_log(f'image download failed url={_safe_image_log_url(image_url)} error={exc}')
            finally:
                if response is not None:
                    response.close()
    finally:
        session.close()
    return media


def fmt(data, content_type='answer'):
    content_type = str(data.get('content_type') or content_type or 'answer') if isinstance(data, dict) else content_type
    if not isinstance(data, dict):
        data = {}

    q = data.get('question') or {}
    a = data.get('author') or {}
    if not isinstance(a, dict):
        a = {}
    ts = data.get('created_time', 0)
    title = _rich_text(data.get('title') or data.get('name'))
    if isinstance(q, dict):
        title = title or _rich_text(q.get('title'))
    elif q:
        title = title or _rich_text(q)

    content = _rich_text(
        data.get('content')
        or data.get('body')
        or data.get('excerpt')
        or data.get('description')
        or data.get('detail')
        or data.get('text')
    )
    if not content and data.get('html'):
        content = _rich_text(data.get('html'))

    image_urls = extract_image_urls(data)

    return {
        'content_type': content_type,
        'question': title or '(无标题)',
        'title': title or '(无标题)',
        'author': a.get('name', '(匿名)'),
        'author_url': a.get('url_token', ''),
        'content': strip_html(content, image_urls=image_urls),
        'image_urls': image_urls,
        'time': datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=8))
        ).strftime('%Y-%m-%d %H:%M') if ts else '未知',
    }


def parse_link(link, patient=False, cookie=None, cookie_file=None):
    content_type = classify_link(link)
    if content_type == 'answer':
        data = fetch(link, patient=patient, cookie=cookie, cookie_file=cookie_file)
    elif content_type in {'article', 'post', 'question'}:
        data = fetch_resource(link, content_type, cookie=cookie, cookie_file=cookie_file)
    else:
        raise ValueError(f'无法识别知乎链接: {str(link or "")[:80]}')
    return fmt(data, content_type=content_type)


def parse_cli(argv):
    clean_args = []
    options = {'cookie': None, 'cookie_file': None}
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == '--cookie':
            if index + 1 >= len(argv):
                raise ValueError('--cookie 需要一个值')
            options['cookie'] = argv[index + 1]
            index += 2
            continue
        if item == '--cookie-file':
            if index + 1 >= len(argv):
                raise ValueError('--cookie-file 需要一个路径')
            options['cookie_file'] = argv[index + 1]
            index += 2
            continue
        clean_args.append(item)
        index += 1
    args = [item for item in clean_args if not item.startswith('-')]
    flags = [item for item in clean_args if item.startswith('-')]
    return args, flags, options


if __name__ == '__main__':
    try:
        args, flags, options = parse_cli(sys.argv[1:])
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    patient = '--patience' in flags or '-p' in flags

    if '--test' in flags:
        link = args[0] if len(args) > 0 else None
        count = int(args[1]) if len(args) > 1 else 10
        if not link:
            print("用法: --test <链接> [次数]")
            sys.exit(1)
        ok = 0
        for i in range(1, count + 1):
            print(f"\n--- #{i}/{count} ---", flush=True)
            try:
                d = fetch(
                    link,
                    patient=True,
                    cookie=options['cookie'],
                    cookie_file=options['cookie_file'],
                )
                r = fmt(d)
                print(f"📌 {r['question']}")
                print(f"👤 {r['author']} @{r['author_url']}")
                print(f"📝 {r['content'][:80]}...")
                ok += 1
            except Exception as e:
                print(f"❌ {e}")
            if i < count:
                time.sleep(8)
        print(f"\n✅ {ok}/{count} 成功")
        sys.exit(0 if ok == count else 1)

    link = args[0] if args else None
    if not link:
        link = "https://www.zhihu.com/question/1951390530626889625/answer/2032324947259942097"
        print(f"用法: python3 zhihu_parser.py <链接>\n")
        print(f"演示: {link}\n")

    debug_log(f"main start link={link} patient={patient}")
    try:
        d = fetch(
            link,
            patient=patient,
            cookie=options['cookie'],
            cookie_file=options['cookie_file'],
        )
        r = fmt(d)
    except Exception as e:
        print()
        print(f"❌ {e}")
        print()
        sys.exit(1)

    print()
    print('═' * 55)
    print(f"📌 问题：{r['question']}")
    print(f"👤 回答者：{r['author']} (@{r['author_url']})")
    print(f"📝 回答：")
    print(r['content'])
    print(f"🕐 {r['time']}")
    print('═' * 55)
    print()
