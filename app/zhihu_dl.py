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
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import urllib.error, http.cookiejar

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
DEFAULT_COOKIE_FILE = os.path.join('config', 'zhihu_cookies.txt')
ANSWER_INCLUDE = 'content,voteup_count,comment_count,created_time,updated_time,author,question'
LOGIN_COOKIE_NAMES = frozenset({'z_c0'})


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


def extract_ids(url):
    m = re.search(r'/question/(\d+)/answer/(\d+)', url)
    if m: return m.group(1), m.group(2)
    m = re.search(r'/answer/(\d+)', url)
    if m: return None, m.group(1)
    m = re.search(r'answer_id=(\d+)', url)
    if m: return None, m.group(1)
    if url.isdigit(): return None, url
    raise ValueError(f"无法解析链接: {url[:80]}")


def strip_html(text):
    text = html.unescape(text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<p[^>]*>', '\n', text)
    text = re.sub(r'</p>', '\n', text)
    text = re.sub(r'<[^>]+>', '', text)
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


def fmt(data):
    q = data.get('question') or {}
    a = data.get('author') or {}
    ts = data.get('created_time', 0)
    return {
        'question': q.get('title', '(无标题)'),
        'author': a.get('name', '(匿名)'),
        'author_url': a.get('url_token', ''),
        'content': strip_html(data.get('content', '')),
        'time': datetime.fromtimestamp(ts, tz=timezone(timedelta(hours=8))
        ).strftime('%Y-%m-%d %H:%M') if ts else '未知',
    }


def parse_link(link, patient=False, cookie=None, cookie_file=None):
    return fmt(fetch(link, patient=patient, cookie=cookie, cookie_file=cookie_file))


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
