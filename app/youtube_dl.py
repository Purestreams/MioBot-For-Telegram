import yt_dlp
import asyncio
import functools
from typing import Optional
import httpx
import re
import os
import hashlib
import subprocess
import logging
from urllib.parse import urljoin

TELEGRAM_VIDEO_MAX_SIZE_BYTES = 50 * 1024 * 1024
COMPRESSION_TARGET_RATIO = 0.96
MAX_FILENAME_BYTES = 240
logger = logging.getLogger(__name__)

DEFAULT_HTTP_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/135.0.0.0 Safari/537.36'
    ),
}

BILIBILI_HTTP_HEADERS = {
    **DEFAULT_HTTP_HEADERS,
    'Referer': 'https://www.bilibili.com/',
    'Origin': 'https://www.bilibili.com',
}

BILIBILI_URL_REGEX = (
    r'(https?://)?(?:www\.|m\.)?'
    r'(bilibili\.com/|b23\.tv/)'
    r'(?:video/|watch\?bvid=)?'
    r'([A-Za-z0-9_-]{6,12})'
    r'(?:[/?#][^\s]*)?'
)


def _is_bilibili_url(url: str) -> bool:
    return bool(re.match(BILIBILI_URL_REGEX, url or ''))


def _extract_bilibili_canonical_url(url: str) -> Optional[str]:
    match = re.search(r'https?://www\.bilibili\.com/video/[^/?]+', url or '')
    if not match:
        return None
    return match.group(0)


def _build_ydl_base_opts(url: str) -> dict:
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'noprogress': True,
        'noplaylist': True,
        'retries': 3,
        'fragment_retries': 3,
        'socket_timeout': 30,
        'http_headers': dict(DEFAULT_HTTP_HEADERS),
    }
    if _is_bilibili_url(url):
        ydl_opts['http_headers'] = dict(BILIBILI_HTTP_HEADERS)
    return ydl_opts


def _truncate_filename_component(component: str, extension: str = '', *, suffix: str = '') -> str:
    normalized_component = (component or '').strip() or 'video'
    extension = extension or ''
    suffix = suffix or ''

    full_name = f'{normalized_component}{suffix}{extension}'
    if len(os.fsencode(full_name)) <= MAX_FILENAME_BYTES:
        return normalized_component

    digest = hashlib.sha1(os.fsencode(full_name)).hexdigest()[:10]
    marker = f'_{digest}'
    reserved_bytes = len(os.fsencode(f'{suffix}{marker}{extension}'))
    budget = max(16, MAX_FILENAME_BYTES - reserved_bytes)

    encoded = os.fsencode(normalized_component)
    truncated = encoded[:budget]
    while truncated:
        try:
            truncated_component = truncated.decode('utf-8')
            break
        except UnicodeDecodeError:
            truncated = truncated[:-1]
    else:
        truncated_component = 'video'

    truncated_component = truncated_component.rstrip(' ._') or 'video'
    return f'{truncated_component}{marker}'


def _normalize_output_path(output_path: str) -> str:
    if not output_path:
        return output_path

    directory, filename = os.path.split(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if not filename or '%(' in filename:
        return output_path

    stem, extension = os.path.splitext(filename)
    normalized_stem = _truncate_filename_component(stem, extension)
    if normalized_stem == stem:
        return output_path
    return os.path.join(directory, f'{normalized_stem}{extension}')


def _build_compressed_output_path(input_path: str) -> str:
    directory, filename = os.path.split(input_path)
    stem, _ = os.path.splitext(filename)
    compressed_stem = _truncate_filename_component(stem, '.mp4', suffix='_compressed')
    return os.path.join(directory, f'{compressed_stem}_compressed.mp4')

async def download_video_720p_h264(url, output_path='output/%(title)s.%(ext)s'):
    """
    Downloads a video from a URL to a 720p H.264 MP4 file asynchronously.

    Args:
        url (str): The URL of the video to download.
        output_path (str): The output template for the filename.
                           Defaults to the video's title.
    """
    
    normalized_output_path = _normalize_output_path(output_path)

    ydl_opts = {
        **_build_ydl_base_opts(url),
        # Prefer H.264 when available, but keep a broader fallback so site-side
        # format changes do not turn into hard download failures.
        'format': (
            'bestvideo[height<=720][vcodec^=avc1]+bestaudio/'
            'bestvideo[height<=720][vcodec^=avc]+bestaudio/'
            'bestvideo[height<=720]+bestaudio/'
            'best[height<=720]/best'
        ),
        'merge_output_format': 'mp4',
        'outtmpl': normalized_output_path,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',
        }],
        'postprocessor_args': [
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '128k',
        ],
    }

    loop = asyncio.get_running_loop()

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info("Starting download for: %s", url)
            # Run the synchronous download method in a separate thread
            await loop.run_in_executor(
                None, functools.partial(ydl.download, [url])
            )
            logger.info("Download completed successfully")
    except Exception as e:
        logger.error("Video download failed for %s: %s", url, e)
        raise


async def compress_video_if_needed(input_path: str, max_size_bytes: int = TELEGRAM_VIDEO_MAX_SIZE_BYTES) -> str:
    """
    Compresses a video with ffmpeg when it exceeds Telegram bot size limits.

    Args:
        input_path (str): Source video path.
        max_size_bytes (int): Maximum allowed size in bytes.

    Returns:
        str: Original path if already within limit, otherwise compressed path.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video file not found: {input_path}")

    if os.path.getsize(input_path) <= max_size_bytes:
        return input_path

    compressed_path = _build_compressed_output_path(input_path)

    target_size_bytes = min(max_size_bytes, max(1, int(max_size_bytes * COMPRESSION_TARGET_RATIO)))
    compression_attempts = [
        {"preset": "fast", "crf": "30", "audio_bitrate": "96k", "video_filter": None},
        {
            "preset": "medium",
            "crf": "34",
            "audio_bitrate": "80k",
            "video_filter": "scale=-2:480:force_original_aspect_ratio=decrease",
        },
        {
            "preset": "medium",
            "crf": "36",
            "audio_bitrate": "64k",
            "video_filter": "scale=-2:360:force_original_aspect_ratio=decrease",
        },
    ]

    loop = asyncio.get_running_loop()
    last_size_bytes: Optional[int] = None
    for attempt_number, attempt in enumerate(compression_attempts, start=1):
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
        ]
        if attempt["video_filter"]:
            cmd.extend(["-vf", str(attempt["video_filter"])])
        cmd.extend(
            [
                "-c:v", "libx264",
                "-preset", str(attempt["preset"]),
                "-crf", str(attempt["crf"]),
                "-c:a", "aac",
                "-b:a", str(attempt["audio_bitrate"]),
                "-movflags", "+faststart",
                "-fs", str(target_size_bytes),
                compressed_path,
            ]
        )

        try:
            await loop.run_in_executor(
                None,
                functools.partial(
                    subprocess.run,
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                ),
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is required to compress oversized videos.") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to compress video: {exc.stderr}") from exc

        if not os.path.exists(compressed_path):
            raise RuntimeError("Compressed video file was not created.")

        last_size_bytes = os.path.getsize(compressed_path)
        if last_size_bytes <= max_size_bytes:
            return compressed_path

        logger.info(
            "Compression attempt %d still exceeds Telegram size limit: %.2fMB > %.2fMB",
            attempt_number,
            last_size_bytes / (1024 * 1024),
            max_size_bytes / (1024 * 1024),
        )

    size_limit_mb = max_size_bytes / (1024 * 1024)
    current_size_mb = (last_size_bytes or 0) / (1024 * 1024)
    raise RuntimeError(
        f"Compressed video still exceeds the {size_limit_mb:.1f}MB limit after {len(compression_attempts)} attempts "
        f"(current size: {current_size_mb:.1f}MB)."
    )

    return compressed_path

async def get_video_title(url: str) -> Optional[str]:
    """
    Extracts the title of a video from a URL without downloading.

    Args:
        url (str): The URL of the video.

    Returns:
        str: The title of the video, or None if it can't be fetched.
    """
    ydl_opts = _build_ydl_base_opts(url)
    loop = asyncio.get_running_loop()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Use run_in_executor for the synchronous extract_info method
            info_dict = await loop.run_in_executor(
                None, lambda: ydl.extract_info(url, download=False)
            )
            title = (info_dict or {}).get('title')
            if not isinstance(title, str) or not title.strip():
                return None
            title = ''.join(c for c in title if c.isalnum() or c.isspace())
            title = title.replace(' ', '_')
            title = title.strip()
            return title
    except Exception as e:
        logger.warning("Failed to fetch video title for %s: %s", url, e)
        return None


async def resolve_caption_url(video_url: str) -> str:
    """Resolve canonical Bilibili URL for caption display when available."""
    if re.match(BILIBILI_URL_REGEX, video_url):
        permanent_url = await get_bilibili_permanent_url(video_url)
        if permanent_url:
            return permanent_url
    return video_url


async def download_video_to_file(video_url: str, output_file_path: str) -> str:
    """Download supported video media and return a display title."""
    video_title = await get_video_title(video_url)
    await download_video_720p_h264(video_url, output_path=_normalize_output_path(output_file_path))
    return video_title or "Video"
    
async def get_bilibili_permanent_url(url: str) -> Optional[str]:
    """
    Fetches the permanent URL for a Bilibili video.

    Args:
        url (str): The original Bilibili video URL.

    Returns:
        str: The permanent URL of the video, or None if it can't be fetched.
    """
    if not isinstance(url, str) or not url.strip():
        return None

    canonical_from_input = _extract_bilibili_canonical_url(url)
    if canonical_from_input:
        return canonical_from_input

    try:
        async with httpx.AsyncClient(
            follow_redirects=False,
            timeout=20.0,
            headers=BILIBILI_HTTP_HEADERS,
        ) as client:
            response = await client.head(url)
            response.raise_for_status()
    except Exception as e:
        logger.warning("Failed to resolve Bilibili URL %s: %s", url, e)
        return None

    response_headers = getattr(response, 'headers', {}) or {}
    location = response_headers.get('location', '')
    resolved_url = str(response.url)
    if location:
        resolved_url = urljoin(str(response.url), location)

    canonical_url = _extract_bilibili_canonical_url(resolved_url)
    if canonical_url:
        return canonical_url

    logger.warning("Could not extract Bilibili permanent URL from: %s", resolved_url)
    return None


if __name__ == '__main__':
    # Replace with the URL of the video you want to download
    # video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    video_url = 'https://b23.tv/Enqggyo'
    
    async def main():
        # Example of downloading
        # await download_video_720p_h264(video_url)

        # Example of getting just the title
        title = await get_video_title(video_url)
        if title:
            print(f"Video Title: {title}")
        
        await download_video_720p_h264(video_url, output_path=f'output/{title}.mp4')

    async def test_bilibili_url():
        original_url = 'https://b23.tv/Enqggyo'
        permanent_url = await get_bilibili_permanent_url(original_url)
        if permanent_url:
            print(f"Permanent URL: {permanent_url}")
        else:
            print("Failed to fetch permanent URL.")

    #asyncio.run(main())
    asyncio.run(test_bilibili_url())
