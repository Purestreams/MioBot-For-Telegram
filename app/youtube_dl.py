import yt_dlp
import asyncio
import functools
from typing import Union
import httpx
import re
import os
import subprocess

TELEGRAM_VIDEO_MAX_SIZE_BYTES = 50 * 1024 * 1024

async def download_video_720p_h264(url, output_path='output/%(title)s.%(ext)s'):
    """
    Downloads a video from a URL to a 720p H.264 MP4 file asynchronously.

    Args:
        url (str): The URL of the video to download.
        output_path (str): The output template for the filename.
                           Defaults to the video's title.
    """
    
    ydl_opts = {
        # Select the best 720p video with h264 codec and the best audio,
        # and merge them into an mp4 file.
        'format': 'bestvideo[height<=720][vcodec^=avc]+bestaudio/best[height<=720][vcodec^=avc]',
        'merge_output_format': 'mp4',
        'outtmpl': output_path,
        'postprocessors': [{
            'key': 'FFmpegVideoConvertor',
            'preferedformat': 'mp4',  # The container format
        }],
        'postprocessor_args': [
            '-c:v', 'copy',  # Copy the video stream without re-encoding
            '-c:a', 'aac',   # Re-encode the audio to AAC
            '-b:a', '128k',  # Set the audio bitrate to 128Kbps
        ],
    }

    loop = asyncio.get_running_loop()

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"Starting download for: {url}")
            # Run the synchronous download method in a separate thread
            await loop.run_in_executor(
                None, functools.partial(ydl.download, [url])
            )
            print("Download completed successfully.")
            # Return the title of the video
    except Exception as e:
        print(f"An error occurred: {e}")


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

    base, _ = os.path.splitext(input_path)
    compressed_path = f"{base}_compressed.mp4"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "28",
        "-c:a", "aac",
        "-b:a", "96k",
        "-movflags", "+faststart",
        "-fs", str(max_size_bytes),
        compressed_path,
    ]

    loop = asyncio.get_running_loop()
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
    if os.path.getsize(compressed_path) > max_size_bytes:
        size_limit_mb = max_size_bytes / (1024 * 1024)
        raise RuntimeError(f"Compressed video still exceeds the {size_limit_mb:.1f}MB limit.")

    return compressed_path

async def get_video_title(url: str) -> Union[str, None]:
    """
    Extracts the title of a video from a URL without downloading.

    Args:
        url (str): The URL of the video.

    Returns:
        str: The title of the video, or None if it can't be fetched.
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    loop = asyncio.get_running_loop()
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Use run_in_executor for the synchronous extract_info method
            info_dict = await loop.run_in_executor(
                None, lambda: ydl.extract_info(url, download=False)
            )
            title = info_dict.get('title')
            title = ''.join(c for c in title if c.isalnum() or c.isspace())
            title = title.replace(' ', '_')
            title = title.strip()
            return title
    except Exception as e:
        print(f"An error occurred while fetching video title: {e}")
        return None
    
async def get_bilibili_permanent_url(url: str) -> Union[str, None]:
    """
    Fetches the permanent URL for a Bilibili video.

    Args:
        url (str): The original Bilibili video URL.

    Returns:
        str: The permanent URL of the video, or None if it can't be fetched.
    """
    try:
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.head(url)
            #return str(response.url)
    except Exception as e:
        print(f"An error occurred while fetching permanent URL: {e}")
        return None
    
    if 'location' in response.headers:
        return get_bilibili_permanent_url(response.headers['location'])
    else:
        #return response.url
        pattern = r'https?://www\.bilibili\.com/video/[^/?]+'

        match = re.search(pattern, str(response.url))
        if match:
            return match.group(0)
        else:
            print(f"Could not extract permanent URL from: {response.url}")
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
