import yt_dlp
import asyncio
import functools
from typing import Union

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

if __name__ == '__main__':
    # Replace with the URL of the video you want to download
    video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ'
    
    async def main():
        # Example of downloading
        # await download_video_720p_h264(video_url)

        # Example of getting just the title
        title = await get_video_title(video_url)
        if title:
            print(f"Video Title: {title}")

    asyncio.run(main())