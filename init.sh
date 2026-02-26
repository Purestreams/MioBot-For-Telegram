apt-get update
apt-get install texlive-full texlive-xetex texlive-latex-extra ffmpeg -y
python3 -m pip install --upgrade pip
python3 -m pip install python-telegram-bot markdown2 pillow aiofiles aiohttp requests beautifulsoup4 playwright openai aiosqlite reportlab yt-dlp pypdfium2 numpy fastembed onnxruntime --upgrade
playwright install chromium --only-shell --with-deps
