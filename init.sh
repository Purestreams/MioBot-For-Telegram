#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-xetex texlive-latex-extra texlive-pstricks texlive-lang-chinese ffmpeg -y

curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --extra med
uv run playwright install chromium --only-shell

#python3 -m pip install --upgrade pip
#python3 -m pip install python-telegram-bot markdown2 pillow aiofiles aiohttp requests beautifulsoup4 playwright openai aiosqlite reportlab yt-dlp pypdfium2 numpy fastembed onnxruntime --upgrade



uv run python main.py