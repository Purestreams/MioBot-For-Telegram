#!/usr/bin/env bash
set -euo pipefail

apt-get update
apt-get install texlive-latex-recommended texlive-fonts-recommended texlive-xetex texlive-latex-extra ffmpeg -y

curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
#python3 -m pip install --upgrade pip
#python3 -m pip install python-telegram-bot markdown2 pillow aiofiles aiohttp requests beautifulsoup4 playwright openai aiosqlite reportlab yt-dlp pypdfium2 numpy fastembed onnxruntime --upgrade

# Playwright optimization:
# 1) Do not use --with-deps here because system deps are already handled above.
# 2) Skip browser download when headless shell is already cached.
PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-$HOME/.cache/ms-playwright}"
export PLAYWRIGHT_BROWSERS_PATH

if find "$PLAYWRIGHT_BROWSERS_PATH" -maxdepth 1 -type d -name 'chromium_headless_shell-*' | grep -q .; then
	echo "Playwright chromium headless shell already installed, skipping download."
else
	playwright install chromium --only-shell
fi


uv run python main.py