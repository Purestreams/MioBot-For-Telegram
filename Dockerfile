FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

EXPOSE 8765

# Runtime packages: ffmpeg for yt-dlp post-processing, fonts for CJK rendering,
# and XeLaTeX dependencies for /med2jpg.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    fonts-noto-cjk \
    texlive-xetex \
    texlive-latex-extra \
    texlive-pstricks \
    texlive-lang-chinese \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal files first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY app ./app
COPY config ./config
COPY webadmin ./webadmin
COPY main.py ./
COPY init.sh ./

# Install project and Playwright browser runtime.
RUN python -m pip install --upgrade pip \
    && python -m pip install ".[med]" \
    && python -m playwright install --with-deps chromium

RUN chmod +x /app/init.sh

# Prepare runtime directories.
RUN mkdir -p /app/output /app/data

# Use an unprivileged user at runtime.
RUN useradd -m -u 10001 botuser \
    && chown -R botuser:botuser /app /ms-playwright
USER botuser

CMD ["./init.sh"]
