FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

# Runtime packages: ffmpeg for yt-dlp post-processing, fonts for CJK rendering.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    ca-certificates \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Copy minimal files first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY app ./app
COPY config ./config
COPY main.py ./

# Install project and Playwright browser runtime.
RUN python -m pip install --upgrade pip \
    && python -m pip install . \
    && python -m playwright install --with-deps chromium

# Prepare runtime directories.
RUN mkdir -p /app/output /app/data

# Use an unprivileged user at runtime.
RUN useradd -m -u 10001 botuser \
    && chown -R botuser:botuser /app /ms-playwright
USER botuser

CMD ["python", "main.py"]
