# MioBot Web Admin

A private FastAPI admin UI for MioBot.

## Run

```bash
uv run miobot-webadmin
```

Default URL:

```text
http://127.0.0.1:8765
```

Running `python main.py` also starts webadmin automatically in a child process. When running the Docker image, publish port `8765` from the container if you need browser access.

## Login

Create a one-time login URL in a private Telegram chat with the bot:

```text
/webadmin_token
/webadmin_token 30m
```

Tokens are short-lived and single-use. The web session is stored in an HttpOnly signed cookie.

## Features

- Switch UI language between Chinese and English.
- View dashboard counts and DB schema version.
- Browse chat IDs and recent chat messages.
- View and edit user memory summaries.
- Edit or archive user memory facts.
- Accept or reject pending memory candidates.
- View, add, edit, and archive chat-scoped global memory facts.

## Runtime Settings

```text
WEBADMIN_HOST=127.0.0.1
WEBADMIN_PORT=8765
WEBADMIN_BASE_URL=http://127.0.0.1:8765
WEBADMIN_SESSION_TTL_SECONDS=43200
WEBADMIN_COOKIE_SECURE=0
WEBADMIN_SECRET_KEY=
```

Set `WEBADMIN_SECRET_KEY` to a long random value in production. Keep the service bound to localhost unless it is protected by HTTPS and an authenticated reverse proxy.
