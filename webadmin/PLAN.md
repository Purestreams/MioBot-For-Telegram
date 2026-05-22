# Web Admin Plan

## Implementation Status

The first usable version is implemented in this folder. It provides a FastAPI app, Telegram-issued one-time token login, signed web sessions, a static admin UI with Chinese/English switching, chat browsing, personal memory editing, memory candidate review, and chat-scoped global memory editing.

## Goal

Build a private admin web UI for MioBot that lets authorized admins inspect chat history and manage user/global memory without exposing Telegram credentials or long-lived web passwords.

## Authentication Flow

1. Admin opens a private Telegram chat with the bot.
2. Admin runs a command such as `/webadmin_token`.
3. Bot verifies the sender with the existing `TELEGRAM_ADMIN_USER_IDS` logic.
4. Bot creates a short-lived, single-use login token and stores only a hash in the database.
5. Bot replies in private chat with a local/admin URL containing the raw token, or with the token text to paste into the web login page.
6. Web admin exchanges the token for a signed session cookie.
7. Token is marked used and cannot be reused.

Suggested token policy:

- 10 minute expiry.
- Single use.
- Store `token_hash`, `telegram_user_id`, `telegram_username`, `expires_at`, `used_at`, `created_at`.
- Use HTTPS when exposed outside localhost.
- Do not log raw tokens.

## Database Additions

Add a versioned migration for:

```sql
CREATE TABLE webadmin_login_tokens (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  token_hash TEXT NOT NULL UNIQUE,
  admin_user_id INTEGER,
  admin_username TEXT,
  expires_at DATETIME NOT NULL,
  used_at DATETIME,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_webadmin_login_tokens_expiry
ON webadmin_login_tokens (expires_at, used_at);
```

The existing `db_metadata.schema_version` should be bumped when this lands.

## Backend Shape

Create a small FastAPI app under `webadmin/`:

```text
webadmin/
  __init__.py
  app.py              # FastAPI app factory
  auth.py             # token exchange, session cookies, admin checks
  database.py         # thin async wrappers around app.database helpers
  schemas.py          # Pydantic response/request models
  templates/          # optional server-rendered pages
  static/             # optional CSS/JS
```

Dependencies to consider:

- `fastapi`
- `uvicorn`
- `itsdangerous` or signed cookie support
- `jinja2` if using server-rendered HTML

## Initial Pages

1. Login page
   - Token input.
   - No username/password.

2. Dashboard
   - DB schema version.
   - Message count.
   - User memory count.
   - Global memory chat list.

3. Chat browser
   - List chat IDs with message counts and latest activity.
   - Open a chat timeline.
   - Filter/search messages.
   - Show reply-chain metadata.

4. User memory manager
   - List users with summaries/facts/candidates.
   - View one user memory.
   - Edit summary.
   - Edit/archive facts.
   - Accept/reject candidates.
   - Trigger refresh.

5. Group global memory manager
   - List chat IDs.
   - View facts for one chat.
   - Add/update/archive facts.

## API Sketch

```text
POST /auth/token-login
POST /auth/logout
GET  /api/me
GET  /api/dashboard
GET  /api/chats
GET  /api/chats/{chat_id}/messages?limit=100&before_id=...
GET  /api/memory/users
GET  /api/memory/users/{telegram_user_key}
PUT  /api/memory/users/{telegram_user_key}/summary
POST /api/memory/users/{telegram_user_key}/refresh
POST /api/memory/candidates/{candidate_id}/accept
POST /api/memory/candidates/{candidate_id}/reject
GET  /api/global-memory/chats
GET  /api/global-memory/chats/{chat_id}
POST /api/global-memory/chats/{chat_id}/facts
DELETE /api/global-memory/chats/{chat_id}/facts/{fact_id}
```

## Telegram Command Additions

Add private admin-only command:

```text
/webadmin_token
```

Optional args:

```text
/webadmin_token 30m
```

Keep it conservative at first: max expiry 30 minutes, default 10 minutes.

## Security Notes

- Bind to `127.0.0.1` by default.
- If exposed remotely, put it behind HTTPS and a reverse proxy.
- Use `SameSite=Lax`, `HttpOnly`, `Secure` cookies when HTTPS is enabled.
- Add CSRF protection for form posts if server-rendered pages are used.
- Require admin token/session for every route except login/static assets.
- Redact secrets and avoid returning Telegram bot token/runtime config.

## Implementation Phases

1. Add token table migration and token helper functions.
2. Add `/webadmin_token` private admin command.
3. Create minimal FastAPI app with token login and dashboard.
4. Add chat browser read-only pages.
5. Add user memory editing flows.
6. Add group global memory editing flows.
7. Add tests for auth, token expiry/single-use behavior, and database helpers.
8. Add run instructions and optional `miobot-webadmin` script.
