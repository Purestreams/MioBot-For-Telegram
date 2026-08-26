# MioBot

MioBot is an async Telegram bot for rendering text, downloading media, and joining group conversations with retrieval-augmented context and per-user long-term memory.

The default README language is English. A Chinese version is available below: [中文说明](#中文说明).

## Highlights

- Markdown, plain text, `.txt`, and `.md` rendering to image.
- Automatic YouTube, Bilibili, Twitter/X, and Zhihu link handling.
- Context-aware group replies powered by recent chat history, hybrid RAG, and personal memory.
- Photo and sticker understanding through Ark vision models.
- Optional sticker replies, including sticker-only reactions, selected from the cached sticker database with quality tags and usage cooldown.
- SQLite-backed message history, embeddings, sticker cache, memory summaries, structured facts, and memory candidates.
- Private Telegram admin tools for inspecting, regenerating, and editing user memory.

## Features

### Text To Image

- `/md2jpg` renders Markdown directly to an image.
- `/text2jpg` asks the configured LLM to convert plain text into Markdown first, then renders it.
- Uploaded `.txt` and `.md` files use the same rendering pipeline.

Examples:

```text
/md2jpg ,,,# Title
Some *markdown* here,,,

/text2jpg ,,,Some plain text here,,,
```

Related modules: [app/text2md.py](app/text2md.py), [app/md2jpg.py](app/md2jpg.py).

### Media Download

- Text messages containing one or more YouTube, Bilibili, Twitter/X, or Zhihu links trigger the media flow automatically; links can be protocol-less, punctuation-wrapped, or Telegram text links.
- YouTube and Bilibili use [app/youtube_dl.py](app/youtube_dl.py), prefer MP4 up to 720p, and try ffmpeg compression when Telegram size limits are exceeded.
- Twitter/X uses [app/twitter_downloader.py](app/twitter_downloader.py) and supports text/article posts, photos, videos, GIFs, and any combination of those payloads.
- Supported links in group photo captions are processed alongside the image’s normal vision/reply pipeline.
- Zhihu uses [app/zhihu_dl.py](app/zhihu_dl.py) with separate handling for answer/response links, column articles, posts/ideas, and bare questions.
- Zhihu rich-content images are extracted from API/HTML payloads, downloaded from the image CDN without forwarding API cookies, and sent as Telegram photo albums (with large-image document fallback).
- After successful media delivery, the original link message is deleted.

### Group Replies

Group text, photo, and sticker messages converge into the same reply pipeline in [main.py](main.py):

1. Store the message and reply-chain metadata in SQLite.
2. Read cached personal memory and schedule background memory maintenance.
3. Detect direct triggers: replying to the bot, mentioning `@BotUsername`, `mioo`, or `小小宫`.
4. For ambient messages, run an activation probe before replying.
5. Build prompt context from recent messages, hybrid RAG results, personal memory, and runtime state.
6. Generate the reply, optionally use a matching cached sticker with or without text, and store bot responses back into the database.

### Multimodal Context

- Photos are summarized by [app/image2text.py](app/image2text.py) before entering the group reply pipeline.
- Stickers are described once and cached in `sticker_descriptions`; cached `file_id` values can also be reused for outbound sticker replies.
- Sticker cache rows include searchable tags, mood, `safe_for_reply`, `use_count`, and `last_used_at` so outbound replies avoid risky or overused stickers.
- Vision uses Ark Responses API derived from the single `ARK_API_ENDPOINT` setting. You do not need a separate responses endpoint variable.

### Personal Memory

MioBot keeps memory in layers:

- `user_memories`: compact summary text for prompt injection.
- `user_memory_facts`: structured long-term facts with type, confidence, evidence message IDs, and active/archive state.
- `user_memory_candidates`: pending fast/slow candidates extracted from high-signal user messages.

High-signal messages create pending candidates in the background. Fast candidates or enough accumulated slow candidates trigger consolidation into structured facts. Memory refreshes can also archive stale or contradictory fact IDs returned by the model.

### Extra Commands

- `/crypto` returns Coinbase price data and Allez APR snapshots.
- `/med2jpg` converts natural-language prescription requests into a rendered medical image. This requires optional TeX/PDF dependencies.

### Web Admin

MioBot includes a small FastAPI web admin under [webadmin](webadmin). It is intended for private operation, bound to `127.0.0.1` by default.

Start it with:

```bash
miobot-webadmin
```

or:

```bash
uv run miobot-webadmin
```

Running `main.py` also starts the web admin automatically in a child process. In Docker, the default container command is still `python main.py`, so publish the port explicitly when you need browser access, for example `-p 8765:8765`.

Create a one-time login URL from a private Telegram chat with the bot:

```text
/webadmin_token
/webadmin_token 30m
```

The admin UI supports Chinese and English, browsing chat history, editing personal memory summaries and facts, accepting or rejecting memory candidates, and editing group-scoped global memory.

## Architecture

| Area | Files | Notes |
| --- | --- | --- |
| Entrypoint and handlers | [main.py](main.py) | Telegram setup, handler registration, startup health checks |
| Runtime config | [app/runtime_config.py](app/runtime_config.py) | Loads env files and derives Ark chat/Responses endpoints |
| LLM abstraction | [app/ai_model.py](app/ai_model.py) | Ark, Azure OpenAI, and Ollama chat completion wrapper |
| Rendering | [app/text2md.py](app/text2md.py), [app/md2jpg.py](app/md2jpg.py) | Text shaping, HTML rendering, Playwright screenshot |
| Media | [app/youtube_dl.py](app/youtube_dl.py), [app/twitter_downloader.py](app/twitter_downloader.py), [app/zhihu_dl.py](app/zhihu_dl.py) | Download, compression, captions, fallback parsing, and type-aware Zhihu extraction |
| Group replies | [app/reply2message.py](app/reply2message.py), [main.py](main.py) | Activation probe and reply generation |
| Storage and RAG | [app/database.py](app/database.py), [app/rag_embeddings.py](app/rag_embeddings.py) | SQLite, embeddings, vector search, keyword search, reindexing |
| Personal memory | [app/user_memory.py](app/user_memory.py) | Candidate extraction, summary refresh, structured facts |
| Vision and stickers | [app/image2text.py](app/image2text.py) | Ark Responses API image/sticker understanding |
| Shared helpers | [app/main_helpers.py](app/main_helpers.py) | URL parsing, reply metadata, RAG query building |

## Implementation Details

### Startup And Runtime Configuration

Startup begins in [main.py](main.py). The first important step is calling [app/runtime_config.py](app/runtime_config.py) before importing modules that read environment variables at import time. Runtime values are loaded with first-value-wins semantics: an existing process environment value is kept, then `config/runtime.env` is loaded, then `config/runtime.local.env`, then built-in defaults are applied.

The LLM provider is configured once through [app/ai_model.py](app/ai_model.py). `LLM_PROVIDER` selects `ark`, `azure`, or `ollama`. Ark uses one external endpoint, `ARK_API_ENDPOINT`; runtime helpers derive both `/chat/completions` and `/responses` URLs from that value so text and vision stay on the same base endpoint.

Before polling starts, the bot:

1. Verifies FastEmbed can load the configured embedding model.
2. Initializes or migrates all SQLite tables.
3. Checks embedding metadata in the database.
4. Automatically reindexes old or drifted embeddings when the runtime embedding signature no longer matches stored rows.
5. Registers Telegram command handlers, message handlers, and the global error handler.

### Telegram Handler Layout

[main.py](main.py) owns the Telegram orchestration layer. It registers command handlers first, then document/text/photo/sticker handlers:

- `/start` returns a short capability hint.
- `/help` returns the current feature and command list.
- `/md2jpg` and `/text2jpg` share the same command parser and rendering path.
- `/med2jpg` calls the prescription generator and renderer.
- `/crypto` replies with price and APR data from [app/cryto.py](app/cryto.py).
- `Document.ALL` handles uploaded `.txt` and `.md` files.
- text messages go through media-link detection first; group text without media links goes into the group reply pipeline.
- photos and stickers are converted to textual context, then reuse the same group reply pipeline.
- private memory admin commands are restricted by chat type and `TELEGRAM_ADMIN_USER_IDS`.

The global error handler catches Telegram polling conflicts separately, which makes duplicate bot instances easier to diagnose.

### Text And Markdown Rendering

Text rendering is split into two concerns:

- [app/text2md.py](app/text2md.py) turns plain text into Markdown through the configured LLM.
- [app/md2jpg.py](app/md2jpg.py) converts Markdown to HTML and captures it as an image with Playwright.

Commands use the `,,,content,,,` wrapper so the bot can distinguish command syntax from the payload. Uploaded `.md` files bypass text-to-Markdown conversion; uploaded `.txt` files are converted first. Temporary output files are written under `output/` and cleaned up after sending or on errors.

### Media Download Flow

Text messages are inspected for supported media URLs by helpers in [app/main_helpers.py](app/main_helpers.py). YouTube and Bilibili links are handled by [app/youtube_dl.py](app/youtube_dl.py), which uses `yt-dlp`, picks a Telegram-friendly MP4 format, and calls ffmpeg compression when the file is too large.

Twitter/X links are handled by [app/twitter_downloader.py](app/twitter_downloader.py). The extractor supports direct tweet parsing plus fallback services. It can return photos, videos, GIFs, or text-only content. Because the extraction stack uses synchronous network and parsing work, [main.py](main.py) runs it through `asyncio.to_thread()` so the async Telegram event loop is not blocked.

Zhihu answer/response links, articles, posts/ideas, and bare questions are classified separately by [app/zhihu_dl.py](app/zhihu_dl.py). Answers use the answer API; other content uses its resource API with an HTML page fallback. Rich image references are returned separately from cleaned text, downloaded without API cookies, and sent as Telegram albums. This synchronous parser also runs through `asyncio.to_thread()`.

After media is sent, status messages and original link messages are deleted best-effort. Cleanup failures are logged but do not break the reply flow.

### Group Reply Pipeline

The core group reply path is `_handle_group_ai_reply_pipeline()` in [main.py](main.py). Text messages, photo summaries, and sticker descriptions all flow through this function.

The pipeline does the following:

1. Builds stable sender identity using the Telegram numeric user ID when available.
2. Extracts reply-chain metadata, including the parent Telegram message ID and parent display name.
3. Reads cached personal memory via `get_personal_memory_context()` without waiting for a refresh.
4. Schedules background personal memory refresh and candidate extraction.
5. Stores the user message in SQLite and returns the database message ID for evidence tracking.
6. Detects direct triggers: reply-to-bot, `@BotUsername`, `mioo`, or `小小宫`.
7. For ambient messages, asks [app/reply2message.py](app/reply2message.py) whether a reply is worthwhile.
8. Builds a richer RAG query from the current message, sender display, reply context, image summary, and sticker summary.
9. Fetches recent messages and related historical messages from [app/database.py](app/database.py).
10. Calls `generate_group_reply()` and stores the bot reply back into `messages`.

This design keeps reply latency low: expensive memory refreshes, candidate extraction, and sync media extraction are moved out of the main async path where possible.

The prompt layout is shaped for provider-side KV or prompt-cache reuse. A shared context-protocol system message and earlier history are emitted as append-only messages. On the next turn, the previous latest message is appended to that history rather than rewriting one monolithic prompt. Durable memory, RAG results, message-specific context, direct-address flags, and runtime state follow the history; the newest message remains the final message. Ambient reply probes use a shorter history and no RAG; when a busy chat's recent-history window rolls, a small oldest-message anchor preserves a stable prefix.

### SQLite Storage Model

[app/database.py](app/database.py) initializes and migrates the local SQLite database. The main tables are:

| Table | Purpose |
| --- | --- |
| `messages` | Chat messages, Telegram message IDs, reply-chain metadata, stable user keys |
| `message_embeddings` | One embedding row per message, including model/backend/signature metadata |
| `sticker_descriptions` | Cached natural-language sticker descriptions |
| `user_memories` | Per-user compact memory summary |
| `user_memory_facts` | Active/archived structured long-term facts with confidence and evidence IDs |
| `user_memory_candidates` | Pending/accepted/rejected memory candidates extracted from messages |

`add_message()` commits the message row before running embedding generation. That keeps slow embedding work from holding a SQLite write lock. Embedding insertion is best-effort: if embedding fails, the message still remains available for recent context and future memory refreshes.

### RAG Retrieval

RAG uses a hybrid retrieval strategy:

- recent context comes from chronological `messages` rows in the same chat;
- vector search uses FastEmbed vectors stored in `message_embeddings`;
- keyword search scans recent historical messages for lexical overlap;
- vector and keyword results are merged, deduplicated, and trimmed by context budgets.

The RAG query is not just the raw user message. [app/main_helpers.py](app/main_helpers.py) extracts keywords and augments the query with relevant message-specific context, including reply targets, image text, sticker descriptions, and sender display. This makes retrieval more likely to find the conversation thread the user is actually referring to.

The `miobot-rag` CLI exposes health and reindex commands. Health checks compare stored embedding signatures against the current runtime embedding signature; reindex rebuilds embeddings globally or for one chat.

### Personal Memory System

[app/user_memory.py](app/user_memory.py) manages long-term user memory. It intentionally separates memory into three layers:

1. `user_memory_candidates`: lightweight pending facts extracted from high-signal user messages.
2. `user_memory_facts`: durable structured facts used as the primary source of truth.
3. `user_memories.memory_text`: a compact summary used directly in prompts.

Candidate extraction is rule-based and runs in the background after a message is stored. Messages containing signals like `remember`, `from now on`, `我喜欢`, `以后`, `项目`, or `目标` can produce candidates. Explicit future preference language creates `fast` candidates; weaker durable signals create `slow` candidates.

Memory refresh consolidates pending candidates, existing facts, and historical messages through the configured LLM. The expected JSON output contains:

```json
{
  "memory_text": "short durable summary lines",
  "facts": [
    {
      "type": "preference",
      "text": "Prefers concise implementation plans",
      "confidence": 0.84,
      "evidence_message_ids": [123]
    }
  ],
  "archive_fact_ids": [12]
}
```

`archive_fact_ids` lets the model deactivate stale or contradictory facts instead of keeping conflicting memories active. Empty memories bootstrap from all previous messages for that user. Manual `/memory_refresh` forces regeneration from history.

### Memory Admin Tools

Private admin commands are implemented in [main.py](main.py). Access control checks both chat type and the configured admin list. The list accepts Telegram numeric IDs and `@usernames`.

Admin commands cover three kinds of operations:

- inspection: list users, view one user's memory, search memory text and facts, list pending candidates;
- regeneration: force a full memory refresh from history;
- manual curation: replace summary text, accept/reject candidates, edit active facts, archive active facts.

This gives the automatic memory system a review path, so generated memory can be corrected without editing the database by hand.

### Image And Sticker Understanding

[app/image2text.py](app/image2text.py) sends image payloads to Ark Responses API. File reads and base64 encoding are offloaded with `asyncio.to_thread()`. The response parser supports both `output_text` and nested Responses API output blocks.

Stickers use the same image understanding path when a visual file or thumbnail is available. Animated or video stickers fall back to their thumbnail if possible. If vision fails, the bot stores a simple fallback description based on sticker metadata, such as emoji, set name, or sticker type.

### Async And Reliability Choices

The bot is built on async `python-telegram-bot`, but not every dependency is async. The implementation avoids blocking the reply flow by:

- using `asyncio.to_thread()` for synchronous Twitter extraction and image file reads;
- committing SQLite messages before embedding work starts;
- tracking fallback background tasks when no Telegram `Application.create_task()` is available;
- treating cleanup operations as best-effort;
- keeping personal memory refresh out of the immediate reply path.

### Test Coverage

Tests live under [test](test). The non-live suite covers provider configuration, rendering logic, media extraction helpers, database migrations, RAG retrieval, group reply flow, image/sticker handling, memory refresh, memory candidates, and admin commands.

Live tests are intentionally separate because they need network access, real credentials, or external services:

```bash
uv run pytest test --ignore=test/test_llm_live.py --ignore=test/test_video_download_live.py
```

## Configuration

Runtime config is loaded from:

1. Existing process environment variables.
2. `config/runtime.env`.
3. `config/runtime.local.env`.
4. Built-in defaults from [app/runtime_config.py](app/runtime_config.py).

Existing values win. For local secrets, prefer `config/runtime.local.env` and avoid duplicating the same key in multiple files.

Start from [config/runtime.env.template](config/runtime.env.template):

```dotenv
# Telegram
TELEGRAM_BOT_USERNAME=MioooooooooBot
TELEGRAM_BOT_KEY=
TELEGRAM_ADMIN_USER_IDS=

# Provider selection: ark | zan | azure | ollama
LLM_PROVIDER=ark
LLM_ENABLE_THINKING=0

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-04-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini

# Ark text and vision
ARK_API_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/chat/completions
ARK_API_KEY=
ARK_MODEL=doubao-seed-1-8-251228
ARK_VISION_MODEL=doubao-seed-1-6-251015

# ZAN (OpenAI-compatible chat and vision)
ZAN_OPENAI_BASE_URL=https://ai.zan.top/v1
ZAN_API_KEY=
ZAN_MODEL=gpt-5.6-luna
# Optional; defaults to ZAN_MODEL.
ZAN_VISION_MODEL=

# Ollama
OLLAMA_ENDPOINT=http://100.69.97.8:11434
OLLAMA_MODEL=gpt-oss:20b

# Database and retrieval
DB_FILE=data/message_history.db
MESSAGE_REVIEW_BACK=80
RAG_TOP_K=12
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_REINDEX_ON_STARTUP=background

# Personal memory
MEMORY_CANDIDATE_EXTRACTION_ENABLED=1
MEMORY_CANDIDATE_AUTO_REFRESH_COUNT=3
```

Ark now uses a single endpoint variable:

- Configure only `ARK_API_ENDPOINT`.
- Text completions use the derived `/chat/completions` endpoint.
- Vision and sticker understanding use the derived `/responses` endpoint.

For `LLM_PROVIDER=zan`, chat, image, and sticker understanding use ZAN's
OpenAI-compatible `/chat/completions` endpoint. `ZAN_VISION_MODEL` defaults to
`ZAN_MODEL`. Set `RAG_REINDEX_ON_STARTUP` to `background` (default), `blocking`,
or `disabled`; the manual fallback is `miobot-rag reindex`.

Optional Twitter/X cookie configuration:

- `TWITTER_COOKIE`
- `TWITTER_COOKIE_FILE`
- Default cookie file: `config/x.com_cookies.txt`

## Admin Commands

Set `TELEGRAM_ADMIN_USER_IDS` to comma- or space-separated immutable Telegram numeric user IDs. Admin commands only work in private chat with the bot.

### Managing Personal Memory In Private Chat

1. Add yourself to `TELEGRAM_ADMIN_USER_IDS`, for example `TELEGRAM_ADMIN_USER_IDS=123456789`.
2. Restart the bot so the runtime config is reloaded.
3. Open a private Telegram chat with the bot. These commands are rejected in groups.
4. Send `/memory_help` to see the available memory commands.
5. Send `/memories` to list users that have message history, summary memory, or structured facts.
6. Copy a user key such as `tg_user:123456789`, then inspect it with `/memory tg_user:123456789`.
7. Use `/memory_refresh tg_user:123456789` to rebuild that user's memory from history when the summary looks stale or empty.
8. Use `/memory_set tg_user:123456789 <new summary>` to manually replace the compact summary.
9. Use `/memory_candidates tg_user:123456789` to review pending extracted facts, then `/memory_accept <candidate_id>` or `/memory_reject <candidate_id>`.
10. Use `/memory_fact_set <fact_id> <new text>` or `/memory_fact_delete <fact_id>` to edit or archive structured facts.

Typical private-chat flow:

```text
/memories
/memory tg_user:123456789
/memory_candidates tg_user:123456789
/memory_accept 7
/memory_refresh tg_user:123456789
```

The summary in `user_memories` is what gets injected into normal group-reply prompts. Structured facts in `user_memory_facts` are also shown in that personal memory context. Candidate commands only affect pending candidates; they do not edit the compact summary unless you accept candidates and then refresh or set the summary manually.

| Command | Purpose |
| --- | --- |
| `/memory_help` | Show memory admin commands |
| `/memories` | List users with message history, summaries, or facts |
| `/memory <user>` | View one user's summary and structured facts |
| `/memory_search <keyword>` | Search memory summaries and facts |
| `/memory_refresh <user>` | Regenerate one user's memory from history |
| `/memory_set <user> <text>` | Replace one user's summary memory |
| `/memory_candidates [user]` | List pending memory candidates |
| `/memory_accept <candidate_id>` | Accept a candidate into structured facts |
| `/memory_reject <candidate_id>` | Reject a candidate |
| `/memory_fact_set <fact_id> <text>` | Edit an active structured fact |
| `/memory_fact_delete <fact_id>` | Archive an active structured fact |

`<user>` accepts either a numeric Telegram user ID or `tg_user:<id>`.

## Installation

### Local Runtime

1. Install Python 3.11 or newer.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Install project dependencies:

```bash
uv sync
```

4. Install Playwright Chromium:

```bash
uv run playwright install chromium
```

5. Install common system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-noto-cjk
```

Optional dependencies:

```bash
# Azure provider
uv pip install openai

# /med2jpg support
sudo apt-get install -y texlive-xetex texlive-latex-extra texlive-pstricks texlive-lang-chinese
uv sync --extra med
```

## Running

```bash
uv run miobot
```

Equivalent:

```bash
uv run python main.py
```

On startup, MioBot initializes SQLite, validates the embedding backend, checks embedding metadata, and automatically reindexes when the stored embedding signature is stale.

## RAG Maintenance

```bash
uv run miobot-rag health
uv run miobot-rag reindex
uv run miobot-rag reindex --chat-id 123456
```

Use these commands to inspect embedding health or rebuild embeddings globally or per chat.

## Docker

Build:

```bash
docker build -t miobot:latest .
```

Run:

```bash
docker run --rm -it \
  --name miobot \
  -v "$PWD/config/runtime.local.env:/app/config/runtime.local.env:ro" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/output:/app/output" \
  miobot:latest
```

The Dockerfile includes core Python dependencies, ffmpeg, Noto CJK fonts, Playwright Chromium runtime dependencies, and `/med2jpg` runtime dependencies (XeLaTeX, Chinese TeX packages, barcode support, and the `med` Python extra). Azure support may still require extending the image.

## Testing

Run non-live tests:

```bash
uv run pytest test --ignore=test/test_llm_live.py --ignore=test/test_video_download_live.py
```

Live tests require real credentials or network access and are intentionally excluded from the default command above.

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Bot exits during startup | FastEmbed model/dependency issue | Check dependencies and allow the first model download |
| Ambient group messages rarely get replies | Activation probe decided not to reply | Reply to the bot or mention `mioo`, `小小宫`, or `@BotUsername` |
| Photos or stickers do not affect replies | Ark vision config missing | Set `ARK_API_KEY`, `ARK_API_ENDPOINT`, and `ARK_VISION_MODEL` |
| Twitter/X extraction fails | Protected content, login required, or rate limits | Provide `config/x.com_cookies.txt` or cookie env vars |
| `/med2jpg` fails | TeX/PDF dependencies missing | Install `texlive-xetex`, `texlive-latex-extra`, `texlive-pstricks`, `texlive-lang-chinese`, then run `uv sync --extra med` |
| Azure provider fails | Missing `openai` package or Azure settings | Install `openai` and complete Azure env vars |

## License

This project is licensed under the GNU GPLv3. See [LICENSE](LICENSE) for details.

---

# 中文说明

MioBot 是一个异步 Telegram 机器人，用于文本渲染、媒体下载，以及带有 RAG 上下文和用户长期记忆的群聊回复。

## 功能概览

- Markdown、纯文本、`.txt`、`.md` 转图片。
- 自动处理 YouTube、Bilibili、Twitter/X、知乎链接。
- 基于最近聊天、混合 RAG、个人记忆的群聊回复。
- 使用 Ark 视觉模型理解图片和贴纸。
- 使用 SQLite 保存消息、embedding、贴纸缓存、记忆摘要、结构化 facts、记忆候选。
- 管理员可在 Telegram 私聊里查看、刷新、编辑用户记忆。

## 主要能力

### 文本转图片

- `/md2jpg` 直接渲染 Markdown。
- `/text2jpg` 先调用当前 LLM 把纯文本整理成 Markdown，再渲染。
- 上传 `.txt` 或 `.md` 文件也会走同一套渲染流程。

示例：

```text
/md2jpg ,,,# Title
Some *markdown* here,,,

/text2jpg ,,,Some plain text here,,,
```

### 媒体下载

- 文本消息中出现 YouTube、Bilibili、Twitter/X、知乎链接时自动触发。
- YouTube / Bilibili 使用 [app/youtube_dl.py](app/youtube_dl.py)，优先下载不高于 720p 的 MP4，超过 Telegram 限制时尝试压缩。
- Twitter/X 使用 [app/twitter_downloader.py](app/twitter_downloader.py)，支持图片、视频、GIF 和纯文本兜底。
- 知乎链接由 [app/zhihu_dl.py](app/zhihu_dl.py) 区分回答/回复、专栏文章、想法/帖子和问题，分别抓取对应内容。
- 知乎正文中的图片链接会独立提取并发送为 Telegram 图片组；超大图片自动改为文件发送，下载图片时不会转发 API 登录 cookie。
- 媒体成功发送后会删除原始链接消息。

### 群聊回复

群聊文字、图片、贴纸最终都会进入 [main.py](main.py) 的统一回复流水线：

1. 写入消息和回复链元数据。
2. 读取缓存的个人记忆，并后台维护记忆。
3. 检测直接触发：回复机器人、提到 `@BotUsername`、`mioo` 或 `小小宫`。
4. 普通环境消息先经过 activation probe 判断要不要回复。
5. 组合最近消息、RAG 检索、个人记忆和运行时状态。
6. 生成回复并把机器人回复写回数据库。

### 多模态上下文

- 图片先由 [app/image2text.py](app/image2text.py) 生成文字/视觉摘要。
- 贴纸首次出现时生成一句描述并写入 `sticker_descriptions`，之后复用缓存。
- Ark 现在只需要一个 `ARK_API_ENDPOINT`；图片/贴纸所需的 `/responses` endpoint 会自动推导。

### 个人记忆

个人记忆分为三层：

- `user_memories`：用于 prompt 注入的压缩摘要。
- `user_memory_facts`：结构化长期 facts，带类型、置信度、证据消息和 active/archive 状态。
- `user_memory_candidates`：从高信号消息里后台抽取的 pending 候选。

明确“记住/以后/remember/from now on”的内容会进入 fast candidate；普通偏好、身份、项目、目标等会进入 slow candidate。fast candidate 或累计足够多的 slow candidate 会触发合并进 facts。模型也可以返回 `archive_fact_ids` 来归档冲突或过期事实。

## 实现细节

### 启动和运行时配置

启动入口在 [main.py](main.py)。程序会先调用 [app/runtime_config.py](app/runtime_config.py) 加载运行时配置，再导入依赖环境变量的模块。配置采用 first-value-wins：已有进程环境变量优先，然后加载 `config/runtime.env`、`config/runtime.local.env`，最后补内置默认值。

[app/ai_model.py](app/ai_model.py) 负责统一配置文本模型 provider。`LLM_PROVIDER` 可选 `ark`、`azure`、`ollama`。Ark 只暴露一个外部配置 `ARK_API_ENDPOINT`；运行时会自动推导 `/chat/completions` 和 `/responses`，避免文本和视觉 endpoint 分开维护。

开始 polling 前，启动流程会完成这些工作：

1. 检查 FastEmbed 和 embedding 模型是否可用。
2. 初始化或迁移 SQLite 表结构。
3. 检查数据库里已有 embedding 的模型签名。
4. 如果 embedding 签名漂移，自动 reindex。
5. 注册 Telegram handlers 和全局错误处理器。

### Telegram Handler 结构

[main.py](main.py) 是 Telegram 编排层。它注册命令、文件、文本、图片、贴纸和管理员私聊 handler：

- `/start` 返回简短能力说明。
- `/help` 返回当前所有功能和命令列表。
- `/md2jpg` 和 `/text2jpg` 共用命令解析和渲染路径。
- `/med2jpg` 调用处方 JSON 生成和图片渲染。
- `/crypto` 调用 [app/cryto.py](app/cryto.py) 返回价格和 APR 信息。
- 上传 `.txt` / `.md` 文件会进入文档渲染流程。
- 文本消息先检测媒体链接；没有媒体链接的群聊文本进入群聊回复流程。
- 群聊图片和贴纸会先变成文字上下文，再进入同一条回复流程。
- 记忆管理命令只允许配置过的管理员在私聊中使用。

全局错误处理器会单独识别 Telegram polling conflict，方便排查重复运行 bot 的情况。

### 文本和 Markdown 渲染

文本渲染拆成两层：

- [app/text2md.py](app/text2md.py)：通过当前 LLM 把纯文本整理成 Markdown。
- [app/md2jpg.py](app/md2jpg.py)：把 Markdown 转 HTML，再用 Playwright 截图成图片。

命令正文用 `,,,content,,,` 包起来，避免命令参数和正文混在一起。上传 `.md` 文件会直接渲染；上传 `.txt` 文件会先转 Markdown。临时文件写入 `output/`，发送完成或出错后都会清理。

### 媒体下载流程

[app/main_helpers.py](app/main_helpers.py) 负责识别文本里的媒体链接。YouTube 和 Bilibili 走 [app/youtube_dl.py](app/youtube_dl.py)，使用 `yt-dlp` 下载 Telegram 友好的 MP4，并在文件过大时尝试 ffmpeg 压缩。

Twitter/X 走 [app/twitter_downloader.py](app/twitter_downloader.py)，支持图片、视频、GIF 和纯文本兜底。因为这部分有同步网络和解析逻辑，[main.py](main.py) 使用 `asyncio.to_thread()` 执行，避免阻塞 async Telegram event loop。

知乎回答/回复、文章、想法/帖子和问题链接走 [app/zhihu_dl.py](app/zhihu_dl.py)，回答使用回答 API，其他类型使用对应资源 API 并在必要时回退到页面解析；正文图片会与清理后的文本分开提取、下载并发送。因为这部分也是同步网络请求，[main.py](main.py) 同样通过 `asyncio.to_thread()` 执行，避免阻塞 Telegram 的 async 事件循环。

媒体发送成功后，状态消息和原始链接消息会 best-effort 删除；删除失败只记录日志，不影响主流程。

### 群聊回复流水线

核心逻辑是 [main.py](main.py) 里的 `_handle_group_ai_reply_pipeline()`。群聊文本、图片摘要、贴纸描述都会进入这里。

流水线步骤：

1. 根据 Telegram numeric user ID 构造稳定用户 key。
2. 提取回复链元数据，包括父消息 Telegram ID 和父消息发送者。
3. 读取缓存个人记忆，不在主回复路径里等待刷新。
4. 后台调度个人记忆刷新和候选记忆抽取。
5. 把用户消息写入 SQLite，并拿到 DB message ID 作为 evidence。
6. 判断 direct trigger：回复 bot、提到 `@BotUsername`、`mioo`、`小小宫`。
7. 普通环境消息先调用 [app/reply2message.py](app/reply2message.py) 判断是否值得回复。
8. 用当前消息、发送者、回复关系、图片摘要、贴纸描述构造更丰富的 RAG query。
9. 从 [app/database.py](app/database.py) 读取最近上下文和相关历史消息。
10. 调用 `generate_group_reply()` 生成回复，并把 bot 回复写回 `messages`。

这条设计的重点是保持回复低延迟：记忆刷新、候选抽取、同步媒体解析都尽量放到主回复路径之外。

Prompt 结构专门照顾 provider 侧 KV cache / prompt cache。共享的上下文协议与稳定规则放在 system prompt，早期聊天历史以可追加的独立 message 发送；下一轮会把上一条最新消息追加到历史末尾，而不会从头重写一个大 prompt。长期个人记忆、RAG 结果、当前消息专属上下文、direct-address flags 与 runtime state 均位于历史之后，最新消息始终最后发送。普通环境消息的 Probe 使用更短的历史且不触发 RAG；高频群聊的历史窗口滚动时，会保留少量最早消息作为稳定锚点。

### SQLite 存储模型

[app/database.py](app/database.py) 初始化和迁移 SQLite。主要表包括：

| 表 | 用途 |
| --- | --- |
| `messages` | 群聊消息、Telegram message ID、回复链元数据、稳定用户 key |
| `message_embeddings` | 每条消息的 embedding，以及模型、backend、signature |
| `sticker_descriptions` | 贴纸自然语言描述缓存 |
| `user_memories` | 用户压缩记忆摘要 |
| `user_memory_facts` | 结构化长期 facts，带置信度、证据和 active/archive 状态 |
| `user_memory_candidates` | 从消息中抽取的 pending/accepted/rejected 记忆候选 |

`add_message()` 会先提交消息行，再生成 embedding。这样慢 embedding 不会持有 SQLite 写锁。embedding 写入是 best-effort；即使失败，消息仍然能用于最近上下文和未来记忆刷新。

### RAG 检索

RAG 是 hybrid retrieval：

- 最近上下文从同一 chat 的 `messages` 按时间读取。
- 向量检索使用 FastEmbed 和 `message_embeddings`。
- keyword 检索扫描历史消息做词面重合。
- 两路结果合并、去重，再按上下文预算裁剪。

RAG query 不只是当前文本。[app/main_helpers.py](app/main_helpers.py) 会加入回复目标、图片文字、贴纸描述、发送者显示名等上下文，让检索更容易命中当前对话真正指向的历史内容。

`miobot-rag` CLI 提供 `health` 和 `reindex`。`health` 用于检查 embedding 签名，`reindex` 用于全库或单 chat 重建 embedding。

### 个人记忆系统

[app/user_memory.py](app/user_memory.py) 维护长期用户记忆，分三层：

1. `user_memory_candidates`：从高信号消息中后台抽取的候选事实。
2. `user_memory_facts`：结构化、可归档的长期事实，是主要真实来源。
3. `user_memories.memory_text`：注入 prompt 的压缩摘要。

候选抽取是轻量规则式的。包含 `remember`、`from now on`、`我喜欢`、`以后`、`项目`、`目标` 等信号的消息可能生成 candidate。明确未来偏好会进入 `fast`，较弱但可能长期有用的信息进入 `slow`。

记忆刷新会把 pending candidates、已有 facts 和历史消息交给 LLM 合并。期望 JSON 结构是：

```json
{
  "memory_text": "short durable summary lines",
  "facts": [
    {
      "type": "preference",
      "text": "Prefers concise implementation plans",
      "confidence": 0.84,
      "evidence_message_ids": [123]
    }
  ],
  "archive_fact_ids": [12]
}
```

`archive_fact_ids` 用于停用冲突或过期 facts。空记忆会从该用户全部历史消息 bootstrap；管理员 `/memory_refresh` 会强制从历史重新生成。

### 记忆管理员工具

管理员工具在 [main.py](main.py) 中实现。权限检查同时校验私聊和 `TELEGRAM_ADMIN_USER_IDS`，只接受不会变更的 Telegram 数字 ID。

管理员能力分三类：

- 查看：列出用户、查看单人记忆、搜索 summary/facts、查看 pending candidates。
- 重新生成：从历史消息强制刷新某个用户记忆。
- 人工修正：替换 summary、接受/拒绝 candidate、编辑 active fact、归档 active fact。

这样自动记忆不是黑箱；管理员可以在 Telegram 内直接校正，不需要手动改数据库。

### 图片和贴纸理解

[app/image2text.py](app/image2text.py) 把图片发给 Ark Responses API。文件读取和 base64 编码通过 `asyncio.to_thread()` offload。响应解析支持 `output_text`，也支持 Responses API 的嵌套 output block。

贴纸复用同一套视觉理解。animated/video sticker 会优先使用缩略图；如果视觉理解失败，会根据 emoji、set name、sticker 类型生成 fallback 描述并缓存。

### Async 和可靠性设计

虽然项目基于 async `python-telegram-bot`，但部分依赖不是 async。当前实现通过这些方式避免阻塞：

- Twitter/X 同步提取放进 `asyncio.to_thread()`。
- 图片文件读取和 base64 编码放进 `asyncio.to_thread()`。
- SQLite 消息先提交，再跑 embedding。
- 没有 Telegram `Application.create_task()` 时，会跟踪 fallback background task。
- 清理失败只记录日志，不中断主流程。
- 个人记忆刷新不放在主回复路径里等待。

### 测试覆盖

测试位于 [test](test)。非 live 测试覆盖 provider 配置、渲染逻辑、媒体提取 helper、数据库迁移、RAG 检索、群聊回复、图片/贴纸理解、记忆刷新、记忆候选和管理员命令。

live 测试需要真实凭据、网络或外部服务，因此默认命令会排除：

```bash
uv run pytest test --ignore=test/test_llm_live.py --ignore=test/test_video_download_live.py
```

## 配置

运行时配置来源顺序：

1. 进程环境变量。
2. `config/runtime.env`。
3. `config/runtime.local.env`。
4. [app/runtime_config.py](app/runtime_config.py) 中的默认值。

已有值优先。建议把本地密钥放在 `config/runtime.local.env`，并避免同一个 key 在多个文件里重复配置。

常用配置如下，完整模板见 [config/runtime.env.template](config/runtime.env.template)：

```dotenv
# Telegram
TELEGRAM_BOT_USERNAME=MioooooooooBot
TELEGRAM_BOT_KEY=
TELEGRAM_ADMIN_USER_IDS=

# Provider: ark | zan | azure | ollama
LLM_PROVIDER=ark
LLM_ENABLE_THINKING=0

# Ark text and vision
ARK_API_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/chat/completions
ARK_API_KEY=
ARK_MODEL=doubao-seed-1-8-251228
ARK_VISION_MODEL=doubao-seed-1-6-251015

# ZAN（OpenAI 兼容聊天与视觉）
ZAN_OPENAI_BASE_URL=https://ai.zan.top/v1
ZAN_API_KEY=
ZAN_MODEL=gpt-5.6-luna
ZAN_VISION_MODEL=

# Database and retrieval
DB_FILE=data/message_history.db
MESSAGE_REVIEW_BACK=80
RAG_TOP_K=12
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RAG_REINDEX_ON_STARTUP=background
```

Ark endpoint 已统一：只配置 `ARK_API_ENDPOINT`。文本模型使用自动推导出的 `/chat/completions`，图片和贴纸使用自动推导出的 `/responses`。

使用 `LLM_PROVIDER=zan` 时，文本、图片和贴纸理解均通过 ZAN 的 OpenAI 兼容
`/chat/completions` 接口；未设置 `ZAN_VISION_MODEL` 时默认使用 `ZAN_MODEL`。
`RAG_REINDEX_ON_STARTUP` 可设为 `background`（默认）、`blocking` 或 `disabled`；
手动重建命令为 `miobot-rag reindex`。

## 管理员命令

`TELEGRAM_ADMIN_USER_IDS` 只支持 Telegram 数字用户 ID，逗号或空格分隔。管理员命令只在私聊 bot 时生效。

### 在私聊里管理个人记忆

1. 先把自己加入 `TELEGRAM_ADMIN_USER_IDS`，例如 `TELEGRAM_ADMIN_USER_IDS=123456789`。
2. 重启 bot，让运行时配置重新加载。
3. 在 Telegram 里打开和 bot 的私聊。下面这些命令在群聊里会被拒绝。
4. 发送 `/memory_help` 查看所有记忆管理命令。
5. 发送 `/memories` 列出目前有消息历史、summary 记忆或结构化 facts 的用户。
6. 复制用户 key，例如 `tg_user:123456789`，然后用 `/memory tg_user:123456789` 查看这个人的当前记忆。
7. 如果记忆为空、过期或明显不对，用 `/memory_refresh tg_user:123456789` 从历史消息重新生成。
8. 如果要直接覆盖 summary，用 `/memory_set tg_user:123456789 <新的摘要>`。
9. 用 `/memory_candidates tg_user:123456789` 查看待审核候选事实，再用 `/memory_accept <candidate_id>` 或 `/memory_reject <candidate_id>` 接受/拒绝。
10. 用 `/memory_fact_set <fact_id> <新内容>` 或 `/memory_fact_delete <fact_id>` 编辑/归档结构化 fact。

一个常见的私聊操作流程：

```text
/memories
/memory tg_user:123456789
/memory_candidates tg_user:123456789
/memory_accept 7
/memory_refresh tg_user:123456789
```

`user_memories` 里的 summary 会进入普通群聊回复的 personal memory prompt；`user_memory_facts` 里的结构化 facts 也会展示在 personal memory context 里。candidate 命令只处理 pending 候选；接受 candidate 会写入 facts，但如果想马上改 summary，可以再执行 `/memory_refresh` 或直接 `/memory_set`。

| 命令 | 作用 |
| --- | --- |
| `/memory_help` | 查看记忆管理命令 |
| `/memories` | 列出有消息历史、summary 或 facts 的用户 |
| `/memory <user>` | 查看某个用户的 summary 和 facts |
| `/memory_search <keyword>` | 搜索 summary 和 facts |
| `/memory_refresh <user>` | 从历史消息重新生成某个用户的记忆 |
| `/memory_set <user> <text>` | 替换某个用户的 summary 记忆 |
| `/memory_candidates [user]` | 查看 pending 记忆候选 |
| `/memory_accept <candidate_id>` | 接受候选并写入 facts |
| `/memory_reject <candidate_id>` | 拒绝候选 |
| `/memory_fact_set <fact_id> <text>` | 编辑 active fact |
| `/memory_fact_delete <fact_id>` | 归档 active fact |

`<user>` 可以是 Telegram 数字用户 ID，也可以是 `tg_user:<id>`。

## 安装与运行

1. 安装 Python 3.11+。
2. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
3. 安装依赖：

```bash
uv sync
uv run playwright install chromium
```

常见系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-noto-cjk
```

启动：

```bash
uv run miobot
```

等价方式：

```bash
uv run python main.py
```

## RAG 维护

```bash
uv run miobot-rag health
uv run miobot-rag reindex
uv run miobot-rag reindex --chat-id 123456
```

用于查看 embedding 健康状态，或全量/按 chat 重建 embedding。

## Docker

```bash
docker build -t miobot:latest .
docker run --rm -it \
  --name miobot \
  -v "$PWD/config/runtime.local.env:/app/config/runtime.local.env:ro" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/output:/app/output" \
  miobot:latest
```

Dockerfile 已包含核心 Python 依赖、ffmpeg、Noto CJK 字体、Playwright Chromium 运行依赖，以及 `/med2jpg` 运行所需的 XeLaTeX、中文 TeX 包、barcode 支持和 `med` Python extra。Azure 支持仍然可能需要自行扩展镜像。

## 测试

```bash
uv run pytest test --ignore=test/test_llm_live.py --ignore=test/test_video_download_live.py
```

live 测试需要真实凭据或网络访问，默认不包含在上面的命令里。

## 常见问题

| 现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| 启动时 FastEmbed 报错 | 模型或依赖不可用 | 检查依赖，并允许首次模型下载 |
| 群聊普通消息很少回复 | activation probe 判断不需要回复 | 直接回复 bot 或提到 `mioo` / `小小宫` / `@BotUsername` |
| 图片或贴纸不影响回复 | Ark vision 配置不完整 | 配置 `ARK_API_KEY`、`ARK_API_ENDPOINT`、`ARK_VISION_MODEL` |
| Twitter/X 提取失败 | 受保护内容、需要登录态或限流 | 提供 `config/x.com_cookies.txt` 或 cookie 环境变量 |
| `/med2jpg` 失败 | 缺少 TeX/PDF 依赖 | 安装 `texlive-xetex`、`texlive-latex-extra`、`texlive-pstricks`、`texlive-lang-chinese`，然后执行 `uv sync --extra med` |

## License

本项目使用 GNU GPLv3。详见 [LICENSE](LICENSE)。
