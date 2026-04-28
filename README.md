# MioBot

MioBot 是一个异步 Telegram 机器人，当前主要围绕三条能力线工作：

- Markdown / 纯文本 / `.txt` / `.md` 转图片
- YouTube、Bilibili、Twitter/X 链接自动下载与回传
- 群聊里的轻量多模态回复，底层使用 SQLite + FastEmbed RAG + 用户长期记忆

English summary: MioBot is an async Telegram bot for text-to-image rendering, media downloading, and context-aware group replies backed by SQLite, FastEmbed, and configurable LLM providers.

## Current Features

### 1. Markdown / Text -> Image

- `/md2jpg` 直接把 Markdown 渲染成图片。
- `/text2jpg` 先调用当前配置的 LLM 把纯文本整理成 Markdown，再渲染成图片。
- 上传 `.txt` 或 `.md` 文件也会走同一套渲染链路。
- 当前 Telegram handler 默认输出 JPG 文件，并使用 `formal_code` 主题。

示例：

```text
/md2jpg ,,,# Title
Some *markdown* here,,,

/text2jpg ,,,Some plain text here,,,
```

对应模块：

- [`app.text2md.plain_text_to_markdown`](app/text2md.py)
- [`app.md2jpg.md_to_image`](app/md2jpg.py)

### 2. Media Download

- 文本消息里出现 YouTube、Bilibili、Twitter/X 链接时会自动触发下载流程，不需要额外命令。
- YouTube / Bilibili 走 [`app.youtube_dl`](app/youtube_dl.py)，优先下载不高于 720p 的 MP4，并在超过 Telegram 50MB 限制时尝试用 ffmpeg 压缩。
- Twitter/X 走 [`app.twitter_downloader`](app/twitter_downloader.py)，支持图片、视频、GIF，以及只回文字内容的兜底路径。
- 媒体成功发送后，机器人会删除原始链接消息。

### 3. Contextual Group Replies

群聊回复的当前实现已经不是旧版 README 里的“随机 1/5 触发”。现在的流程是：

- 每条进入回复流水线的群消息先写入 SQLite。
- 直接触发条件包括：回复机器人、提到 `@BotUsername`、提到 `mioo`、提到 `小小宫`。
- 直接触发时跳过激活探测，直接进入 RAG 检索和回复生成。
- 普通环境消息会先经过 [`app.reply2message.should_activate_reply`](app/reply2message.py) 做一次“要不要回复”的判断。
- 真正生成回复时使用：
  - 最近聊天窗口，默认 `MESSAGE_REVIEW_BACK=80`
  - 同 chat 全量 embedding 索引上的相似消息检索，默认 `RAG_TOP_K=12`
  - 用户长期记忆摘要
  - 回复链元数据，例如“当前消息回复了谁、回复了什么”

相关模块：

- [`main.py`](main.py)
- [`app.reply2message`](app/reply2message.py)
- [`app.database`](app/database.py)
- [`app.user_memory`](app/user_memory.py)
- [`app.rag_embeddings`](app/rag_embeddings.py)

### 4. Multimodal Group Context

- 群聊图片会先走 [`app.image2text.image_to_text`](app/image2text.py)，提取文字和视觉摘要后再进入同一条群聊回复流水线。
- 群聊贴纸会先尝试生成一句自然语言描述；首次见到的贴纸会写入 `sticker_descriptions` 缓存，后续直接复用。
- 图片和贴纸理解当前使用 Ark Responses API，不走 `LLM_PROVIDER` 抽象层。

这意味着：

- 文字生成可以选 `ark` / `azure` / `ollama`
- 但如果你想让“群聊图片/贴纸理解”生效，仍然需要配置 `ARK_API_KEY`

### 5. `/med2jpg`

- `/med2jpg` 会先把自然语言需求转换成结构化处方 JSON，再生成 PDF，最后转成 JPG。
- 这一功能依赖 LaTeX 和 `pypdfium2`，不属于最小运行时的一部分。

相关模块：

- [`app.med.generate_med`](app/med.py)
- [`app.med.generate_jpg_from_med_json`](app/med.py)

### 6. `/crypto`

- `/crypto` 会拉取 Coinbase 价格信息，以及 Allez SOL / Allez USDC APR 信息并直接回复。
- 相关实现位于 [`app.cryto`](app/cryto.py)。

## Architecture Overview

| Concern | Main Files | Notes |
| --- | --- | --- |
| Entrypoint / handler wiring | [main.py](main.py) | 启动、注册 Telegram handlers、启动前健康检查 |
| Runtime config | [app/runtime_config.py](app/runtime_config.py) | 读取 `config/runtime.env`、`config/runtime.local.env` 和默认值 |
| LLM provider abstraction | [app/ai_model.py](app/ai_model.py) | 统一封装 Ark / Azure / Ollama |
| Markdown rendering | [app/text2md.py](app/text2md.py), [app/md2jpg.py](app/md2jpg.py) | 文本整理、HTML 渲染、Playwright 截图 |
| Media download | [app/youtube_dl.py](app/youtube_dl.py), [app/twitter_downloader.py](app/twitter_downloader.py) | 下载、压缩、标题和 caption 处理 |
| Group reply logic | [app/reply2message.py](app/reply2message.py), [main.py](main.py) | 激活探测、回复生成、direct trigger |
| SQLite + RAG | [app/database.py](app/database.py), [app/rag_embeddings.py](app/rag_embeddings.py) | 消息存储、embedding、向量检索、健康检查与重建 |
| Personal memory | [app/user_memory.py](app/user_memory.py) | 每个 Telegram 用户按 UTC 天刷新长期记忆 |
| Vision / sticker understanding | [app/image2text.py](app/image2text.py) | 图片 OCR/摘要、贴纸一句话描述 |
| Shared helpers | [app/main_helpers.py](app/main_helpers.py) | URL regex、触发判断、reply relation 构造 |
| Prescription rendering | [app/med.py](app/med.py) | 处方 JSON -> PDF -> JPG |

## Runtime Flow

### Startup

`main.py` 的启动顺序大致如下：

1. 通过 [`app.runtime_config.bootstrap_runtime_environment`](app/runtime_config.py) 加载运行时配置。
2. 通过 [`app.ai_model.configure_llm`](app/ai_model.py) 选择文本 LLM provider。
3. 调用 [`app.rag_embeddings.ensure_fastembed_ready`](app/rag_embeddings.py) 做 embedding fail-fast 检查。
4. 初始化 SQLite 表结构。
5. 执行 embedding 健康检查；如果检测到旧 embedding 或签名漂移，会自动 reindex。
6. 注册 Telegram command / message handlers 并开始 polling。

### Group Reply Pipeline

群聊文字、图片、贴纸最终都会尽量汇合到同一个 `_handle_group_ai_reply_pipeline`：

1. 把消息和回复链元数据写入 `messages`
2. 刷新或读取该用户的长期记忆
3. 判断是否是 direct trigger；不是的话先做 activation probe
4. 组装 prompt context：最近聊天窗口 + RAG 检索结果 + 用户记忆 + 运行时状态
5. 生成回复并把机器人的回复再次写回 `messages`

### Storage Model

当前数据库包含至少以下几类数据：

- `messages`: 原始群聊消息与 reply-chain 元数据
- `message_embeddings`: 每条消息的 embedding、模型签名、后端信息
- `sticker_descriptions`: 贴纸描述缓存
- `user_memories`: 用户长期记忆摘要

注意：当前实现不会把历史消息裁剪到“最近 100 条”。数据库会持续积累消息；真正送给模型的“最近窗口”大小由 `MESSAGE_REVIEW_BACK` 控制，默认值是 80。

## Configuration

### Persona / Background Knowledge

编辑 [`config/info.txt`](config/info.txt)，每行写一个稳定事实或设定，群聊回复时会被注入系统提示里。

### Runtime Environment Files

运行时配置由 [`app/runtime_config.py`](app/runtime_config.py) 加载：

- `config/runtime.env`
- `config/runtime.local.env`
- 内置默认值

当前实现使用“first value wins”语义：

- 进程环境变量如果已经存在，不会被文件覆盖
- 在文件里，较早加载到的 key 不会被较晚的文件覆盖

因此更安全的实践是：

- 把一个 key 只放在一个地方
- 如果使用 `runtime.local.env`，尽量不要在 `runtime.env` 里重复同名 key

### Key Environment Variables

最常用的变量如下，完整模板见 [`config/runtime.env.template`](config/runtime.env.template)：

```dotenv
# Telegram
TELEGRAM_BOT_USERNAME=MioooooooooBot
TELEGRAM_BOT_KEY=

# Provider selection: ark | azure | ollama
LLM_PROVIDER=ark
LLM_ENABLE_THINKING=0

# Azure text model
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=2024-04-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-5-mini

# Ark text + vision
ARK_API_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/chat/completions
ARK_API_KEY=
ARK_MODEL=doubao-seed-1-8-251228
ARK_RESPONSES_ENDPOINT=https://ark.cn-beijing.volces.com/api/v3/responses
ARK_VISION_MODEL=doubao-seed-1-6-251015

# Ollama
OLLAMA_ENDPOINT=http://100.69.97.8:11434
OLLAMA_MODEL=gpt-oss:20b

# Database / retrieval
DB_FILE=data/message_history.db
MESSAGE_REVIEW_BACK=80
RAG_TOP_K=12
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

额外可选项：

- `TWITTER_COOKIE` 或 `TWITTER_COOKIE_FILE`
- 默认 cookie 文件位置是 `config/x.com_cookies.txt`

## Installation

### Core Runtime

1. 安装 Python 3.11+。
2. 安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)。
3. 安装项目依赖：

```bash
uv sync
```

4. 安装 Playwright Chromium：

```bash
uv run playwright install chromium
```

5. 安装系统依赖。

Debian / Ubuntu 常见最小组合：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg fonts-noto-cjk
```

### Optional Dependencies

如果你需要以下功能，还要额外装依赖：

- 使用 Azure provider:

```bash
uv pip install openai
```

- 使用 `/med2jpg`:

```bash
sudo apt-get install -y texlive-xetex texlive-latex-extra texlive-lang-chinese
uv pip install pypdfium2
```

- 需要 Twitter/X 受保护内容、登录态内容或更稳定提取时：
  - 导出浏览器 cookies 到 `config/x.com_cookies.txt`
  - 或设置 `TWITTER_COOKIE` / `TWITTER_COOKIE_FILE`

仓库里的 [`init.sh`](init.sh) 提供了一个偏 Debian/Ubuntu 的本地启动脚本，可作为参考。

## Running

推荐方式：

```bash
uv run miobot
```

等价方式：

```bash
uv run python main.py
```

首次运行时会：

- 初始化 `DB_FILE` 指向的 SQLite 文件，默认是 `data/message_history.db`
- 检查 FastEmbed 是否可用
- 检查 embedding 健康状态，并在需要时自动执行 reindex

## RAG Maintenance CLI

项目还暴露了一个维护脚本入口：

```bash
uv run miobot-rag health
uv run miobot-rag reindex
uv run miobot-rag reindex --chat-id 123456
```

用途：

- 查看当前 embedding 签名、数据库里已有的 embedding profile，以及是否需要重建
- 手动对整个库或单个 chat 重新生成 embedding

## Docker

构建：

```bash
docker build -t miobot:latest .
```

运行：

```bash
docker run --rm -it \
  --name miobot \
  -v "$PWD/config/runtime.local.env:/app/config/runtime.local.env:ro" \
  -v "$PWD/data:/app/data" \
  -v "$PWD/output:/app/output" \
  miobot:latest
```

当前 Dockerfile 已包含：

- 项目核心 Python 依赖
- ffmpeg
- Noto CJK 字体
- Playwright Chromium 运行时

当前 Dockerfile 未包含：

- `openai` Python 包，所以如果要在容器里使用 Azure provider，需要扩展镜像
- `pypdfium2` 和 TeX Live，所以 `/med2jpg` 默认不可用，需要扩展镜像

如果使用 `LLM_PROVIDER=ollama`，还需要确保容器能访问 `OLLAMA_ENDPOINT`。

## Command Summary

| Action | Trigger |
| --- | --- |
| Start | `/start` |
| Markdown -> image | `/md2jpg ,,,...markdown...,,,` |
| Plain text -> image | `/text2jpg ,,,...plain text...,,,` |
| Prescription -> image | `/med2jpg ...` |
| Crypto snapshot | `/crypto` |
| File -> image | 上传 `.txt` 或 `.md` |
| Media download | 直接发送 YouTube / Bilibili / Twitter/X 链接 |
| Group reply on text | 群聊文本，满足 direct trigger 或 activation probe |
| Group reply on photo | 群聊图片 |
| Group reply on sticker | 群聊贴纸 |

补充说明：

- `/md2jpg` 和 `/text2jpg` 仍然要求正文用前后 `,,,` 包起来。
- 纯文本私聊不会自动触发聊天回复；没有命令时，文本 handler 主要做“媒体链接检测”或“群聊回复”。

## Troubleshooting

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| Bot exits at startup with fastembed error | `fastembed` 模型或依赖不可用 | 确认当前环境已安装依赖，并允许首次模型下载 |
| Ambient group messages rarely get a reply | 这是当前设计，不再是随机 1/5；需要 activation probe 判定为值得回复 | 直接回复机器人或提到 `mioo` / `小小宫` / `@BotUsername` |
| Photos or stickers do not influence replies | 没有配置 Ark vision 相关变量 | 配置 `ARK_API_KEY`、`ARK_RESPONSES_ENDPOINT`、`ARK_VISION_MODEL` |
| Twitter/X extraction fails on some posts | 受保护内容、需要登录态、或被站点限流 | 提供 `config/x.com_cookies.txt` 或设置 cookie 变量 |
| `/med2jpg` fails with LaTeX or PDF errors | 缺少 `xelatex`、中文 LaTeX 支持或 `pypdfium2` | 安装 TeX Live 中文支持和 `pypdfium2` |
| Azure provider fails | 当前环境缺少 `openai` 包或 Azure 配置不完整 | 安装 `openai`，并补齐 Azure 相关环境变量 |

## License

This project is licensed under the GNU GPLv3. See [LICENSE](LICENSE) for details.
