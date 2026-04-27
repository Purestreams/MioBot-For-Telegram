"""Telegram bot entrypoint and handler orchestration."""

# general imports
import datetime
import io
import logging
import os
import random
import time
from typing import Optional

from telegram import InputMediaPhoto, Update
from telegram.constants import ParseMode
from telegram.error import Conflict
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


# private imports
from app.runtime_config import bootstrap_runtime_environment, get_runtime_value

# Load config/runtime env before importing modules that read env at import time.
bootstrap_runtime_environment()

from app.md2jpg import md_to_image
from app.text2md import plain_text_to_markdown
from app.twitter_downloader import (
    TwitterDownloader,
    build_twitter_caption,
    format_tweet_text_for_reply,
    is_twitter_status_url,
    summarize_tweet_text,
)
from app.youtube_dl import (
    download_video_to_file,
    compress_video_if_needed,
    resolve_caption_url,
)
from app.reply2message import should_reply_and_generate
from app.database import init_db, add_message, get_prompt_context_parts
from app.image2text import image_to_text

from app.cryto import get_Allez_APR, get_Allez_USDC_APR, get_Price_Coinbase

from app.med import generate_jpg_from_med_json, generate_med
from app.ai_model import configure_llm
from app.main_helpers import (
    OUTPUT_DIR,
    MD2JPG_REGEX,
    TEXT2JPG_REGEX,
    _build_output_path,
    _remove_file_if_exists,
    _delete_message_if_exists,
    _extract_video_url,
    _is_reply_to_this_bot,
    _display_name_from_user,
    _build_reply_relation_payload,
    _match_command_payload,
    _build_rag_query_from_message,
    _is_group_chat,
    _extract_search_keywords,
)

AZURE_OPENAI_ENDPOINT = get_runtime_value("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = get_runtime_value("AZURE_OPENAI_API_KEY")
TELEGRAM_BOT_USERNAME = get_runtime_value("TELEGRAM_BOT_USERNAME")
TELEGRAM_BOT_KEY = get_runtime_value("TELEGRAM_BOT_KEY")
ARK_ENDPOINT = get_runtime_value("ARK_API_ENDPOINT")
ARK_API_KEY = get_runtime_value("ARK_API_KEY")

AZURE_OPENAI_API_VERSION = get_runtime_value("AZURE_OPENAI_API_VERSION")

# Models: Phi-4-mini-instruct, Phi-4 or gpt-4.1-nano
AZURE_OPENAI_DEPLOYMENT_NAME = get_runtime_value("AZURE_OPENAI_DEPLOYMENT_NAME")

ARK_API_KEY = get_runtime_value("ARK_API_KEY")
ARK_MODEL = get_runtime_value("ARK_MODEL")
LLM_PROVIDER = get_runtime_value("LLM_PROVIDER")
if LLM_PROVIDER:
    normalized_provider = LLM_PROVIDER.strip().lower()
    if normalized_provider in {"azure_openai", "azure-openai", "azureopenai"}:
        LLM_PROVIDER = "azure"
    else:
        LLM_PROVIDER = normalized_provider


def _warn_missing_runtime_env(provider: str) -> None:
    required = ["TELEGRAM_BOT_KEY"]
    if provider == "azure":
        required += ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_DEPLOYMENT_NAME"]
    elif provider == "ark":
        required += ["ARK_API_KEY", "ARK_API_ENDPOINT", "ARK_MODEL"]
    elif provider == "ollama":
        required += ["OLLAMA_ENDPOINT", "OLLAMA_MODEL"]

    missing = [name for name in required if not get_runtime_value(name)]
    if missing:
        logger.warning(
            "Missing runtime env values at startup: %s. Check config/runtime.env or config/runtime.local.env.",
            ", ".join(missing),
        )


_warn_missing_runtime_env(LLM_PROVIDER or "ark")

configure_llm(
    provider=LLM_PROVIDER,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_api_key=AZURE_OPENAI_API_KEY,
    azure_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    ark_api_key=ARK_API_KEY,
    ark_model=ARK_MODEL,
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)
TELEGRAM_CAPTION_LIMIT = 1024


def _truncate_caption_text(text: str, max_chars: int = TELEGRAM_CAPTION_LIMIT) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


async def _handle_twitter_media_message(
    *,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    video_url: str,
    sender_display: str,
    status_message,
) -> bool:
    """Handle Twitter/X media. Returns True when request is fully handled."""
    if not update.message or not update.effective_chat:
        return False

    twitter_downloader = TwitterDownloader()
    media_list, text_dict = twitter_downloader.extract_twitter_media(video_url)

    tweet_text = summarize_tweet_text(text_dict)

    image_medias = [media for media_type, media in media_list if media_type == 'pic']
    video_medias = [media for media_type, media in media_list if media_type == 'vid']
    gif_medias = [media for media_type, media in media_list if media_type == 'gif']
    raw_text_caption = (tweet_text[:900] + "...") if len(tweet_text) > 900 else tweet_text
    text_caption = format_tweet_text_for_reply(raw_text_caption, video_url)

    if not image_medias and not video_medias and not gif_medias and not text_caption:
        raise ValueError(
            "Could not extract video, images, or text from this tweet. "
            "It may be deleted, protected, region-restricted, or blocked by auth/cookie settings."
        )

    if len(image_medias) > 1:
        album_caption = build_twitter_caption(text_caption, sender_display, video_url)
        media_group = []
        for index, image_bytes in enumerate(image_medias, start=1):
            image_buffer = io.BytesIO(image_bytes)
            image_buffer.name = f"tweet_image_{index}.jpg"
            is_first_image = index == 1
            media_group.append(
                InputMediaPhoto(
                    media=image_buffer,
                    caption=album_caption if is_first_image else None,
                    parse_mode=ParseMode.HTML if is_first_image else None,
                )
            )
        await context.bot.send_media_group(
            chat_id=update.effective_chat.id,
            media=media_group,
            ##reply_to_message_id=update.message.message_id,
        )
    elif len(image_medias) == 1:
        image_caption = build_twitter_caption(text_caption, sender_display, video_url)
        image_buffer = io.BytesIO(image_medias[0])
        image_buffer.name = "tweet_image_1.jpg"
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=image_buffer,
            ##reply_to_message_id=update.message.message_id,
            caption=image_caption,
            parse_mode=ParseMode.HTML,
        )

    if video_medias and not image_medias:
        for index, video_bytes in enumerate(video_medias, start=1):
            video_buffer = io.BytesIO(video_bytes)
            video_buffer.name = f"tweet_video_{index}.mp4"
            is_first_video = index == 1
            video_caption = build_twitter_caption(text_caption, sender_display, video_url) if is_first_video else None
            await context.bot.send_video(
                chat_id=update.effective_chat.id,
                video=video_buffer,
                ##reply_to_message_id=update.message.message_id if is_first_video else None,
                caption=video_caption,
                parse_mode=ParseMode.HTML if is_first_video else None,
            )

    for index, gif_bytes in enumerate(gif_medias, start=1):
        gif_buffer = io.BytesIO(gif_bytes)
        gif_buffer.name = f"tweet_gif_{index}.mp4"
        is_primary_document = not image_medias and not video_medias and index == 1
        gif_caption = build_twitter_caption(text_caption, sender_display, video_url) if is_primary_document else None
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=gif_buffer,
            ##reply_to_message_id=update.message.message_id if is_primary_document else None,
            caption=gif_caption,
            parse_mode=ParseMode.HTML if gif_caption else None,
        )

    if text_caption and not image_medias and not video_medias and not gif_medias:
        text_body = build_twitter_caption(text_caption, sender_display, video_url)
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=text_body,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )

    await _delete_message_if_exists(status_message)
    await update.message.delete()
    return True


async def _render_and_send_image_from_markdown(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    markdown_input: str,
    output_file_path: str,
) -> None:
    if not update.message or not update.effective_chat:
        return

    await md_to_image(md_text=markdown_input, output_path=output_file_path, theme='formal_code')
    with open(output_file_path, 'rb') as photo:
        await context.bot.send_document(
            chat_id=update.effective_chat.id,
            document=photo,
            reply_to_message_id=update.message.message_id,
        )


async def _handle_md2jpg_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    markdown_input: str,
) -> None:
    if not update.message:
        return

    if not markdown_input:
        await update.message.reply_text("Please provide some markdown content inside the triple quotes.")
        return

    output_file_path = _build_output_path("md", update.message.message_id)
    status_message = None
    try:
        status_message = await update.message.reply_text("Generating your image, please wait a moment...")
        await _render_and_send_image_from_markdown(update, context, markdown_input, output_file_path)
        await _delete_message_if_exists(status_message)
    except Exception as e:
        logger.error(f"Error during image generation or sending: {e}")
        await update.message.reply_text("Sorry, I encountered an error while creating your image.")
        await _delete_message_if_exists(status_message)
    finally:
        _remove_file_if_exists(output_file_path)


async def _handle_text2jpg_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    plain_text_input: str,
) -> None:
    if not update.message:
        return

    if not plain_text_input:
        await update.message.reply_text("Please provide some text content inside the triple quotes.")
        return

    output_file_path = _build_output_path("text", update.message.message_id)
    status_message = None
    try:
        status_message = await update.message.reply_text("Converting your text to markdown, please wait a moment...")
        generated_markdown = await plain_text_to_markdown(plain_text_input)
        await status_message.edit_text("Generating your image from markdown, please wait a moment...")
        await _render_and_send_image_from_markdown(update, context, generated_markdown, output_file_path)
        await _delete_message_if_exists(status_message)
    except Exception as e:
        logger.error(f"Error during image generation or sending: {e}")
        await update.message.reply_text("Sorry, I encountered an error while creating your image.")
        await _delete_message_if_exists(status_message)
    finally:
        _remove_file_if_exists(output_file_path)


# -------- Telegram Bot Handlers --------

# Start command handler
# This handler sends a welcome message when the /start command is issued.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    if not update.message:
        return

    await update.message.reply_text(
        """Hi! I can convert Markdown to an image. Send me a message like:\n\n /md2jpg ,,,Your markdown here,,, \n\n'or\n\n /text2jpg ,,,Your plain text here,,, \n\nI can also download YouTube videos if you send me a link, and I might reply to messages in this group if I find them interesting, nya~"""
    )


async def handle_md2jpg_and_text2jpg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /md2jpg and /text2jpg commands to generate images."""
    if not update.message or not update.message.text:
        return
    logger.info(f"Received text for rendering: {update.message.text if update.message else 'No message text'}")
    message_text = update.message.text

    markdown_input = _match_command_payload(message_text, MD2JPG_REGEX)
    if markdown_input is not None:
        await _handle_md2jpg_request(update, context, markdown_input)

    plain_text_input = _match_command_payload(message_text, TEXT2JPG_REGEX)
    if plain_text_input is not None:
        await _handle_text2jpg_request(update, context, plain_text_input)


# Handle .txt or .md files to render as image
async def handle_text_or_markdown_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle .txt or .md files to render as image."""
    if not update.message or not update.message.document:
        return

    document_file = update.message.document
    file_name = document_file.file_name
    if not file_name:
        return

    is_already_markdown = file_name.endswith('.md')

    if file_name.endswith(('.txt', '.md')):
        tg_file = await document_file.get_file()
        downloaded_path = await tg_file.download_to_drive(
            custom_path=os.path.join(OUTPUT_DIR, file_name)
        )

        with open(downloaded_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        output_file_path = _build_output_path("file", update.message.message_id)

        status_message = None
        try:
            status_message = await update.message.reply_text("Converting your file to markdown, please wait a moment...")

            if not is_already_markdown:
                generated_markdown = await plain_text_to_markdown(file_content)
            else:
                generated_markdown = file_content

            await status_message.edit_text("Generating your image from markdown, please wait a moment...")

            await _render_and_send_image_from_markdown(update, context, generated_markdown, output_file_path)
            await _delete_message_if_exists(status_message)
        except Exception as e:
            logger.error(f"Error during image generation or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while creating your image.")
            await _delete_message_if_exists(status_message)
        finally:
            _remove_file_if_exists(output_file_path)
            _remove_file_if_exists(downloaded_path)


# Handle Group AI Replies
async def handle_group_ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle group messages and occasionally reply using AI."""
    if not update.message or not update.message.text:
        return

    await _handle_group_ai_reply_pipeline(update, update.message.text)


async def _handle_group_ai_reply_pipeline(
    update: Update,
    message_text: str,
    *,
    additional_context: Optional[list[str]] = None,
) -> None:
    """Shared group-reply flow for text-like content."""
    if not update.message:
        return

    if not update.effective_chat or not update.effective_user:
        return

    chat_id = update.effective_chat.id
    sender_display = _display_name_from_user(update.effective_user)
    stored_message_text, relation_context = _build_reply_relation_payload(update, message_text)
    merged_additional_context = list(additional_context or []) + relation_context
    replied_message = update.message.reply_to_message
    reply_to_tg_id = getattr(replied_message, "message_id", None) if replied_message else None
    reply_to_username = _display_name_from_user(getattr(replied_message, "from_user", None)) if replied_message else None

    logger.info("Adding message to history for %s", sender_display)
    await add_message(
        chat_id=chat_id,
        username=sender_display,
        content=stored_message_text,
        telegram_message_id=getattr(update.message, "message_id", None),
        reply_to_telegram_message_id=reply_to_tg_id,
        reply_to_username=reply_to_username,
    )

    is_reply_to_bot = _is_reply_to_this_bot(update, TELEGRAM_BOT_USERNAME)
    if is_reply_to_bot:
        logger.info("User %s replied to the bot.", sender_display)

    # 1 in 5 chance to consider replying, unless it's a reply to the bot.
    if not is_reply_to_bot and random.randint(1, 5) != 1:
        return

    rag_query = _build_rag_query_from_message(message_text)
    history_messages, rag_related_messages = await get_prompt_context_parts(chat_id, query=rag_query)

    ai_reply = await should_reply_and_generate(
        message_history=history_messages,
        rag_related_messages=rag_related_messages,
        additional_context=merged_additional_context or None,
        is_reply_to_bot=is_reply_to_bot,
    )

    if ai_reply:
        try:
            sent_message = await update.message.reply_text(ai_reply)
            await add_message(
                chat_id=chat_id,
                username="mioo_bot",
                content=ai_reply,
                telegram_message_id=getattr(sent_message, "message_id", None),
                reply_to_telegram_message_id=getattr(update.message, "message_id", None),
                reply_to_username=sender_display,
            )
        except Exception as e:
            logger.error(f"Error sending AI reply: {e}")

# Handle text messages: download video links (YouTube/Bilibili/Twitter), else pass to group AI handler
async def handle_text_for_youtube_or_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages: download supported video links, else pass to group AI handler."""
    if not update.message or not update.message.text:
        return

    if not update.effective_chat or not update.effective_user:
        return

    sender_display = _display_name_from_user(update.effective_user)
    message_text = update.message.text.strip()
    video_url = _extract_video_url(message_text)

    if video_url:
        status_message = None
        output_file_path = ""
        cleanup_paths: set[str] = set()
        try:
            status_message = await update.message.reply_text("Downloading your video, please wait a moment...")

            if is_twitter_status_url(video_url):
                await _handle_twitter_media_message(
                    update=update,
                    context=context,
                    video_url=video_url,
                    sender_display=sender_display,
                    status_message=status_message,
                )
                return

            output_file_name = f"{update.message.message_id}_{str(datetime.datetime.now().timestamp())}.mp4"
            output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

            video_title = await download_video_to_file(video_url, output_file_path)

            cleanup_paths.add(output_file_path)

            file_to_send_path = await compress_video_if_needed(output_file_path)
            cleanup_paths.add(file_to_send_path)

            await status_message.edit_text("Download completed successfully. Sending the video...")
            caption_url = await resolve_caption_url(video_url)
            video_caption = _truncate_caption_text(
                f'{video_title}\n<a href="{caption_url}">original link</a>\nRequested by: {sender_display}'
            )

            with open(file_to_send_path, 'rb') as video:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=video,
                    reply_to_message_id=update.message.message_id,
                    caption=video_caption,
                    parse_mode=ParseMode.HTML,
                )

            await _delete_message_if_exists(status_message)
            await update.message.delete()
        except Exception as e:
            logger.error(f"Error during video download or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while processing this media link.")
            await _delete_message_if_exists(status_message)
        finally:
            for path in cleanup_paths:
                _remove_file_if_exists(path)
    else:
        if update.effective_chat.type in ['group', 'supergroup']:
            logger.info(f"Non-video-link message in group chat: {message_text}")
            await handle_group_ai_reply(update, context)
        return


async def handle_photo_for_group_ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle Telegram photos in groups by converting image content to text and reusing AI reply flow."""
    if not update.message or not update.message.photo:
        return
    if not _is_group_chat(update):
        return

    photo = update.message.photo[-1]
    photo_path = _build_output_path("photo", update.message.message_id, extension="jpg")

    try:
        tg_file = await context.bot.get_file(photo.file_id)
        await tg_file.download_to_drive(custom_path=photo_path)

        image_text = await image_to_text(photo_path)
        caption_text = (update.message.caption or "").strip()

        if not image_text and not caption_text:
            return

        combined_parts = []
        if image_text:
            combined_parts.append(image_text)
        if caption_text:
            combined_parts.append(f"caption: {caption_text}")
        synthesized_text = "\n".join(combined_parts)

        await _handle_group_ai_reply_pipeline(
            update,
            synthesized_text,
            additional_context=[
                "input_type: image",
                f"image_message_id: {update.message.message_id}",
                f"captured_at_unix: {int(time.time())}",
            ],
        )
    except Exception as e:
        logger.error(f"Error handling group image message: {e}")
    finally:
        _remove_file_if_exists(photo_path)


async def handle_crypto_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    '''
    the get_crypto_prices, get_Allez_APR, get_Allez_USDC_APR will return something like:
    {'SOL': 181.888, 'ETH': 3827.993, 'BTC': 398.588, 'USDC': 1.0, 'USDT': 1.0}
    {'name': 'Allez SOL', 'APR_24H': '11.65%', 'APR_7D': '6.33%', 'APR_30D': '5.78%', 'APR_90D': '5.88%', 'Total_Supply': '10.43M'}
    {'name': 'Allez USDC', 'APR_24H': '3.57%', 'APR_7D': '4.61%', 'APR_30D': '5.01%', 'APR_90D': '10.85%', 'Total_Supply': '59.94M'}'''
    if not update.message:
        return
    
    try:
        # prices = await get_Price(["BTC", "ETH", "SOL"])
        prices = await get_Price_Coinbase(["SOL", "USDC", "BTC", "ETH", "USDT"])
        # Sort of prices by key
        prices = dict(sorted(prices.items()))
        allez_sol_apr = await get_Allez_APR()
        allez_usdc_apr = await get_Allez_USDC_APR()

        price_lines = [f"{token}: ${price}" for token, price in prices.items()]
        price_message = "Current Crypto Prices:\n" + "\n".join(price_lines)

        allez_sol_lines = [f"{key}: {value}" for key, value in allez_sol_apr.items()]
        allez_sol_message = "\n\n <a href=\"https://kamino.com/lend/allez-sol\">Allez SOL</a> APR Info:\n" + "\n".join(allez_sol_lines)

        allez_usdc_lines = [f"{key}: {value}" for key, value in allez_usdc_apr.items()]
        allez_usdc_message = '\n\n <a href="https://kamino.com/lend/allez-usdc">Allez USDC</a> APR Info:\n' + "\n".join(allez_usdc_lines)

        full_message = price_message + allez_sol_message + allez_usdc_message

        await update.message.reply_text(full_message, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except Exception as e:
        logger.error(f"Error fetching crypto prices: {e}")
        await update.message.reply_text("Sorry, I encountered an error while fetching crypto prices.")


async def handle_medjpg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /med2jpg command to generate med image from text."""
    if not update.message or not update.message.text:
        return
    logger.info(f"Received text for MED rendering: {update.message.text if update.message else 'No message text'}")
    message_text = update.message.text
    await update.message.reply_text("Processing your MED image request...")
    json_prompt = await generate_med(message_text)
    if not json_prompt:
        await update.message.reply_text("Failed to generate MED JSON from the provided text.")
        return
    output_file_path = _build_output_path("med", update.message.message_id)
    status_message = None
    try:
        status_message = await update.message.reply_text("Generating your MED image, please wait a moment...")

        # Convert the generated prescription data straight to JPG
        jpg_path = await generate_jpg_from_med_json(json_prompt, output_file_path)
        if not jpg_path or not os.path.exists(jpg_path):
            raise FileNotFoundError(f"MED JPG not created at {jpg_path}")

        if not update.effective_chat:
            return

        with open(jpg_path, 'rb') as photo:
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=photo,
                reply_to_message_id=update.message.message_id
            )
        await _delete_message_if_exists(status_message)
    except Exception as e:
        logger.error(f"Error during MED image generation or sending: {e}")
        await update.message.reply_text("Sorry, I encountered an error while creating your MED image.")
        await _delete_message_if_exists(status_message)
    finally:
        _remove_file_if_exists(output_file_path)


async def handle_application_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Global error handler for Telegram application-level exceptions."""
    err = context.error
    if isinstance(err, Conflict):
        logger.warning(
            "Telegram polling conflict: another bot instance is calling getUpdates. "
            "Stop other instances or switch to webhook mode."
        )
        return

    logger.exception("Unhandled telegram error", exc_info=err)


def register_handlers(application: Application) -> None:
    """Register all command and message handlers in one place."""
    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))

    # Commands for rendering to image
    application.add_handler(CommandHandler("md2jpg", handle_md2jpg_and_text2jpg))
    application.add_handler(CommandHandler("text2jpg", handle_md2jpg_and_text2jpg))

    # Command for rendering med
    application.add_handler(CommandHandler("med2jpg", handle_medjpg))

    # Documents (.txt, .md)
    application.add_handler(MessageHandler(filters.Document.ALL, handle_text_or_markdown_document))

    # General text: YouTube downloads or group AI replies
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_for_youtube_or_group))

    # Group images: convert to text and pass through AI reply process
    application.add_handler(MessageHandler(filters.PHOTO & ~filters.COMMAND, handle_photo_for_group_ai_reply))

    # Cryto info command
    application.add_handler(CommandHandler("crypto", handle_crypto_command))


def main() -> None:
    """Start the bot."""

    # Initialize the database
    init_db()

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_KEY).read_timeout(30).write_timeout(30).build()

    register_handlers(application)
    application.add_error_handler(handle_application_error)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Start the bot
    main()
