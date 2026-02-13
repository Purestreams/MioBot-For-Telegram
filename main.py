"""Telegram bot entrypoint and handler orchestration."""

# general imports
import datetime
import logging
import os
import random
import re
from typing import Optional

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters


# private imports
import secret
from app.md2jpg import md_to_image
from app.text2md import plain_text_to_markdown
from app.youtube_dl import download_video_720p_h264, get_video_title
from app.reply2message import should_reply_and_generate
from app.database import init_db, add_message, get_prompt_context_parts

from app.cryto import get_Allez_APR, get_Allez_USDC_APR, get_Price_Coinbase

from app.med import generate_jpg_from_med_json, generate_med
from app.ai_model import configure_llm


AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, TELEGRAM_BOT_USERNAME, TELEGRAM_BOT_KEY, ARK_ENDPOINT, ARK_API_KEY = secret.pass_secret_variables()

secret.set_environment()

OUTPUT_DIR = "output"
AZURE_OPENAI_API_VERSION = "2024-04-01-preview"

# Models: Phi-4-mini-instruct, Phi-4 or gpt-4.1-nano
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-5-mini"  # or "phi-4-mini-instruct" or "phi-4" or 'gpt-4.1-nano' or 'gpt-4.1-mini'

ARK_API_KEY = os.getenv("ARK_API_KEY")
ARK_MODEL = os.getenv("ARK_MODEL")  # or "deepseek-r1-250528"
LLM_PROVIDER = os.getenv("LLM_PROVIDER") or os.getenv("AI_PROVIDER")
if LLM_PROVIDER:
    normalized_provider = LLM_PROVIDER.strip().lower()
    if normalized_provider in {"azure_openai", "azure-openai", "azureopenai"}:
        LLM_PROVIDER = "azure"
    else:
        LLM_PROVIDER = normalized_provider

configure_llm(
    provider=LLM_PROVIDER,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_api_key=AZURE_OPENAI_API_KEY,
    azure_api_version=AZURE_OPENAI_API_VERSION,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT_NAME,
    ark_api_key=ARK_API_KEY,
    ark_model=ARK_MODEL,
)

# URL regex patterns
YOUTUBE_URL_REGEX = (
    r'(https?://)?(www\.)?'
    r'(youtube\.com/|youtu\.be/|youtube-nocookie\.com/)'
    r'(?:watch\?v=|embed/|v/|shorts/|live/)?'
    r'([a-zA-Z0-9_-]{11})'
)
BILIBILI_URL_REGEX = (
    r'(https?://)?(?:www\.|m\.)?'
    r'(bilibili\.com/|b23\.tv/)'
    r'(?:video/|watch\?bvid=)?'
    r'([A-Za-z0-9_-]{6,12})'
    r'(?:[/?#][^\s]*)?'
)

MD2JPG_REGEX = r'/md2jpg(?:@\w+)?\s*,,,(.*),,,'
TEXT2JPG_REGEX = r'/text2jpg(?:@\w+)?\s*,,,(.*),,,'

RAG_KEYWORD_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "for", "with", "and", "or", "but", "if", "then",
    "this", "that", "it", "as", "at", "by", "from", "about", "just", "very",
    "you", "your", "me", "my", "we", "our", "they", "their", "he", "she", "his", "her",
}


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)



def _build_output_path(prefix: str, message_id: int, extension: str = "jpg") -> str:
    return os.path.join(OUTPUT_DIR, f"{prefix}_{message_id}.{extension}")


def _remove_file_if_exists(path) -> None:
    if path and os.path.exists(path):
        os.remove(path)


async def _delete_message_if_exists(message) -> None:
    if message:
        await message.delete()


def _extract_video_url(message_text: str) -> Optional[str]:
    youtube_match = re.search(YOUTUBE_URL_REGEX, message_text)
    bilibili_match = re.search(BILIBILI_URL_REGEX, message_text)

    if youtube_match:
        return youtube_match.group(0)
    if bilibili_match:
        return bilibili_match.group(0)
    return None


def _is_reply_to_this_bot(update: Update) -> bool:
    message = update.message
    if not message or not message.reply_to_message:
        return False

    from_user = message.reply_to_message.from_user
    return bool(
        from_user
        and from_user.is_bot
        and from_user.username == TELEGRAM_BOT_USERNAME
    )


def _match_command_payload(message_text: str, regex_pattern: str) -> Optional[str]:
    match = re.search(regex_pattern, message_text, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _extract_search_keywords(message_text: str, *, max_keywords: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_]{2,}", message_text.lower())
    keywords: list[str] = []
    seen = set()

    for token in tokens:
        if token in RAG_KEYWORD_STOPWORDS:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def _build_rag_query_from_message(message_text: str) -> str:
    keywords = _extract_search_keywords(message_text)
    if keywords:
        return " ".join(keywords)
    return message_text


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

    if not update.effective_chat or not update.effective_user:
        return

    chat_id = update.effective_chat.id
    message_text = update.message.text

    print(f"Adding message to history for chat {update.effective_user.full_name}: {message_text}")
    await add_message(
        chat_id=chat_id,
        username=update.effective_user.full_name,
        content=message_text
    )

    is_reply_to_bot = _is_reply_to_this_bot(update)
    if is_reply_to_bot:
        logger.info(f"User {update.effective_user.full_name} replied to the bot.")

    # 1 in 5 chance to consider replying, unless it's a reply to the bot.
    if not is_reply_to_bot and random.randint(1, 5) != 1:
        return

    rag_query = _build_rag_query_from_message(message_text)
    history_messages, rag_related_messages = await get_prompt_context_parts(chat_id, query=rag_query)

    ai_reply = await should_reply_and_generate(
        message_history=history_messages,
        rag_related_messages=rag_related_messages,
        is_reply_to_bot=is_reply_to_bot,
    )

    if ai_reply:
        await add_message(
            chat_id=chat_id,
            username="mioo_bot",
            content=ai_reply
        )
        try:
            await update.message.reply_text(ai_reply)
        except Exception as e:
            logger.error(f"Error sending AI reply: {e}")

# Handle text messages: download YouTube videos, else pass to group AI handler
async def handle_text_for_youtube_or_group(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages: download YouTube videos, else pass to group AI handler."""
    if not update.message or not update.message.text:
        return

    if not update.effective_chat or not update.effective_user:
        return

    message_text = update.message.text.strip()
    video_url = _extract_video_url(message_text)

    if video_url:

        status_message = None
        try:
            status_message = await update.message.reply_text("Downloading your video, please wait a moment...")

            video_title = await get_video_title(video_url)
            output_file_name = f"{video_title}_{update.message.message_id}_{str(datetime.datetime.now().timestamp())}.mp4"
            output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

            await download_video_720p_h264(video_url, output_path=output_file_path)

            await status_message.edit_text("Download completed successfully. Sending the video...")

            with open(output_file_path, 'rb') as video:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=video,
                    reply_to_message_id=update.message.message_id,
                    caption=f'{video_title}\n<a href="{video_url}">original link</a>\nRequested by: {update.effective_user.full_name}',
                    parse_mode=ParseMode.HTML
                )
            await _delete_message_if_exists(status_message)
            await update.message.delete()
            _remove_file_if_exists(output_file_path)
        except Exception as e:
            logger.error(f"Error during video download or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while downloading your video.")
            await _delete_message_if_exists(status_message)
    else:
        if update.effective_chat.type in ['group', 'supergroup']:
            logger.info(f"Non-YouTube message in group chat: {message_text}")
            await handle_group_ai_reply(update, context)
        return


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

    # Cryto info command
    application.add_handler(CommandHandler("crypto", handle_crypto_command))


def main() -> None:
    """Start the bot."""

    # Initialize the database
    init_db()

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_KEY).read_timeout(30).write_timeout(30).build()

    register_handlers(application)

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Start the bot
    main()