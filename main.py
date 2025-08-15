# general imports
import logging
import os
import re
import datetime
import random
import asyncio
from collections import defaultdict, deque
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler
from telegram.constants import ParseMode


# private imports
import secret
from app.md2jpg import md_to_image
from app.text2md import plain_text_to_markdown
from app.youtube_dl import download_video_720p_h264, get_video_title
from app.reply2message import should_reply_and_generate
from app.database import init_db, add_message, get_messages


AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, TELEGRAM_BOT_USERNAME, TELEGRAM_BOT_KEY = secret.pass_secret_variables()
OUTPUT_DIR = "output"
AZURE_OPENAI_API_VERSION = "2024-04-01-preview"

# Models: Phi-4-mini-instruct, Phi-4 or gpt-4.1-nano
AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-5-mini"  # or "phi-4-mini-instruct" or "phi-4" or 'gpt-4.1-nano' or 'gpt-4.1-mini'

# A robust YouTube URL regex pattern (ID = 11 chars)
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


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)



# -------- Telegram Bot Handlers --------

# Start command handler
# This handler sends a welcome message when the /start command is issued.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text(
        """Hi! I can convert Markdown to an image. Send me a message like:\n\n /md2jpg ,,,Your markdown here,,, \n\n'or\n\n /text2jpg ,,,Your plain text here,,, \n\nI can also download YouTube videos if you send me a link, and I might reply to messages in this group if I find them interesting, nya~"""
    )


async def handle_md2jpg_and_text2jpg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /md2jpg and /text2jpg commands to generate images."""
    if not update.message or not update.message.text:
        return
    logger.info(f"Received text for rendering: {update.message.text if update.message else 'No message text'}")
    message_text = update.message.text

    # This regex handles optional bot username in the command, e.g., md2jpg@MioooooooooBot
    match_md2jpg = re.search(r'/md2jpg(?:@\w+)?\s*,,,(.*),,,', message_text, re.DOTALL)
    match_text2jpg = re.search(r'/text2jpg(?:@\w+)?\s*,,,(.*),,,', message_text, re.DOTALL)

    # md2jpg command handling
    if match_md2jpg:
        markdown_input = match_md2jpg.group(1).strip()
        if not markdown_input:
            await update.message.reply_text("Please provide some markdown content inside the triple quotes.")
            return

        output_file_name = f"md_{update.message.message_id}.jpg"
        output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

        status_message = None
        try:
            status_message = await update.message.reply_text("Generating your image, please wait a moment...")

            await md_to_image(md_text=markdown_input, output_path=output_file_path, theme='formal_code')

            with open(output_file_path, 'rb') as photo:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=photo,
                    reply_to_message_id=update.message.message_id
                )
            await status_message.delete()
        except Exception as e:
            logger.error(f"Error during image generation or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while creating your image.")
            await status_message.delete() if status_message else None
        finally:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)

    # text2jpg command handling
    if match_text2jpg:
        plain_text_input = match_text2jpg.group(1).strip()
        if not plain_text_input:
            await update.message.reply_text("Please provide some text content inside the triple quotes.")
            return

        output_file_name = f"text_{update.message.message_id}.jpg"
        output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

        status_message = None
        try:
            status_message = await update.message.reply_text("Converting your text to markdown, please wait a moment...")

            generated_markdown = await plain_text_to_markdown(
                plain_text_input, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME
            )

            await status_message.edit_text("Generating your image from markdown, please wait a moment...")

            await md_to_image(md_text=generated_markdown, output_path=output_file_path, theme='formal_code')

            with open(output_file_path, 'rb') as photo:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=photo,
                    reply_to_message_id=update.message.message_id
                )
            await status_message.delete()
        except Exception as e:
            logger.error(f"Error during image generation or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while creating your image.")
            await status_message.delete() if status_message else None
        finally:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)


# Handle .txt or .md files to render as image
async def handle_text_or_markdown_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle .txt or .md files to render as image."""
    if not update.message or not update.message.document:
        return

    document_file = update.message.document
    is_already_markdown = document_file.file_name.endswith('.md')

    if document_file.file_name.endswith(('.txt', '.md')):
        tg_file = await document_file.get_file()
        downloaded_path = await tg_file.download_to_drive(
            custom_path=os.path.join(OUTPUT_DIR, document_file.file_name)
        )

        with open(downloaded_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

        output_file_name = f"file_{update.message.message_id}.jpg"
        output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

        status_message = None
        try:
            status_message = await update.message.reply_text("Converting your file to markdown, please wait a moment...")

            if not is_already_markdown:
                generated_markdown = await plain_text_to_markdown(
                    file_content, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME
                )
            else:
                generated_markdown = file_content

            await status_message.edit_text("Generating your image from markdown, please wait a moment...")

            await md_to_image(md_text=generated_markdown, output_path=output_file_path, theme='formal_code')

            with open(output_file_path, 'rb') as photo:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=photo,
                    reply_to_message_id=update.message.message_id
                )
            await status_message.delete()
        except Exception as e:
            logger.error(f"Error during image generation or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while creating your image.")
            await status_message.delete() if status_message else None
        finally:
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
            if os.path.exists(downloaded_path):
                os.remove(downloaded_path)


# Handle Group AI Replies
async def handle_group_ai_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle group messages and occasionally reply using AI."""
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    message_text = update.message.text

    print(f"Adding message to history for chat {update.effective_user.full_name}: {message_text}")
    await add_message(
        chat_id=chat_id,
        username=update.effective_user.full_name,
        content=message_text
    )

    is_reply_to_bot = False
    if (
        update.message.reply_to_message and
        update.message.reply_to_message.from_user and
        update.message.reply_to_message.from_user.is_bot and
        update.message.reply_to_message.from_user.username == TELEGRAM_BOT_USERNAME
    ):
        is_reply_to_bot = True
        logger.info(f"User {update.effective_user.full_name} replied to the bot.")

    # 1 in 5 chance to consider replying, unless it's a reply to the bot.
    if not is_reply_to_bot and random.randint(1, 5) != 1:
        return

    ai_reply = await should_reply_and_generate(
        message_history=await get_messages(chat_id),
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        is_reply_to_bot=is_reply_to_bot
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

    message_text = update.message.text.strip()
    youtube_match = re.search(YOUTUBE_URL_REGEX, message_text)
    bilibili_match = re.search(BILIBILI_URL_REGEX, message_text)

    match = youtube_match or bilibili_match

    if match:
        if youtube_match:
            youtube_url = youtube_match.group(0)
        elif bilibili_match:
            youtube_url = bilibili_match.group(0)
        else:
            logger.error("No valid YouTube or Bilibili URL found in the message.")
            return

        status_message = None
        try:
            status_message = await update.message.reply_text("Downloading your video, please wait a moment...")

            video_title = await get_video_title(youtube_url)
            output_file_name = f"{video_title}_{update.message.message_id}_{str(datetime.datetime.now().timestamp())}.mp4"
            output_file_path = os.path.join(OUTPUT_DIR, output_file_name)

            await download_video_720p_h264(youtube_url, output_path=output_file_path)

            await status_message.edit_text("Download completed successfully. Sending the video...")

            with open(output_file_path, 'rb') as video:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=video,
                    reply_to_message_id=update.message.message_id,
                    caption=f'{video_title}\n<a href="{youtube_url}">original link</a>\nRequested by: {update.effective_user.full_name}',
                    parse_mode=ParseMode.HTML
                )
            await status_message.delete()
            await update.message.delete()
            if os.path.exists(output_file_path):
                os.remove(output_file_path)
        except Exception as e:
            logger.error(f"Error during video download or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while downloading your video.")
            await status_message.delete() if status_message else None
    else:
        if update.effective_chat.type in ['group', 'supergroup']:
            logger.info(f"Non-YouTube message in group chat: {message_text}")
            await handle_group_ai_reply(update, context)
        return



def main() -> None:
    """Start the bot."""

    # Initialize the database
    init_db()

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_KEY).read_timeout(30).write_timeout(30).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))

    # Commands for rendering to image
    application.add_handler(CommandHandler("md2jpg", handle_md2jpg_and_text2jpg))
    application.add_handler(CommandHandler("text2jpg", handle_md2jpg_and_text2jpg))

    # Documents (.txt, .md)
    application.add_handler(MessageHandler(filters.Document.ALL, handle_text_or_markdown_document))

    # General text: YouTube downloads or group AI replies
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_for_youtube_or_group))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Start the bot
    main()