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

# In-memory message history
#MESSAGE_REVIEW_BACK = 40
#message_history = defaultdict(lambda: deque(maxlen=MESSAGE_REVIEW_BACK))


# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text(
        """Hi! I can convert Markdown to an image. Send me a message like:\n\n /md2jpg ,,,Your markdown here,,, \n\n'or\n\n /text2jpg ,,,Your plain text here,,, \n\nI can also download YouTube videos if you send me a link, and I might reply to messages in this group if I find them interesting, nya~"""
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages to convert markdown to image."""
    if not update.message or not update.message.text:
        return
    logger.info(f"Received message: {update.message.text if update.message else 'No message text'}")
    text = update.message.text
    # Use a flexible regex to find the command and content
    # This regex handles optional bot username in the command, e.g., md2jpg@MioooooooooBot
    match_md2jpg = re.search(r'/md2jpg(?:@\w+)?\s*,,,(.*),,,', text, re.DOTALL)
    match_text2jpg = re.search(r'/text2jpg(?:@\w+)?\s*,,,(.*),,,', text, re.DOTALL)

    # md2jpg command handling
    if match_md2jpg:
        md_content = match_md2jpg.group(1).strip()
        if not md_content:
            await update.message.reply_text("Please provide some markdown content inside the triple quotes.")
            return

        # Create a unique filename for the output image
        output_filename = f"md_{update.message.message_id}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        massage_block = None
        try:
            # Let the user know the process has started
            massage_block = await update.message.reply_text("Generating your image, please wait a moment...")

            # The md_to_image function is now asynchronous
            await md_to_image(md_text=md_content, output_path=output_path, theme='formal_code')

            # Send the image back to the user
            with open(output_path, 'rb') as photo:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=photo,
                    reply_to_message_id=update.message.message_id
                )
            
            await massage_block.delete()

        except Exception as e:
            logger.error(f"Error during image generation or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while creating your image.")
            await massage_block.delete() if massage_block else None

        finally:
            # Clean up by deleting the generated image file
            if os.path.exists(output_path):
                os.remove(output_path)

    # text2jpg command handling
    if match_text2jpg:
        text_content = match_text2jpg.group(1).strip()

        if not text_content:
            await update.message.reply_text("Please provide some text content inside the triple quotes.")
            return

        # Create a unique filename for the output image
        output_filename = f"text_{update.message.message_id}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        massage_block = None
        try:
            massage_block = await update.message.reply_text("Converting your text to markdown, please wait a moment...")

            # Convert plain text to markdown
            markdown_content = await plain_text_to_markdown(text_content, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME)


            # Let the user know the process has started, change the massage of massage_block
            await massage_block.edit_text("Generating your image from markdown, please wait a moment...")

            # The md_to_image function is now asynchronous
            await md_to_image(md_text=markdown_content, output_path=output_path, theme='formal_code')

            # Send the image back to the user
            with open(output_path, 'rb') as photo:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=photo,
                    reply_to_message_id=update.message.message_id
                )

            await massage_block.delete()

        except Exception as e:
            logger.error(f"Error during image generation or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while creating your image.")
            await massage_block.delete() if massage_block else None

        finally:
            # Clean up by deleting the generated image file
            if os.path.exists(output_path):
                os.remove(output_path)

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle document messages to convert .txt or .md files to image."""
    if not update.message or not update.message.document:
        return

    file = update.message.document

    skip_convert = False
    if file.file_name.endswith('.md'):
        skip_convert = True

    if file.file_name.endswith(('.txt', '.md')):
        # Download the file
        telegram_file = await file.get_file()
        file_path = await telegram_file.download_to_drive(custom_path=os.path.join(OUTPUT_DIR, file.file_name))
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Create a unique filename for the output image
        output_filename = f"file_{update.message.message_id}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        massage_block = None
        try:
            massage_block = await update.message.reply_text("Converting your file to markdown, please wait a moment...")

            # Convert plain text to markdown
            if not skip_convert:
                markdown_content = await plain_text_to_markdown(content, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT_NAME)
            else:
                markdown_content = content

            # Let the user know the process has started, change the massage of massage_block
            await massage_block.edit_text("Generating your image from markdown, please wait a moment...")

            # The md_to_image function is now asynchronous
            await md_to_image(md_text=markdown_content, output_path=output_path, theme='formal_code')

            # Send the image back to the user
            with open(output_path, 'rb') as photo:
                await context.bot.send_document(
                    chat_id=update.effective_chat.id,
                    document=photo,
                    reply_to_message_id=update.message.message_id
                )

            await massage_block.delete()

        except Exception as e:
            logger.error(f"Error during image generation or sending: {e}")
            await update.message.reply_text("Sorry, I encountered an error while creating your image.")
            await massage_block.delete() if massage_block else None

        finally:
            # Clean up by deleting the generated image file and downloaded file
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(file_path):
                os.remove(file_path)

async def handle_group_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles text messages in groups for potential replies."""
    if not update.message or not update.message.text:
        return

    chat_id = update.effective_chat.id
    text = update.message.text

    # Add current message to history with username
    print(f"Adding message to history for chat {update.effective_user.full_name}: {text}")
    #message_history[chat_id].append(
    #    f"username: {update.effective_user.full_name} \n content: {text}"
    #)
    await add_message(
        chat_id=chat_id,
        username=update.effective_user.full_name,
        content=text
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


    # Randomly decide whether to check for a reply to avoid spamming
    # 1 in 2 chance to consider replying, unless it's a reply to the bot.
    if not is_reply_to_bot and random.randint(1, 5) != 1:
        return

    # Call the AI to see if we should reply
    reply_content = await should_reply_and_generate(
        #message_history=list(message_history[chat_id]),
        message_history=await get_messages(chat_id),
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        is_reply_to_bot=is_reply_to_bot
    )

    if reply_content:
        #message_history[chat_id].append(f"Mioo Bot: {reply_content}")
        await add_message(
            chat_id=chat_id,
            username="mioo_bot",
            content=reply_content
        )
        try:
            await update.message.reply_text(reply_content)
        except Exception as e:
            logger.error(f"Error sending AI reply: {e}")


async def handle_all_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all text messages to download YouTube videos."""
    if not update.message or not update.message.text:
        return

    text = update.message.text.strip()
    # This regex is more robust for finding YouTube URLs anywhere in the message
    youtube_regex = (
        r'(https?://)?(www\.)?'
        r'(youtube\.com/|youtu\.be/|youtube-nocookie\.com/)'
        r'(?:watch\?v=|embed/|v/|shorts/|live/)?'
        r'([a-zA-Z0-9_-]{11})'
    )

    match = re.search(youtube_regex, text)

    if not match:
        # If not a youtube link, and it's a group chat, let the group handler deal with it
        if update.effective_chat.type in ['group', 'supergroup']:
            logger.info(f"Non-YouTube message in group chat: {text}")
            await handle_group_text(update, context)
        return

    youtube_url = match.group(0) # The matched URL

    massage_block = None
    try:
        massage_block = await update.message.reply_text("Downloading your video, please wait a moment...")

        title = await get_video_title(youtube_url)
        output_filename = f"{title}_{update.message.message_id}_{str(datetime.datetime.now().timestamp())}.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        # Call the download function
        await download_video_720p_h264(youtube_url, output_path=output_path)

        await massage_block.edit_text("Download completed successfully. Sending the video...")

        # Send the video back to the user
        with open(output_path, 'rb') as video:
            await context.bot.send_document(
                chat_id=update.effective_chat.id,
                document=video,
                reply_to_message_id=update.message.message_id,
                caption=f'{title}\n<a href="{youtube_url}">original link</a>\nRequested by: {update.effective_user.full_name}',
                parse_mode=ParseMode.HTML
            )
        await massage_block.delete()
        # delete the link message
        await update.message.delete()
        # delete the downloaded video file
        if os.path.exists(output_path):
            os.remove(output_path)
    except Exception as e:
        logger.error(f"Error during video download or sending: {e}")
        await update.message.reply_text("Sorry, I encountered an error while downloading your video.")
        await massage_block.delete() if massage_block else None



def main() -> None:
    """Start the bot."""

    # Initialize the database
    init_db()

    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_BOT_KEY).read_timeout(30).write_timeout(30).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(CommandHandler("md2jpg", handle_message))
    application.add_handler(CommandHandler("text2jpg", handle_message))

    # on file with .txt or .md extension - convert to image
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    # on youtube video download command - download video
    # This handler now also routes non-command, non-youtube-link messages from groups to the new group handler.
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_all_text))

    # application.add_handler(MessageHandler(filters.TEXT, handle_message))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # Start the bot
    main()