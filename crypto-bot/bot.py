import os
import logging
import json
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get token from environment variable
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')

# User preference file
USER_PREFS_FILE = 'user_preferences.json'

def load_user_preferences():
    """Load user preferences from file"""
    try:
        if os.path.exists(USER_PREFS_FILE):
            with open(USER_PREFS_FILE, 'r') as f:
                return json.load(f)
        return {}
    except Exception as e:
        logger.error(f"Error loading user preferences: {e}")
        return {}

def save_user_preferences(prefs):
    """Save user preferences to file"""
    try:
        with open(USER_PREFS_FILE, 'w') as f:
            json.dump(prefs, f)
    except Exception as e:
        logger.error(f"Error saving user preferences: {e}")

def init_user_preferences(user_id):
    """Initialize user preferences if not exists"""
    prefs = load_user_preferences()
    if str(user_id) not in prefs:
        prefs[str(user_id)] = {
            'whale_alert_threshold': 1000000,
            'price_alert_threshold': 10,
            'watched_coins': ['BTC', 'ETH'],
            'chains_monitored': ['ethereum', 'bsc'],
            'alert_frequency': 'high',
            'news_updates': True
        }
        save_user_preferences(prefs)
    return prefs[str(user_id)]

# Command handlers
async def price(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Get price for a specific coin."""
    if not context.args:
        await update.message.reply_text("Please specify a coin symbol, e.g. /price BTC")
        return
    
    coin_symbol = context.args[0].upper()
    await update.message.reply_text(f"Fetching price for {coin_symbol}...")
    
    coin_data = await fetch_coin_data(coin_symbol)
    if coin_data:
        message = (
            f"💰 *{coin_data['name']} ({coin_data['symbol']})*\n\n"
            f"Price: ${coin_data['price']:,.6f}\n"
            f"24h Change: {coin_data['price_change_24h']:+.2f}%\n"
            f"Market Cap: ${coin_data['market_cap']:,.0f}\n"
            f"24h Volume: ${coin_data['volume_24h']:,.0f}"
        )
        await update.message.reply_text(message, parse_mode='Markdown')
    else:
        await update.message.reply_text(f"Sorry, couldn't find data for {coin_symbol}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message\n"
        "/ping - Check if bot is running"
    )

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check if the bot is responsive."""
    await update.message.reply_text("Pong! The bot is running.")

from web3 import Web3

INFURA_API_KEY = os.getenv('INFURA_API_KEY')
ethereum_provider = Web3(Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{INFURA_API_KEY}"))

async def get_latest_eth_block():
    """Get the latest Ethereum block information."""
    try:
        block_number = ethereum_provider.eth.block_number
        return f"Latest Ethereum Block: {block_number}"
    except Exception as e:
        logger.error(f"Error fetching Ethereum block: {e}")
        return "Error fetching blockchain data"

async def eth_block(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Command to get latest Ethereum block."""
    await update.message.reply_text("Fetching latest Ethereum block...")
    block_info = await get_latest_eth_block()
    await update.message.reply_text(block_info)

application = Application.builder().token(TELEGRAM_TOKEN).build()

application.add_handler(CommandHandler("start", start))
application.add_handler(CommandHandler("help", help_command))
application.add_handler(CommandHandler("ping", ping))
application.add_handler(CommandHandler("price", price))
application.add_handler(CommandHandler("ethblock", eth_block))

import asyncio
import datetime

async def scheduled_tasks(context):
    """Run scheduled tasks."""
    try:
        # Fetch news once a day
        current_hour = datetime.datetime.now().hour
        if current_hour == 8:  # Run at 8 AM
            news_articles = await fetch_crypto_news()
            if news_articles:
                # Broadcast to all users who want news
                prefs = load_user_preferences()
                for user_id, user_prefs in prefs.items():
                    if user_prefs.get('news_updates', True):
                        message = "🔥 *DAILY CRYPTO NEWS* 🔥\n\n"
                        
                        for i, article in enumerate(news_articles[:3], 1):
                            message += (
                                f"{i}. *{article['title']}*\n"
                                f"   {article['description'][:100]}...\n"
                                f"   [Read more]({article['url']})\n\n"
                            )
                        
                        try:
                            await context.bot.send_message(
                                chat_id=user_id,
                                text=message,
                                parse_mode='Markdown',
                                disable_web_page_preview=True
                            )
                        except Exception as e:
                            logger.error(f"Error sending news to user {user_id}: {e}")
    except Exception as e:
        logger.error(f"Error in scheduled tasks: {e}")

# Start the Bot
application.run_polling(allowed_updates=Update.ALL_TYPES)

# Schedule the background task
application.job_queue.run_repeating(scheduled_tasks, interval=3600, first=0)

logger.info("Bot started!")

if __name__ == '__main__':
    main()
