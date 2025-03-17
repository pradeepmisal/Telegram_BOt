# Crypto Whale Alert Bot

This is a Telegram bot that sends alerts about crypto transactions and price movements.

## Features

- Alerts for large transactions (whale alerts)
- Price updates for specific cryptocurrencies
- User preferences management
- Integration with various crypto APIs

## Setup

1. Clone the repository.
2. Create a `.env` file in the root directory and add your API keys.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the bot:
   ```bash
   python bot.py
   ```

## Commands

- `/start` - Start the bot
- `/help` - Show available commands
- `/ping` - Check if the bot is running
- `/price <coin_symbol>` - Get the current price for a specific coin
- `/news` - Get the latest crypto news
- `/settings` - Manage user settings
