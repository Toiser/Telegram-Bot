# Telegram AI Concierge Bot (Poker Leads MVP)

An AI-powered Telegram bot that:
- Answers short poker/rakeback FAQs using the OpenAI API
- Collects lead info (country, preferred site, stakes/format)
- Hands off to a human for manual referral negotiation
- Stores basic leads in a local SQLite DB

## 0) Create a Telegram bot token
1. In Telegram, open **@BotFather**
2. `/newbot` → choose a name + username
3. Copy the token

## 1) Configure environment variables
Create a `.env` file (copy from `.env.example`):
```
BOT_TOKEN=123456789:ABCDEF-your-telegram-token
OPENAI_API_KEY=sk-your-openai-key
OPENAI_MODEL=gpt-4o-mini
HUMAN_USERNAME=YourUsername  # without @
```

## 2) Install & run
```
pip install -r requirements.txt
python app.py
```
The bot uses long polling, so no server is needed for testing. Press Ctrl+C to stop.

## 3) Try it
- In Telegram, send `/start` to your bot
- Answer the questions (country → site → stakes)
- Ask a FAQ like: "What is rakeback?" or "Is GG available in Spain?"

## Commands
- `/start` — begins the flow
- `/help` — shows help
- `/delete` — delete your stored lead (GDPR-friendly)
- `/ping` — health check

## Notes
- The bot **never sends referral links**. It hands off to the human (your username).
- DB file `leads.db` is created automatically in the same folder on first run.
- This is a minimal MVP: extend intents, add language detection, and enrich FAQs as needed.
