# Portfolio Lead Capture Bot

A lightweight Telegram bot that walks prospective clients through a short discovery flow. It stores structured lead information in a local SQLite database so you can follow up manually. The bot can run with scripted replies or, if you provide an OpenAI API key, it will use GPT responses to keep the conversation warm and polished.

## Features
- Step-by-step conversational flow that captures project intent, goals, timeline, budget, and contact details.
- Optional OpenAI-powered dialogue (set `OPENAI_API_KEY`) for human-like responses; falls back to scripted prompts otherwise.
- Simple validation and restart command to keep answers tidy.
- SQLite persistence (`leads.db`) so conversations survive restarts.

## Getting Started

1. **Clone / copy the folder** into your workspace.
2. **Create a Telegram bot** via [@BotFather](https://t.me/BotFather) and copy the token.
3. **Create a `.env` file** (example below) alongside `app.py`.

   ```env
   BOT_TOKEN=123456:ABC-DEF
   TEAM_CONTACT=@your_handle
   OPENAI_API_KEY=sk-your-key
   OPENAI_MODEL=gpt-4.1-mini
   ```

   - `TEAM_CONTACT` is optional and defaults to `@your_handle`.
   - `OPENAI_API_KEY` is optional. If you omit it, the bot will use scripted prompts.
   - `OPENAI_MODEL` defaults to `gpt-4.1-mini`, but you can point to any chat-capable model available to your account.

4. **Install dependencies** (use a virtual environment if desired):

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the bot**:

   ```bash
   python app.py
   ```

   The bot will create a `leads.db` file automatically and start polling Telegram for new messages.

## Commands
- `/start` – greet the user and start (or resume) the discovery questions.
- `/help` – display a short help message.
- `/reset` – clear stored answers for the current user and restart the flow.

## Database Schema
All leads are stored in `leads.db` under the `leads` table with the following columns:

| column        | description                              |
|---------------|------------------------------------------|
| user_id       | Telegram numeric identifier              |
| username      | Telegram username                        |
| first_name    | Telegram first name                      |
| language_code | Telegram language code (if provided)     |
| pending_field | The field currently being collected      |
| project_type  | User answer for project type             |
| project_goal  | User answer describing goals             |
| timeline      | User answer describing timeline          |
| budget_range  | User answer describing budget            |
| contact       | Preferred follow-up contact              |
| notes         | Additional info the user shared          |
| status        | `collecting` or `complete`               |
| created_at    | ISO timestamp of record creation         |
| updated_at    | ISO timestamp of last update             |
| completed_at  | ISO timestamp when flow finished         |

## Customization Ideas
- Adjust the `FIELD_FLOW` list inside `app.py` to collect different data.
- Add CSV export or Airtable/Notion integrations for portfolio demos.
- Deploy to a small VPS or serverless worker for live portfolio showcases.

Have fun tailoring it to your personal brand!
