import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

# ---------------------------------------------------------------------------
# Environment & configuration
# ---------------------------------------------------------------------------

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
TEAM_CONTACT = os.getenv("TEAM_CONTACT", "@your_handle")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN missing – add it to a .env file before running the bot.")

if not OPENAI_API_KEY:
    print("[PortfolioBot] OPENAI_API_KEY not provided. Falling back to scripted replies.")

if OpenAI is None and OPENAI_API_KEY:
    print("[PortfolioBot] The openai package is not installed. Install it to enable AI replies.")

client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as exc:
        print(f"[PortfolioBot] Failed to initialise OpenAI client: {exc}")

DB_PATH = os.path.join(os.path.dirname(__file__), "leads.db")

FIELD_FLOW: List[Tuple[str, str]] = [
    (
        "project_type",
        "What type of project are you exploring (e.g. web app, automation, content, consulting)?",
    ),
    ("project_goal", "What goal or problem should this project solve?"),
    ("timeline", "Do you have an ideal timeline or launch date in mind?"),
    (
        "budget_range",
        "What budget range are you comfortable investing? Rough guesses are perfect.",
    ),
    ("contact", "How can we follow up with you? Please share an email address or preferred handle."),
    ("notes", "Any additional context or requirements you'd like us to know?"),
]

FIELD_LABELS = {
    "project_type": "Project type",
    "project_goal": "Goal",
    "timeline": "Timeline",
    "budget_range": "Budget",
    "contact": "Contact",
    "notes": "Notes",
}

MAX_HISTORY_EXCHANGES = 6
CONVERSATIONS: Dict[int, List[Tuple[str, str]]] = {}


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                language_code TEXT,
                pending_field TEXT,
                project_type TEXT,
                project_goal TEXT,
                timeline TEXT,
                budget_range TEXT,
                contact TEXT,
                notes TEXT,
                status TEXT,
                created_at TEXT,
                updated_at TEXT,
                completed_at TEXT
            )
            """
        )


def _dict_from_row(row: Optional[Tuple[Any, ...]]) -> Dict[str, Any]:
    if not row:
        return {}
    columns = [
        "user_id",
        "username",
        "first_name",
        "language_code",
        "pending_field",
        "project_type",
        "project_goal",
        "timeline",
        "budget_range",
        "contact",
        "notes",
        "status",
        "created_at",
        "updated_at",
        "completed_at",
    ]
    return dict(zip(columns, row))


def fetch_lead(user_id: int) -> Dict[str, Any]:
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("SELECT * FROM leads WHERE user_id=?", (user_id,))
        return _dict_from_row(cur.fetchone())


def ensure_lead(
    user_id: int,
    username: Optional[str],
    first_name: Optional[str],
    language_code: Optional[str],
) -> Dict[str, Any]:
    now = datetime.utcnow().isoformat()
    lead = fetch_lead(user_id)

    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        if lead:
            cur.execute(
                """
                UPDATE leads
                   SET username=COALESCE(?, username),
                       first_name=COALESCE(?, first_name),
                       language_code=COALESCE(?, language_code),
                       updated_at=?
                 WHERE user_id=?
                """,
                (username, first_name, language_code, now, user_id),
            )
        else:
            cur.execute(
                """
                INSERT INTO leads (
                    user_id, username, first_name, language_code,
                    pending_field, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, NULL, 'collecting', ?, ?)
                """,
                (user_id, username, first_name, language_code, now, now),
            )
    return fetch_lead(user_id)


def update_lead_field(user_id: int, field: str, value: str) -> None:
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            f"UPDATE leads SET {field}=?, updated_at=? WHERE user_id=?",
            (value, now, user_id),
        )


def set_pending_field(user_id: int, field: Optional[str]) -> None:
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            "UPDATE leads SET pending_field=?, updated_at=? WHERE user_id=?",
            (field, now, user_id),
        )


def mark_complete(user_id: int) -> None:
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            UPDATE leads
               SET status='complete', pending_field=NULL,
                   completed_at=?, updated_at=?
             WHERE user_id=?
            """,
            (now, now, user_id),
        )


def reset_lead(user_id: int) -> None:
    now = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            UPDATE leads
               SET pending_field=NULL,
                   project_type=NULL,
                   project_goal=NULL,
                   timeline=NULL,
                   budget_range=NULL,
                   contact=NULL,
                   notes=NULL,
                   status='collecting',
                   updated_at=?,
                   completed_at=NULL
             WHERE user_id=?
            """,
            (now, user_id),
        )


# ---------------------------------------------------------------------------
# Conversation utilities
# ---------------------------------------------------------------------------

def reset_conversation_history(user_id: int) -> None:
    CONVERSATIONS.pop(user_id, None)


def append_history(user_id: int, role: str, content: str) -> None:
    if not content:
        return
    history = CONVERSATIONS.setdefault(user_id, [])
    history.append((role, content))
    max_entries = MAX_HISTORY_EXCHANGES * 2
    if len(history) > max_entries:
        history[:] = history[-max_entries:]


def get_history(user_id: int) -> List[Tuple[str, str]]:
    return CONVERSATIONS.get(user_id, [])


def next_missing_field(lead: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    for field, question in FIELD_FLOW:
        if not (lead.get(field) or "").strip():
            return field, question
    return None


def missing_fields(lead: Dict[str, Any]) -> List[str]:
    return [
        field
        for field, _ in FIELD_FLOW
        if not (lead.get(field) or "").strip()
    ]


def validate_answer(field: str, value: str) -> Tuple[bool, Optional[str], str]:
    """Return (is_valid, error_message, cleaned_value)."""
    text = (value or "").strip()
    if not text:
        return False, "I didn't catch anything. Could you type it again?", ""

    if field == "contact":
        if "@" not in text and not text.replace("+", "").replace("-", "").replace(" ", "").isdigit():
            return False, "Please share an email, handle, or phone number so we can follow up.", text

    if field == "budget_range":
        text = text.replace("$", "").strip()

    if field == "timeline":
        text = text.replace("ASAP", "as soon as possible")

    return True, None, text


def field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " ").title())


def format_summary(lead: Dict[str, Any]) -> str:
    lines = [
        "Thanks for sharing! Here's what I captured:",
        f"• Project type: {lead.get('project_type', '–')}",
        f"• Goal: {lead.get('project_goal', '–')}",
        f"• Timeline: {lead.get('timeline', '–')}",
        f"• Budget: {lead.get('budget_range', '–')}",
        f"• Contact: {lead.get('contact', '–')}",
    ]
    if lead.get("notes"):
        lines.append(f"• Notes: {lead['notes']}")
    lines.append("")
    lines.append(f"I'll pass this along to {TEAM_CONTACT}. Feel free to drop extra details or questions anytime.")
    return "\n".join(lines)


def fallback_reply(
    lead: Dict[str, Any],
    next_question: Optional[str],
    event: str,
    captured_field: Optional[Tuple[str, str]],
    error_message: Optional[str],
) -> str:
    if event == "start":
        intro = (
            "Hi! I'm a portfolio demo bot that captures project briefs."
            " I'll ask a few short questions to prep a tailored follow-up."
        )
        if next_question:
            return f"{intro} {next_question}"
        return f"{intro} Let me know what you're building!"

    if event == "invalid":
        parts = []
        if error_message:
            parts.append(error_message)
        if next_question:
            parts.append(next_question)
        return " ".join(parts)

    if event == "ask_next":
        ack = ""
        if captured_field:
            ack = f"Got it on {field_label(captured_field[0])}. "
        question = next_question or "Could you share a bit more detail?"
        return f"{ack}{question}"

    if event == "complete":
        return format_summary(lead)

    if event == "post_complete":
        return (
            f"I already have everything I need. I'll share it with {TEAM_CONTACT}. "
            "Let me know if you want to add anything else."
        )

    summary = ", ".join(field_label(f) for f in missing_fields(lead))
    if summary:
        return f"I still need: {summary}. {next_question or 'Can you share a bit more?'}"
    return next_question or "Could you share a little about your project?"


SYSTEM_PROMPT_TEMPLATE = """You are a friendly, concise discovery assistant for a digital product studio.
Your job is to greet users, gather the required details one question at a time, keep the tone upbeat and professional, and acknowledge any information they provide.
Always keep replies to 1–5 sentences unless the context explicitly asks for more.
Never invent contact details or budgets. If information is missing, ask the next question provided.
When an error message is supplied, apologise briefly and restate the appropriate question.
When the event is "complete" or "post_complete", summarise the captured fields clearly (bullet list welcome) and remind the user that {team_contact} will follow up.
"""


def generate_reply(
    user_id: int,
    lead: Dict[str, Any],
    user_message: Optional[str],
    next_question: Optional[str],
    event: str,
    captured_field: Optional[Tuple[str, str]] = None,
    error_message: Optional[str] = None,
    record_user: bool = True,
) -> str:
    history = get_history(user_id)
    missing = missing_fields(lead)
    snapshot_parts = [
        f"{field_label(f)}: {lead.get(f) or '—'}"
        for f, _ in FIELD_FLOW
    ]
    context_lines = [
        f"Event: {event}",
        f"Team contact: {TEAM_CONTACT}",
        f"Missing fields: {', '.join(missing) if missing else 'none'}",
        f"Pending question: {next_question or 'none'}",
        f"Lead snapshot: {', '.join(snapshot_parts)}",
    ]
    if captured_field:
        context_lines.append(
            f"Just captured: {field_label(captured_field[0])} = {captured_field[1]}"
        )
    if error_message:
        context_lines.append(f"Validation issue: {error_message}")

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(team_contact=TEAM_CONTACT)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": "\n".join(context_lines)},
    ]

    for role, content in history[-MAX_HISTORY_EXCHANGES * 2 :]:
        messages.append({"role": role, "content": content})

    if user_message:
        messages.append({"role": "user", "content": user_message})

    reply_text: Optional[str] = None

    if client:
        prompt_text = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
        try:
            resp = client.responses.create(
                model=OPENAI_MODEL,
                input=prompt_text,
                temperature=0.6,
                max_output_tokens=400,
            )
            reply_text = getattr(resp, "output_text", None)
            if not reply_text:
                collected: List[str] = []
                for block in getattr(resp, "output", []) or []:
                    content = block.get("content")
                    if isinstance(content, list):
                        for piece in content:
                            if isinstance(piece, dict) and "text" in piece:
                                collected.append(str(piece["text"]).strip())
                            elif isinstance(piece, str):
                                collected.append(piece.strip())
                    elif isinstance(content, str):
                        collected.append(content.strip())
                reply_text = "\n".join(x for x in collected if x).strip()
        except Exception as err:
            print(f"[PortfolioBot] Responses API failed: {err}")
            try:
                resp = client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=messages,
                    temperature=0.6,
                    max_tokens=400,
                )
                reply_text = resp.choices[0].message.content if resp.choices else None
            except Exception as err2:
                print(f"[PortfolioBot] Chat Completions fallback failed: {err2}")

    if not reply_text:
        reply_text = fallback_reply(lead, next_question, event, captured_field, error_message)

    reply_text = (reply_text or "").strip()

    if record_user and user_message:
        append_history(user_id, "user", user_message)
    append_history(user_id, "assistant", reply_text)

    return reply_text


# ---------------------------------------------------------------------------
# Telegram handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_user is not None
    user = update.effective_user

    reset_conversation_history(user.id)
    lead = ensure_lead(user.id, user.username, user.first_name, user.language_code)

    if lead.get("status") == "complete":
        reset_lead(user.id)
        lead = fetch_lead(user.id)

    next_step = next_missing_field(lead)
    next_question = None
    if next_step:
        set_pending_field(user.id, next_step[0])
        lead = fetch_lead(user.id)
        next_question = next_step[1]

    reply = generate_reply(
        user_id=user.id,
        lead=lead,
        user_message=None,
        next_question=next_question,
        event="start",
        record_user=False,
    )
    await update.message.reply_text(reply)


async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    assert update.effective_user is not None
    user_id = update.effective_user.id

    reset_lead(user_id)
    reset_conversation_history(user_id)
    lead = fetch_lead(user_id)
    next_step = next_missing_field(lead)
    next_question = None
    if next_step:
        set_pending_field(user_id, next_step[0])
        lead = fetch_lead(user_id)
        next_question = next_step[1]

    reply = generate_reply(
        user_id=user_id,
        lead=lead,
        user_message="I typed /reset. Let's start over.",
        next_question=next_question,
        event="reset",
        record_user=False,
    )
    await update.message.reply_text(reply)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not message or not message.text:
        return

    assert update.effective_user is not None
    user = update.effective_user
    text = message.text.strip()
    lead = ensure_lead(user.id, user.username, user.first_name, user.language_code)

    if text.lower() in {"restart", "reset", "start over"}:
        await reset(update, context)
        return

    pending = lead.get("pending_field")
    captured_field: Optional[Tuple[str, str]] = None
    error_message: Optional[str] = None

    if pending:
        field_question = next((item for item in FIELD_FLOW if item[0] == pending), None)
        if field_question:
            is_valid, error_message, cleaned = validate_answer(pending, text)
            if not is_valid:
                reply = generate_reply(
                    user_id=user.id,
                    lead=lead,
                    user_message=text,
                    next_question=field_question[1],
                    event="invalid",
                    captured_field=None,
                    error_message=error_message,
                )
                await message.reply_text(reply)
                return

            update_lead_field(user.id, pending, cleaned)
            set_pending_field(user.id, None)
            lead = fetch_lead(user.id)
            captured_field = (pending, cleaned)
        else:
            set_pending_field(user.id, None)
            lead = fetch_lead(user.id)

    next_step = next_missing_field(lead)
    next_question = None
    event = "message"

    if next_step:
        set_pending_field(user.id, next_step[0])
        lead = fetch_lead(user.id)
        next_question = next_step[1]
        event = "ask_next"
    else:
        if lead.get("status") != "complete":
            mark_complete(user.id)
            lead = fetch_lead(user.id)
            event = "complete"
        else:
            event = "post_complete"

    reply = generate_reply(
        user_id=user.id,
        lead=lead,
        user_message=text,
        next_question=next_question,
        event=event,
        captured_field=captured_field,
        error_message=error_message,
    )
    await message.reply_text(reply)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Welcome! I'll ask a few short questions about your project so "
        "we can prep a tailored follow-up. Commands available:\n"
        "/start – begin or resume the flow\n"
        "/reset – clear previous answers and restart\n"
        "/help – show this message"
    )
    await update.message.reply_text(text)


def main() -> None:
    init_db()

    application = Application.builder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("reset", reset))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Portfolio Lead Capture Bot running... Press Ctrl+C to stop.")
    application.run_polling()


if __name__ == "__main__":
    main()
