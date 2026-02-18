"""
Telegram Poker AI Concierge Bot — v5
=====================================
Merges the best of v3 (persistent DB handoffs, conversation history, inline
site buttons, /human /bot /id commands, rate-limit cooldown) with v4's
leaner AI flow and slot extraction.

Key improvements over v4:
- Handoff sessions persisted in SQLite (survive restarts)
- Conversation history persisted in SQLite (survive restarts)
- OpenAI call uses chat.completions.create (standard, reliable)
- Rate-limit / quota cooldown (60s on 429, 15min on exhaustion)
- Handoff pitch is smart: only offered once, not on the first greeting
- /human, /bot, /id commands added
- Inline site-selection keyboard buttons
- username + first_name stored in leads table
- Global error handler logs all unhandled exceptions
"""

import contextlib
import json
import logging
import os
import re
import sqlite3
import time
import unicodedata
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AsyncOpenAI
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
load_dotenv()
BOT_TOKEN: str = os.getenv("BOT_TOKEN", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
ADMIN_CHAT_ID: Optional[str] = os.getenv("ADMIN_CHAT_ID")
ADMIN_CHAT_ID_INT: Optional[int] = int(ADMIN_CHAT_ID) if ADMIN_CHAT_ID else None
DEALS_TEAM_NAME: str = os.getenv("DEALS_TEAM_NAME", "Deals & Play")

if not BOT_TOKEN:
    raise SystemExit("ERROR: BOT_TOKEN missing in .env")
if not OPENAI_API_KEY:
    raise SystemExit("ERROR: OPENAI_API_KEY missing in .env")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Database — supports PostgreSQL (Railway) and SQLite (local dev)
# ---------------------------------------------------------------------------
DATABASE_URL: Optional[str] = os.getenv("DATABASE_URL")  # set by Railway automatically
DB_PATH = os.path.join(os.path.dirname(__file__), "leads.db")  # used only for SQLite

# Detect backend
if DATABASE_URL:
    try:
        import psycopg2
        import psycopg2.extras
        _USE_PG = True
        logger.info("[DB] Using PostgreSQL")
    except ImportError:
        _USE_PG = False
        logger.warning("[DB] psycopg2 not installed, falling back to SQLite")
else:
    _USE_PG = False
    logger.info("[DB] Using SQLite at %s", DB_PATH)


@contextlib.contextmanager
def _get_conn() -> Generator:
    """Context manager that yields a DB connection (PG or SQLite)."""
    if _USE_PG:
        conn = psycopg2.connect(DATABASE_URL)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _ph() -> str:
    """Return the right placeholder for the active DB (%s for PG, ? for SQLite)."""
    return "%s" if _USE_PG else "?"


def _serial() -> str:
    """Auto-increment syntax for the active DB."""
    return "SERIAL" if _USE_PG else "INTEGER"


def _conflict_ignore() -> str:
    return "ON CONFLICT DO NOTHING" if _USE_PG else "ON CONFLICT(user_id) DO NOTHING"


def init_db() -> None:
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS leads (
                user_id    BIGINT PRIMARY KEY,
                username   TEXT,
                first_name TEXT,
                language   TEXT DEFAULT 'es',
                country    TEXT,
                site       TEXT,
                stakes     TEXT,
                created_at TEXT,
                updated_at TEXT
            )"""
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id         {'SERIAL' if _USE_PG else 'INTEGER'} PRIMARY KEY {'AUTOINCREMENT' if not _USE_PG else ''},
                user_id    BIGINT NOT NULL,
                role       TEXT   NOT NULL,
                content    TEXT   NOT NULL,
                created_at TEXT   NOT NULL
            )"""
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversation_history(user_id, id)"
        )
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS handoff_sessions (
                user_id           BIGINT PRIMARY KEY,
                started_at        TEXT NOT NULL,
                operator_id       BIGINT,
                operator_username TEXT
            )"""
        )
    # For SQLite: migrate older schema that may be missing columns
    if not _USE_PG:
        _migrate_sqlite_leads()


def _migrate_sqlite_leads() -> None:
    """Add columns introduced in v5 to an existing SQLite leads table."""
    new_columns = [
        ("username", "TEXT"),
        ("first_name", "TEXT"),
        ("language", "TEXT DEFAULT 'es'"),
    ]
    with sqlite3.connect(DB_PATH) as con:
        cur = con.cursor()
        cur.execute("PRAGMA table_info(leads)")
        existing = {row[1] for row in cur.fetchall()}
        for col_name, col_def in new_columns:
            if col_name not in existing:
                con.execute(f"ALTER TABLE leads ADD COLUMN {col_name} {col_def}")
                logger.info("[DB] Migrated SQLite: added column '%s'", col_name)


# --- Leads ---

def upsert_lead(
    user_id: int,
    username: Optional[str] = None,
    first_name: Optional[str] = None,
    language: Optional[str] = None,
    country: Optional[str] = None,
    site: Optional[str] = None,
    stakes: Optional[str] = None,
) -> None:
    now = datetime.utcnow().isoformat()
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(f"SELECT 1 FROM leads WHERE user_id={ph}", (user_id,))
        if cur.fetchone():
            cur.execute(
                f"""UPDATE leads SET
                    username=COALESCE({ph}, username),
                    first_name=COALESCE({ph}, first_name),
                    language=COALESCE({ph}, language),
                    country=COALESCE({ph}, country),
                    site=COALESCE({ph}, site),
                    stakes=COALESCE({ph}, stakes),
                    updated_at={ph}
                   WHERE user_id={ph}""",
                (username, first_name, language, country, site, stakes, now, user_id),
            )
        else:
            cur.execute(
                f"""INSERT INTO leads
                    (user_id, username, first_name, language, country, site, stakes, created_at, updated_at)
                   VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})""",
                (user_id, username, first_name, language or "es", country, site, stakes, now, now),
            )


def get_lead(user_id: int) -> Dict[str, Any]:
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT user_id, username, first_name, language, country, site, stakes FROM leads WHERE user_id={ph}",
            (user_id,),
        )
        row = cur.fetchone()
    if not row:
        return {}
    keys = ["user_id", "username", "first_name", "language", "country", "site", "stakes"]
    return dict(zip(keys, row))


def delete_lead(user_id: int) -> None:
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(f"DELETE FROM leads WHERE user_id={ph}", (user_id,))
        cur.execute(f"DELETE FROM conversation_history WHERE user_id={ph}", (user_id,))


# --- Conversation history ---

MAX_HISTORY = 20


def append_history(user_id: int, role: str, content: str) -> None:
    content = (content or "").strip()
    if not content:
        return
    now = datetime.utcnow().isoformat()
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(
            f"INSERT INTO conversation_history (user_id, role, content, created_at) VALUES ({ph},{ph},{ph},{ph})",
            (user_id, role, content, now),
        )


def fetch_history(user_id: int, limit: int = MAX_HISTORY) -> List[Tuple[str, str]]:
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(
            f"""SELECT role, content FROM conversation_history
               WHERE user_id={ph} ORDER BY id DESC LIMIT {ph}""",
            (user_id, limit),
        )
        rows = cur.fetchall()
    return list(reversed([(r[0], r[1]) for r in rows]))


# --- Handoff sessions ---

def create_handoff_session(user_id: int) -> None:
    now = datetime.utcnow().isoformat()
    ph = _ph()
    ci = _conflict_ignore()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(
            f"""INSERT INTO handoff_sessions (user_id, started_at, operator_id, operator_username)
               VALUES ({ph},{ph},NULL,NULL) {ci}""",
            (user_id, now),
        )


def get_handoff_session(user_id: int) -> Optional[Dict[str, Any]]:
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(
            f"SELECT user_id, started_at, operator_id, operator_username FROM handoff_sessions WHERE user_id={ph}",
            (user_id,),
        )
        row = cur.fetchone()
    if not row:
        return None
    keys = ["user_id", "started_at", "operator_id", "operator_username"]
    return dict(zip(keys, row))


def set_handoff_operator(user_id: int, operator_id: int, operator_username: Optional[str]) -> None:
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(
            f"UPDATE handoff_sessions SET operator_id={ph}, operator_username={ph} WHERE user_id={ph}",
            (operator_id, operator_username, user_id),
        )


def clear_handoff_session(user_id: int) -> None:
    ph = _ph()
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(f"DELETE FROM handoff_sessions WHERE user_id={ph}", (user_id,))


def list_handoff_sessions() -> List[Dict[str, Any]]:
    with _get_conn() as con:
        cur = con.cursor()
        cur.execute(
            "SELECT user_id, started_at, operator_id, operator_username FROM handoff_sessions ORDER BY started_at ASC"
        )
        rows = cur.fetchall()
    keys = ["user_id", "started_at", "operator_id", "operator_username"]
    return [dict(zip(keys, row)) for row in rows]


# ---------------------------------------------------------------------------
# Poker constants
# ---------------------------------------------------------------------------
SITES = ["GGPoker", "PokerStars", "partypoker", "888poker", "CoinPoker", "ACR", "iPoker"]

SITE_ALIASES: Dict[str, str] = {
    "gg": "GGPoker",
    "ggpoker": "GGPoker",
    "gg poker": "GGPoker",
    "stars": "PokerStars",
    "pokerstars": "PokerStars",
    "ps": "PokerStars",
    "party": "partypoker",
    "partypoker": "partypoker",
    "888": "888poker",
    "888poker": "888poker",
    "888 poker": "888poker",
    "coin": "CoinPoker",
    "coinpoker": "CoinPoker",
    "acr": "ACR",
    "acrpoker": "ACR",
    "ipoker": "iPoker",
    "i poker": "iPoker",
}

GREETINGS = {
    "hola", "hol", "hello", "hi", "hey", "buenas", "buenas tardes", "buenos dias",
    "buen día", "que tal", "qué tal", "alo", "ola", "salut", "ciao", "bonjour",
    "bom dia", "boa tarde", "boa noite",
}

BANNED_COUNTRY_TOKENS = {
    "torneo", "torneos", "mtt", "cash", "spins", "spin", "sng", "sit", "go",
    "zoom", "plo", "nlh", "nlhe", "nl", "plo5", "plo4", "holdem", "omaha",
    "poker", "rakeback", "cashback", "deal", "deals",
}

HANDOFF_TRIGGERS = [
    "promo", "promos", "oferta", "ofertas", "deal", "deals", "rakeback",
    "cashback", "referido", "afiliado", "bono", "humano", "human", "persona",
    "hablar con alguien", "talk to",
]

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = f"""Eres un consultor de poker amigable y breve que trabaja para {DEALS_TEAM_NAME}. Hablas en español (1–3 frases), tono cercano. Haz solo una pregunta por mensaje.

Objetivo: entender país, sala y stakes/formato del usuario para conectarlos con el mejor deal de rakeback.

Slots a recopilar: country, site (GGPoker/PokerStars/partypoker/888poker/CoinPoker/ACR/iPoker), stakes (texto libre).

Reglas:
- No envíes links ni códigos de afiliado.
- Si el usuario es menor de 18, termina con cortesía.
- Si ya tienes un slot, no lo vuelvas a pedir.
- Sé natural, no repitas la misma pregunta.

Formato de respuesta:
1) Tu respuesta normal (1-3 frases).
2) Última línea SIEMPRE: <slots>{{"country":..., "site":..., "stakes":...}}</slots> con null si falta el dato.
"""

# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.lower().strip())
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


GREETINGS_NORM = {normalize(g) for g in GREETINGS}


def is_greeting(text: str) -> bool:
    return normalize(text) in GREETINGS_NORM


def looks_like_country(text: str) -> bool:
    norm = normalize(text)
    if not norm or norm in GREETINGS_NORM:
        return False
    if len(norm) < 3 or any(ch.isdigit() for ch in norm):
        return False
    tokens = set(norm.split())
    if tokens & BANNED_COUNTRY_TOKENS:
        return False
    if len(norm.split()) > 5:
        return False
    return True


# Prefixes to strip when extracting a country name from a sentence
_COUNTRY_PREFIXES = re.compile(
    r"^(soy de|vivo en|estoy en|juego desde|desde|de|en)\s+",
    re.IGNORECASE,
)


def extract_country(text: str) -> str:
    """Strip common lead-in phrases and return just the country name."""
    cleaned = _COUNTRY_PREFIXES.sub("", text.strip())
    # Title-case the result so it looks clean in the DB
    return cleaned.strip().title() or text.strip().title()


def detect_site(text: str) -> Optional[str]:
    norm = normalize(text)
    if "888" in text:
        return "888poker"
    for key, val in SITE_ALIASES.items():
        if key in norm:
            return val
    return None


def wants_handoff(text: str) -> bool:
    norm = normalize(text)
    return any(t in norm for t in HANDOFF_TRIGGERS)


def is_yes_like(text: str) -> bool:
    norm = normalize(text)
    return bool(re.search(r"\b(si|ok|vale|dale|claro|yes|adelante|perfecto|listo|va|porfa)\b", norm))


def is_no_like(text: str) -> bool:
    norm = normalize(text)
    patterns = [r"\bno\b", r"\bnah\b", r"\botro momento\b", r"\bluego\b", r"\bmas tarde\b", r"\bno gracias\b"]
    return any(re.search(p, norm) for p in patterns)


def parse_slots(output_text: str) -> Tuple[str, Dict[str, Any]]:
    """Extract <slots>{...}</slots> from AI output. Returns (clean_text, slots_dict)."""
    start_tag, end_tag = "<slots>", "</slots>"
    if start_tag in output_text and end_tag in output_text:
        head, _, rest = output_text.partition(start_tag)
        slot_str, _, tail = rest.partition(end_tag)
        cleaned = (head + tail).strip()
        try:
            slots = json.loads(slot_str.strip())
            if isinstance(slots, dict):
                return cleaned, slots
        except Exception:
            pass
    return output_text.strip(), {}


def is_admin_chat(update: Update) -> bool:
    return (
        ADMIN_CHAT_ID_INT is not None
        and update.effective_chat is not None
        and update.effective_chat.id == ADMIN_CHAT_ID_INT
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
_ai_cooldown_until: float = 0.0


async def ai_reply(
    user_text: str,
    history: List[Tuple[str, str]],
    lead: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    global _ai_cooldown_until

    if time.time() < _ai_cooldown_until:
        remaining = int(_ai_cooldown_until - time.time())
        return (
            f"El asistente se está recuperando ({remaining}s). Dime país, sala y stakes y te ayudo.",
            {},
        )

    # Build messages list
    lead_parts = []
    for k in ("country", "site", "stakes"):
        v = lead.get(k)
        if v:
            lead_parts.append(f"{k}={v}")
    lead_ctx = f"Lead info: {', '.join(lead_parts)}" if lead_parts else "Lead info: none"

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{lead_ctx}"}]
    for role, content in history[-MAX_HISTORY:]:
        # map our roles to OpenAI roles
        oai_role = "assistant" if role == "assistant" else "user"
        messages.append({"role": oai_role, "content": content})
    messages.append({"role": "user", "content": user_text})

    try:
        response = await openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=250,
            temperature=0.7,
        )
        text = (response.choices[0].message.content or "").strip()
        return parse_slots(text)
    except Exception as e:
        err = str(e).lower()
        if "insufficient_quota" in err or "quota" in err:
            _ai_cooldown_until = time.time() + 15 * 60
            logger.warning("[AI] Quota exhausted — cooldown 15 min")
        elif "429" in err or "rate" in err:
            _ai_cooldown_until = time.time() + 60
            logger.warning("[AI] Rate limited — cooldown 60 s")
        else:
            logger.error("[AI] Error: %s", e)
        return ("La IA no está disponible ahora mismo. Dime tu país, sala y stakes.", {})


# ---------------------------------------------------------------------------
# Admin notifications
# ---------------------------------------------------------------------------

async def notify_admin_new_lead(context: ContextTypes.DEFAULT_TYPE, lead: Dict[str, Any]) -> None:
    if not ADMIN_CHAT_ID_INT:
        return
    msg = (
        f"🎯 New lead!\n"
        f"User: @{lead.get('username') or 'unknown'} ({lead.get('first_name') or ''})\n"
        f"Country: {lead.get('country') or 'N/A'}\n"
        f"Site: {lead.get('site') or 'N/A'}\n"
        f"Stakes: {lead.get('stakes') or 'N/A'}"
    )
    try:
        await context.bot.send_message(chat_id=ADMIN_CHAT_ID_INT, text=msg)
    except Exception as exc:
        logger.warning("[ADMIN] Could not notify: %s", exc)


async def escalate_to_human(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    lead: Dict[str, Any],
    reason: str,
    ack_message: Optional[str] = None,
) -> None:
    user = update.effective_user
    if not user or not update.message:
        return

    existing = get_handoff_session(user.id)
    if existing:
        if ack_message:
            await update.message.reply_text(ack_message, disable_web_page_preview=True)
            append_history(user.id, "assistant", ack_message)
        return

    create_handoff_session(user.id)

    if ack_message:
        await update.message.reply_text(ack_message, disable_web_page_preview=True)
        append_history(user.id, "assistant", ack_message)

    if not ADMIN_CHAT_ID_INT:
        fallback = f"El equipo {DEALS_TEAM_NAME} no está disponible ahora, pero anoto tu solicitud."
        await update.message.reply_text(fallback, disable_web_page_preview=True)
        append_history(user.id, "assistant", fallback)
        return

    summary_lines = [
        "📞 Human takeover requested",
        f"Reason: {reason}",
        f"User ID: {user.id}",
        f"Username: @{lead.get('username') or user.username or 'unknown'}",
        f"Name: {lead.get('first_name') or user.first_name or ''}",
        f"Country: {lead.get('country') or 'N/A'}",
        f"Site: {lead.get('site') or 'N/A'}",
        f"Stakes: {lead.get('stakes') or 'N/A'}",
    ]
    try:
        await context.bot.send_message(chat_id=ADMIN_CHAT_ID_INT, text="\n".join(summary_lines))
        if update.message.text:
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID_INT,
                text=f"🗨️ Last message:\n{update.message.text.strip()}",
            )
        await context.bot.send_message(chat_id=ADMIN_CHAT_ID_INT, text=f"/reply {user.id}")
    except Exception as exc:
        logger.warning("[ADMIN] Could not send handoff summary: %s", exc)


async def forward_to_operator(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    lead: Dict[str, Any],
) -> None:
    if not update.message or not update.effective_user:
        return
    user = update.effective_user
    text = update.message.text or ""

    if ADMIN_CHAT_ID_INT:
        header = f"💬 @{lead.get('username') or user.username or 'user'} ({lead.get('first_name') or user.first_name or ''}) | ID: {user.id}"
        payload = f"{header}\n{text}" if text else header
        try:
            await context.bot.send_message(chat_id=ADMIN_CHAT_ID_INT, text=payload)
            await context.bot.send_message(chat_id=ADMIN_CHAT_ID_INT, text=f"/reply {user.id}")
        except Exception as exc:
            logger.warning("[ADMIN] Could not forward message: %s", exc)

    if not context.chat_data.get("handoff_live_ack"):
        ack = f"Paso eso al equipo {DEALS_TEAM_NAME}. Te responden aquí mismo."
        await update.message.reply_text(ack, disable_web_page_preview=True)
        append_history(user.id, "assistant", ack)
        context.chat_data["handoff_live_ack"] = True


# ---------------------------------------------------------------------------
# Site keyboard
# ---------------------------------------------------------------------------

def site_keyboard() -> InlineKeyboardMarkup:
    buttons = [[InlineKeyboardButton(s, callback_data=f"site:{s}")] for s in SITES]
    return InlineKeyboardMarkup(buttons)


async def ask_site(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    reply = "¿En qué sala sueles jugar? Toca un botón o escribe el nombre."
    await update.message.reply_text(reply, reply_markup=site_keyboard())
    append_history(u.id, "assistant", reply)
    context.user_data["awaiting_site"] = True


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    context.user_data.clear()
    context.chat_data.pop("handoff_live_ack", None)
    upsert_lead(u.id, u.username, u.first_name, language="es")
    greet = (
        "👋 ¡Hola! Puedo contarte cómo funciona el rakeback y ver el mejor deal para ti. "
        "¿De qué país juegas?"
    )
    await update.message.reply_text(greet, disable_web_page_preview=True)
    append_history(u.id, "assistant", greet)
    context.user_data["awaiting_country"] = True


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    reply = (
        "/start — iniciar\n"
        "/human — hablar con el equipo\n"
        "/bot — volver al asistente\n"
        "/delete — borrar tus datos\n"
        "/ping — comprobar estado"
    )
    await update.message.reply_text(reply)


async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("pong ✅")


async def delete_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    delete_lead(u.id)
    clear_handoff_session(u.id)
    context.user_data.clear()
    context.chat_data.clear()
    await update.message.reply_text("✅ Tus datos se han borrado. Usa /start cuando quieras.")


async def human_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    append_history(u.id, "user", "/human")
    lead = get_lead(u.id)
    ack = f"Perfecto, aviso al equipo {DEALS_TEAM_NAME}. Responderán aquí mismo en breve."
    await escalate_to_human(update, context, lead, reason="user_command", ack_message=ack)


async def bot_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Return control to the AI bot."""
    u = update.effective_user
    append_history(u.id, "user", "/bot")
    session = get_handoff_session(u.id)
    if not session:
        reply = "Ya estás chateando conmigo. ¿En qué te ayudo?"
        await update.message.reply_text(reply, disable_web_page_preview=True)
        append_history(u.id, "assistant", reply)
        return

    clear_handoff_session(u.id)
    context.chat_data.pop("handoff_live_ack", None)
    reply = "Volvemos al bot. ¡Pregunta lo que necesites!"
    await update.message.reply_text(reply, disable_web_page_preview=True)
    append_history(u.id, "assistant", reply)

    if ADMIN_CHAT_ID_INT:
        try:
            await context.bot.send_message(
                chat_id=ADMIN_CHAT_ID_INT,
                text=f"ℹ️ Handoff closed by user {u.id} (@{u.username or 'unknown'}).",
            )
        except Exception as exc:
            logger.warning("[ADMIN] Could not notify handoff end: %s", exc)


async def id_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Return the current chat ID (useful for setting ADMIN_CHAT_ID)."""
    chat_id = update.effective_chat.id if update.effective_chat else "unknown"
    await update.message.reply_text(f"Chat ID: `{chat_id}`", parse_mode="Markdown")


# ---------------------------------------------------------------------------
# Admin-only commands
# ---------------------------------------------------------------------------

async def reply_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin_chat(update):
        return
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("Uso: /reply <user_id> <mensaje>")
        return
    try:
        user_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("User ID debe ser numérico.")
        return
    message = " ".join(context.args[1:]).strip()
    if not message:
        await update.message.reply_text("Mensaje vacío.")
        return

    create_handoff_session(user_id)
    set_handoff_operator(user_id, update.effective_user.id, update.effective_user.username)
    try:
        await context.bot.send_message(chat_id=user_id, text=message, disable_web_page_preview=True)
        append_history(user_id, "human", message)
        await update.message.reply_text("Enviado ✅")
    except Exception as exc:
        await update.message.reply_text(f"No se pudo enviar: {exc}")


async def release_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin_chat(update):
        return
    if not context.args:
        await update.message.reply_text("Uso: /release <user_id>")
        return
    try:
        user_id = int(context.args[0])
    except ValueError:
        await update.message.reply_text("User ID debe ser numérico.")
        return

    if not get_handoff_session(user_id):
        await update.message.reply_text("No hay handoff activo para ese usuario.")
        return

    clear_handoff_session(user_id)
    try:
        await context.bot.send_message(
            chat_id=user_id,
            text="Todo listo por mi parte. El asistente sigue contigo aquí.",
            disable_web_page_preview=True,
        )
        append_history(user_id, "assistant", "Handoff released; assistant resumed.")
    except Exception as exc:
        logger.warning("[ADMIN] Could not notify user %s of release: %s", user_id, exc)
    await update.message.reply_text("Handoff liberado ✅")


async def handoffs_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_admin_chat(update):
        return
    sessions = list_handoff_sessions()
    if not sessions:
        await update.message.reply_text("No hay handoffs activos.")
        return
    lines = ["📋 Handoffs activos:"]
    for s in sessions:
        op = s.get("operator_username") or s.get("operator_id") or "sin asignar"
        lines.append(f"- User {s['user_id']} · desde {s['started_at']} · operador: {op}")
    await update.message.reply_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Callback: site button
# ---------------------------------------------------------------------------

async def site_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    _, site = query.data.split(":", 1)
    u = update.effective_user
    append_history(u.id, "user", f"[Seleccionó sala: {site}]")
    upsert_lead(u.id, u.username, u.first_name, site=site)
    context.user_data.pop("awaiting_site", None)
    reply = f"✅ Sala: *{site}*. ¿Qué formato o stakes juegas? (ej.: MTT, cash NL50, Spins)"
    await query.edit_message_text(reply, parse_mode="Markdown")
    append_history(u.id, "assistant", reply)


# ---------------------------------------------------------------------------
# Main message router
# ---------------------------------------------------------------------------

async def text_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    u = update.effective_user
    msg = (update.message.text or "").strip()
    if not msg:
        return

    append_history(u.id, "user", msg)

    # Ensure lead row exists
    lead = get_lead(u.id)
    if not lead:
        upsert_lead(u.id, u.username, u.first_name, language="es")
        lead = get_lead(u.id) or {}

    # --- Handoff active: forward to operator ---
    if get_handoff_session(u.id):
        await forward_to_operator(update, context, lead)
        return

    # --- Awaiting handoff confirmation ---
    if context.user_data.get("awaiting_handoff_confirm"):
        if is_yes_like(msg):
            context.user_data.pop("awaiting_handoff_confirm", None)
            context.user_data["offered_handoff"] = True
            ack = f"Listo, te conecto con el equipo {DEALS_TEAM_NAME}. Responderán aquí mismo."
            await escalate_to_human(update, context, lead, reason="user_confirmed", ack_message=ack)
            await notify_admin_new_lead(context, lead)
            return
        if is_no_like(msg):
            context.user_data.pop("awaiting_handoff_confirm", None)
            reply = "Sin problema, seguimos por aquí. ¿En qué más te ayudo?"
            await update.message.reply_text(reply, disable_web_page_preview=True)
            append_history(u.id, "assistant", reply)
            return
        # Not a clear yes/no — fall through to normal flow
        context.user_data.pop("awaiting_handoff_confirm", None)

    # --- Explicit human request ---
    if wants_handoff(msg) and not context.user_data.get("offered_handoff"):
        context.user_data["awaiting_handoff_confirm"] = True
        context.user_data["offered_handoff"] = True
        reply = f"¿Te presento al equipo {DEALS_TEAM_NAME} para que sigan contigo en detalle?"
        await update.message.reply_text(reply, disable_web_page_preview=True)
        append_history(u.id, "assistant", reply)
        return

    # --- Quick slot inference (reduce AI round-trips) ---
    if not lead.get("country") and looks_like_country(msg) and not is_greeting(msg):
        country_name = extract_country(msg)
        upsert_lead(u.id, u.username, u.first_name, country=country_name)
        lead = get_lead(u.id) or {}
        context.user_data.pop("awaiting_country", None)
        ack = f"Perfecto, {country_name} — saber el país ayuda a ajustar las promos."
        await update.message.reply_text(ack, disable_web_page_preview=True)
        append_history(u.id, "assistant", ack)
        await ask_site(update, context)
        return

    if not lead.get("site"):
        site_guess = detect_site(msg)
        if site_guess:
            upsert_lead(u.id, u.username, u.first_name, site=site_guess)
            lead = get_lead(u.id) or {}
            context.user_data.pop("awaiting_site", None)
            reply = f"✅ Sala: {site_guess}. ¿Qué formato o stakes juegas?"
            await update.message.reply_text(reply, disable_web_page_preview=True)
            append_history(u.id, "assistant", reply)
            return

    # --- AI reply ---
    history = fetch_history(u.id)
    reply_text, slots = await ai_reply(msg, history, lead)

    # Apply slots returned by AI
    if slots:
        country_val = slots.get("country")
        site_val = slots.get("site")
        stakes_val = slots.get("stakes")
        if country_val and looks_like_country(str(country_val)):
            upsert_lead(u.id, u.username, u.first_name, country=str(country_val))
        if site_val:
            upsert_lead(u.id, u.username, u.first_name, site=str(site_val))
        if stakes_val:
            upsert_lead(u.id, u.username, u.first_name, stakes=str(stakes_val))
        lead = get_lead(u.id) or {}

    # Offer handoff once all slots are filled and it hasn't been offered yet
    if (
        lead.get("country")
        and lead.get("site")
        and lead.get("stakes")
        and not context.user_data.get("offered_handoff")
        and not get_handoff_session(u.id)
    ):
        context.user_data["awaiting_handoff_confirm"] = True
        context.user_data["offered_handoff"] = True
        pitch = (
            f"Tengo todo lo que necesito. ¿Quieres que te conecte con el equipo "
            f"{DEALS_TEAM_NAME} para revisar el mejor deal de rakeback?"
        )
        await update.message.reply_text(pitch, disable_web_page_preview=True)
        append_history(u.id, "assistant", pitch)
        return

    if not reply_text:
        reply_text = "Cuéntame tu país, sala y formato/stakes y te ayudo con las mejores promos."

    append_history(u.id, "assistant", reply_text)
    await update.message.reply_text(reply_text, disable_web_page_preview=True)


# ---------------------------------------------------------------------------
# Global error handler
# ---------------------------------------------------------------------------

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Unhandled exception: %s", context.error, exc_info=context.error)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    init_db()
    app = Application.builder().token(BOT_TOKEN).build()

    # User commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping))
    app.add_handler(CommandHandler("delete", delete_cmd))
    app.add_handler(CommandHandler("human", human_cmd))
    app.add_handler(CommandHandler("bot", bot_cmd))
    app.add_handler(CommandHandler("id", id_cmd))

    # Admin commands
    app.add_handler(CommandHandler("reply", reply_cmd))
    app.add_handler(CommandHandler("release", release_cmd))
    app.add_handler(CommandHandler("handoffs", handoffs_cmd))

    # Inline buttons
    app.add_handler(CallbackQueryHandler(site_button_handler, pattern=r"^site:"))

    # Text messages
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_router))

    # Error handler
    app.add_error_handler(error_handler)

    logger.info("✅ Bot v5 running — Ctrl+C to stop")
    app.run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
