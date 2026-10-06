import os
import io
import re
import time
import asyncio
import threading
import traceback
import requests
import pandas as pd
import mplfinance as mpf

from flask import Flask
from google import genai
from google.genai import types

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

# ============================================================
# ENV
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN")

if not TWELVEDATA_KEY:
    raise RuntimeError("Missing TWELVEDATA_KEY")

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_KEY")


# ============================================================
# CLIENTS
# ============================================================

client = genai.Client(api_key=GEMINI_KEY)


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "XAU/USD"

MIN_RR = 1.50

PRICE_POLL_SECONDS = 5

MANAGEMENT_REVIEW_SECONDS = 15 * 60

MODIFICATION_REJECTION_SECONDS = 60

MAX_TRADE_SECONDS = 75 * 60

ENTRY_WAIT_SECONDS = 5 * 60

ENTRY_TOLERANCE = 0.60

DEBUG_ENABLED = True


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "RustyGold Bot Live", 200


# ============================================================
# GLOBAL STATE
# ============================================================

analysis_tasks = {}
monitor_tasks = {}

trade_states = {}

proposal_counter = 0

working_gemini_model = None

gemini_models_cache = []
gemini_models_cache_time = 0

GEMINI_MODELS_CACHE_SECONDS = 300


# ============================================================
# DEBUG
# ============================================================

async def send_debug(
    bot,
    chat_id,
    stage,
    error,
    extra=None
):
    if not DEBUG_ENABLED:
        return

    text = (
        "🛠 DEBUG\n\n"
        f"Stage: {stage}\n"
        f"Error: {error}\n"
    )

    if extra:
        text += f"\n{extra}"

    try:
        await bot.send_message(
            chat_id=chat_id,
            text=text[:4000]
        )
    except Exception:
        pass


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def now_ts():
    return time.time()


def fmt_price(value):
    if value is None:
        return "N/A"

    return f"{float(value):.2f}"


# ============================================================
# TELEGRAM KEYBOARD
# ============================================================

def main_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🚀 حلل يا جيمني",
                callback_data="START_ANALYSIS"
            )
        ],
        [
            InlineKeyboardButton(
                "🛑 إيقاف",
                callback_data="STOP_ANALYSIS"
            ),
            InlineKeyboardButton(
                "📊 ملخص",
                callback_data="SUMMARY"
            )
        ]
    ])


def stop_keyboard():
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "🛑 إيقاف",
                callback_data="STOP_ANALYSIS"
            )
        ]
    ])


def rejection_keyboard(proposal_id):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(
                "❌ لا تنفذ التعديل",
                callback_data=f"REJECT:{proposal_id}"
            )
        ]
    ])


# ============================================================
# TWELVEDATA
# ============================================================

def get_ohlc(interval, outputsize=150):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_KEY,
        "format": "JSON"
    }

    response = requests.get(
        url,
        params=params,
        timeout=20
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"TwelveData error: {data}"
        )

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    for column in [
        "open",
        "high",
        "low",
        "close"
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna()

    df = df.sort_values(
        "datetime"
    )

    df = df.set_index("datetime")

    return df


# ============================================================
# GOLD API
# ============================================================

def get_live_gold_price():

    url = "https://api.gold-api.com/price/XAU"

    response = requests.get(
        url,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    for key in [
        "price",
        "Price",
        "value"
    ]:
        if key in data:
            value = safe_float(data[key])

            if value is not None:
                return value

    raise RuntimeError(
        f"Gold API response did not contain price: {data}"
    )


# ============================================================
# CHART
# ============================================================

def make_chart(df, title):

    chart_df = df.copy()

    chart_df = chart_df[
        ["open", "high", "low", "close"]
    ]

    buffer = io.BytesIO()

    mpf.plot(
        chart_df,
        type="candle",
        style="charles",
        title=title,
        volume=False,
        figsize=(12, 7),
        savefig=dict(
            fname=buffer,
            dpi=120,
            bbox_inches="tight"
        )
    )

    buffer.seek(0)

    return buffer.read()


# ============================================================
# GEMINI MODEL DISCOVERY
# ============================================================

def normalize_model_name(name):

    if not name:
        return None

    name = str(name).strip()

    if name.startswith("models/"):
        name = name[len("models/"):]

    return name


def model_priority(name):

    n = name.lower()

    score = 100

    if "flash" in n:
        score -= 30

    if "pro" in n:
        score -= 10

    if "preview" in n:
        score += 10

    if "exp" in n:
        score += 20

    if "embedding" in n:
        score += 1000

    if "image" in n:
        score += 1000

    if "audio" in n:
        score += 1000

    if "tts" in n:
        score += 1000

    return score


def get_gemini_models(force_refresh=False):

    global gemini_models_cache
    global gemini_models_cache_time

    current = time.time()

    if (
        not force_refresh
        and gemini_models_cache
        and current - gemini_models_cache_time
        < GEMINI_MODELS_CACHE_SECONDS
    ):
        return gemini_models_cache

    models = []

    try:
        listed = client.models.list()

        for model in listed:

            name = normalize_model_name(
                getattr(model, "name", None)
            )

            if not name:
                continue

            low = name.lower()

            # Ignore obvious non-text-generation models.
            if any(x in low for x in [
                "embedding",
                "aqa",
                "image",
                "tts",
                "audio"
            ]):
                continue

            if name not in models:
                models.append(name)

    except Exception as e:
        print(
            "Gemini model discovery failed:",
            repr(e)
        )

    models.sort(
        key=model_priority
    )

    gemini_models_cache = models
    gemini_models_cache_time = current

    return models


# ============================================================
# GEMINI - COMPLETELY INDEPENDENT REQUEST
# ============================================================

def ask_gemini_fresh_sync(
    contents,
    debug_label="Gemini"
):
    """
    Every call here is a completely new Gemini request.

    No chat.
    No session.
    No previous messages.
    No conversation history.
    """

    global working_gemini_model

    models = get_gemini_models()

    if not models:
        models = [
            "gemini-3.8-flash",
            "gemini-3.8-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]

    ordered_models = []

    if working_gemini_model:
        ordered_models.append(
            working_gemini_model
        )

    for model in models:

        if model not in ordered_models:
            ordered_models.append(model)

    errors = []

    for model in ordered_models:

        try:

            print(
                f"[GEMINI] Trying model: {model}"
            )

            # IMPORTANT:
            # generate_content is called directly.
            # No client.chats.create()
            # No session.
            # No history.

            response = client.models.generate_content(
                model=model,
                contents=contents
            )

            text = getattr(
                response,
                "text",
                None
            )

            if not text:
                raise RuntimeError(
                    "Gemini returned empty text"
                )

            working_gemini_model = model

            print(
                f"[GEMINI] Success: {model}"
            )

            return text.strip()

        except Exception as e:

            error_text = (
                f"{type(e).__name__}: {e}"
            )

            print(
                f"[GEMINI] FAILED {model}: "
                f"{error_text}"
            )

            errors.append(
                f"{model} -> {error_text}"
            )

            continue

    # Refresh model list and make one more complete attempt.
    refreshed = get_gemini_models(
        force_refresh=True
    )

    for model in refreshed:

        if model in ordered_models:
            continue

        try:

            print(
                f"[GEMINI] Retry model: {model}"
            )

            response = client.models.generate_content(
                model=model,
                contents=contents
            )

            text = getattr(
                response,
                "text",
                None
            )

            if not text:
                raise RuntimeError(
                    "Gemini returned empty text"
                )

            working_gemini_model = model

            return text.strip()

        except Exception as e:

            errors.append(
                f"{model} -> "
                f"{type(e).__name__}: {e}"
            )

    raise RuntimeError(
        f"{debug_label}: all Gemini models failed.\n"
        + "\n".join(errors[-15:])
    )


async def ask_gemini_fresh(
    contents,
    debug_label="Gemini"
):

    return await asyncio.to_thread(
        ask_gemini_fresh_sync,
        contents,
        debug_label
    )


# ============================================================
# GEMINI CONTENT BUILDER
# ============================================================

def build_gemini_contents(
    prompt,
    chart_1h,
    chart_5m,
    chart_1m
):

    contents = [
        prompt,
        types.Part.from_bytes(
            data=chart_1h,
            mime_type="image/png"
        ),
        types.Part.from_bytes(
            data=chart_5m,
            mime_type="image/png"
        ),
        types.Part.from_bytes(
            data=chart_1m,
            mime_type="image/png"
        ),
    ]

    return contents


# ============================================================
# INITIAL ANALYSIS PROMPT
# ============================================================

def build_entry_prompt(
    price,
    df_1h,
    df_5m,
    df_1m
):

    return f"""
You are analyzing XAU/USD for a short-term manual scalping decision.

IMPORTANT:
This is a completely fresh analysis.
Do not assume any previous conversation, previous analysis,
previous trade, previous signal, or previous decision.

Current live XAU/USD price:
{price:.2f}

You have three fresh candlestick charts:
1H
5M
1M

Analyze the actual market structure visible in the data.

Look for:
- BOS
- CHoCH
- liquidity sweep
- false breakout
- displacement
- FVG
- order block
- support/resistance
- trend alignment
- nearby liquidity
- invalidation level

Do NOT invent a trade.

Risk/reward requirement:
Minimum acceptable R:R = 1.50.

Calculate it yourself:

BUY:
risk = ENTRY - SL
reward = TP1 - ENTRY

SELL:
risk = SL - ENTRY
reward = ENTRY - TP1

If R:R is below 1.50:
return NO TRADE.

Do NOT move TP artificially just to reach 1.50.
If the natural setup does not provide at least 1.50 R:R,
return NO TRADE.

The entry must be realistic relative to the current market.

Return ONLY this structure:

RESULT: TRADE
DIRECTION: BUY or SELL
ENTRY: number
SL: number
TP1: number
RR: number
REASON: short explanation

OR:

RESULT: NO TRADE
REASON: short explanation

Do not provide multiple alternative trades.
Do not provide multiple entries.
Do not provide vague possibilities.
Choose one setup or say NO TRADE.
"""


# ============================================================
# PARSE INITIAL TRADE
# ============================================================

def extract_value(text, key):

    pattern = (
        rf"{re.escape(key)}"
        rf"\s*:\s*"
        rf"([-+]?\d+(?:\.\d+)?)"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    return safe_float(
        match.group(1)
    )


def parse_trade(text):

    if not text:
        return None

    upper = text.upper()

    if "NO TRADE" in upper:
        return None

    if "RESULT: TRADE" not in upper:
        return None

    direction_match = re.search(
        r"DIRECTION\s*:\s*(BUY|SELL)",
        upper
    )

    if not direction_match:
        return None

    direction = (
        direction_match.group(1)
    )

    entry = extract_value(
        text,
        "ENTRY"
    )

    sl = extract_value(
        text,
        "SL"
    )

    tp = extract_value(
        text,
        "TP1"
    )

    if (
        entry is None
        or sl is None
        or tp is None
    ):
        return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "original_sl": sl,
        "original_tp": tp,
    }


# ============================================================
# RR
# ============================================================

def calculate_rr(
    direction,
    entry,
    sl,
    tp
):

    if direction == "BUY":

        risk = entry - sl
        reward = tp - entry

    else:

        risk = sl - entry
        reward = entry - tp

    if risk <= 0:
        return None

    if reward <= 0:
        return None

    return reward / risk


def validate_trade(trade):

    direction = trade["direction"]

    entry = trade["entry"]

    sl = trade["sl"]

    tp = trade["tp"]

    if direction == "BUY":

        if not sl < entry:
            return False, "BUY SL must be below ENTRY"

        if not tp > entry:
            return False, "BUY TP must be above ENTRY"

    elif direction == "SELL":

        if not sl > entry:
            return False, "SELL SL must be above ENTRY"

        if not tp < entry:
            return False, "SELL TP must be below ENTRY"

    else:

        return False, "Invalid direction"

    rr = calculate_rr(
        direction,
        entry,
        sl,
        tp
    )

    if rr is None:
        return False, "Invalid risk/reward"

    if rr < MIN_RR:
        return (
            False,
            f"R:R {rr:.2f} below {MIN_RR:.2f}"
        )

    trade["rr"] = rr

    return True, "OK"


# ============================================================
# MANAGEMENT PROMPT
# ============================================================

def build_management_prompt(
    state,
    current_price,
    df_1h,
    df_5m,
    df_1m
):

    age_minutes = (
        time.time() - state["opened_at"]
    ) / 60

    return f"""
You are reviewing an EXISTING OPEN XAU/USD trade.

THIS IS NOT A NEW ENTRY ANALYSIS.

This is a completely fresh and independent Gemini request.
You must NOT rely on any previous Gemini answer or conversation.

The trade already exists.

ENTRY IS FIXED:
{state["entry_price"]:.2f}

Direction:
{state["direction"]}

Current price:
{current_price:.2f}

Current SL:
{state["sl"]:.2f}

Current TP:
{state["tp"]:.2f}

Original SL:
{state["original_sl"]:.2f}

Original TP:
{state["original_tp"]:.2f}

Trade age:
{age_minutes:.1f} minutes

Fresh market data:
- 1H
- 5M
- 1M

Your job is ONLY to manage this existing trade.

DO NOT create a new entry.

DO NOT change ENTRY.

ENTRY remains:
{state["entry_price"]:.2f}

You may decide:

KEEP
MOVE_SL
MOVE_TP
MOVE_SL_AND_TP
CLOSE_PROFIT
CLOSE_LOSS
CLOSE_NOW

Consider:
- current market structure
- BOS
- CHoCH
- liquidity
- momentum
- FVG
- order blocks
- invalidation
- whether the current SL is still logical
- whether the current TP is still realistic
- whether profit should be protected
- whether the setup has become invalid
- trade age

If the trade is already profitable, you may move SL beyond ENTRY
to protect profit if market structure supports it.

If the trade is losing and structure is invalidated,
you may recommend closing.

The maximum allowed trade duration is 75 minutes.
If the trade reaches 60-75 minutes, strongly consider closing
unless there is a strong reason to keep it.

IMPORTANT:
Do not invent a new trade.
Do not change ENTRY.
Only manage the existing trade.

Return ONLY:

ACTION: KEEP
NEW_SL: SAME
NEW_TP: SAME
REASON: short explanation

OR:

ACTION: MOVE_SL
NEW_SL: number
NEW_TP: SAME
REASON: short explanation

OR:

ACTION: MOVE_TP
NEW_SL: SAME
NEW_TP: number
REASON: short explanation

OR:

ACTION: MOVE_SL_AND_TP
NEW_SL: number
NEW_TP: number
REASON: short explanation

OR:

ACTION: CLOSE_PROFIT
NEW_SL: SAME
NEW_TP: SAME
REASON: short explanation

OR:

ACTION: CLOSE_LOSS
NEW_SL: SAME
NEW_TP: SAME
REASON: short explanation

OR:

ACTION: CLOSE_NOW
NEW_SL: SAME
NEW_TP: SAME
REASON: short explanation
"""


# ============================================================
# PARSE MANAGEMENT
# ============================================================

def extract_management_value(
    text,
    key
):

    match = re.search(
        rf"{re.escape(key)}\s*:\s*([^\n\r]+)",
        text,
        re.IGNORECASE
    )

    if not match:
        return None

    value = match.group(1).strip()

    if value.upper() == "SAME":
        return None

    value = value.replace(",", "")

    return safe_float(value)


def parse_management(text):

    if not text:
        return None

    action_match = re.search(
        r"ACTION\s*:\s*"
        r"(KEEP|MOVE_SL|MOVE_TP|MOVE_SL_AND_TP|"
        r"CLOSE_PROFIT|CLOSE_LOSS|CLOSE_NOW)",
        text,
        re.IGNORECASE
    )

    if not action_match:
        return None

    action = (
        action_match.group(1)
        .upper()
    )

    new_sl = extract_management_value(
        text,
        "NEW_SL"
    )

    new_tp = extract_management_value(
        text,
        "NEW_TP"
    )

    reason_match = re.search(
        r"REASON\s*:\s*(.+)",
        text,
        re.IGNORECASE
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else "No reason provided"
    )

    return {
        "action": action,
        "new_sl": new_sl,
        "new_tp": new_tp,
        "reason": reason
    }


# ============================================================
# MANAGEMENT VALIDATION
# ============================================================

def validate_management(
    state,
    current_price,
    result
):

    action = result["action"]

    if action in [
        "KEEP",
        "CLOSE_PROFIT",
        "CLOSE_LOSS",
        "CLOSE_NOW"
    ]:
        return True, "OK"

    new_sl = (
        result["new_sl"]
        if result["new_sl"] is not None
        else state["sl"]
    )

    new_tp = (
        result["new_tp"]
        if result["new_tp"] is not None
        else state["tp"]
    )

    direction = state["direction"]

    if direction == "BUY":

        # SL must remain below current price.
        if new_sl >= current_price:
            return (
                False,
                "BUY SL must remain below current price"
            )

        # TP must remain above current price.
        if new_tp <= current_price:
            return (
                False,
                "BUY TP must remain above current price"
            )

    else:

        # SL must remain above current price.
        if new_sl <= current_price:
            return (
                False,
                "SELL SL must remain above current price"
            )

        # TP must remain below current price.
        if new_tp >= current_price:
            return (
                False,
                "SELL TP must remain below current price"
            )

    return True, "OK"


# ============================================================
# PNL
# ============================================================

def calculate_pnl(
    state,
    exit_price
):

    entry = state["entry_price"]

    if state["direction"] == "BUY":
        return exit_price - entry

    return entry - exit_price


# ============================================================
# FORMAT TRADE
# ============================================================

def trade_text(
    trade,
    current_price=None
):

    text = (
        "🎯 صفقة مقترحة\n\n"
        f"الاتجاه: {trade['direction']}\n"
        f"ENTRY: {fmt_price(trade['entry'])}\n"
        f"SL: {fmt_price(trade['sl'])}\n"
        f"TP1: {fmt_price(trade['tp'])}\n"
        f"R:R: {trade['rr']:.2f}\n"
    )

    if current_price is not None:
        text += (
            f"\nالسعر الحالي: "
            f"{fmt_price(current_price)}"
        )

    return text


# ============================================================
# WAIT FOR ENTRY
# ============================================================

async def wait_for_entry(
    bot,
    chat_id,
    trade
):

    start = time.time()

    direction = trade["direction"]

    target = trade["entry"]

    while (
        time.time() - start
        < ENTRY_WAIT_SECONDS
    ):

        try:

            price = await asyncio.to_thread(
                get_live_gold_price
            )

            if direction == "BUY":

                if price >= target - ENTRY_TOLERANCE:

                    return price

            else:

                if price <= target + ENTRY_TOLERANCE:

                    return price

        except Exception as e:

            await send_debug(
                bot,
                chat_id,
                "WAIT_ENTRY",
                repr(e)
            )

        await asyncio.sleep(
            PRICE_POLL_SECONDS
        )

    return None


# ============================================================
# CREATE MANAGEMENT PROPOSAL
# ============================================================

async def create_management_proposal(
    bot,
    chat_id,
    state,
    result
):

    global proposal_counter

    proposal_counter += 1

    proposal_id = proposal_counter

    state["pending_proposal"] = {
        "id": proposal_id,
        "created_at": time.time(),
        "result": result,
    }

    action = result["action"]

    new_sl = (
        result["new_sl"]
        if result["new_sl"] is not None
        else state["sl"]
    )

    new_tp = (
        result["new_tp"]
        if result["new_tp"] is not None
        else state["tp"]
    )

    message = (
        "⚠️ اقتراح تعديل الصفقة\n\n"
        f"Action: {action}\n"
        f"ENTRY: {fmt_price(state['entry_price'])}\n"
        f"SL الحالي: {fmt_price(state['sl'])}\n"
        f"TP الحالي: {fmt_price(state['tp'])}\n\n"
        f"SL المقترح: {fmt_price(new_sl)}\n"
        f"TP المقترح: {fmt_price(new_tp)}\n\n"
        f"السبب:\n{result['reason']}\n\n"
        "⏳ سيتم تنفيذ التعديل تلقائيًا بعد 60 ثانية "
        "إذا لم تضغط الرفض."
    )

    try:

        sent = await bot.send_message(
            chat_id=chat_id,
            text=message,
            reply_markup=rejection_keyboard(
                proposal_id
            )
        )

        state["pending_proposal"]["message_id"] = (
            sent.message_id
        )

        asyncio.create_task(
            finalize_management_proposal(
                bot,
                chat_id,
                state,
                proposal_id
            )
        )

    except Exception as e:

        await send_debug(
            bot,
            chat_id,
            "CREATE_PROPOSAL",
            repr(e)
        )


# ============================================================
# FINALIZE PROPOSAL
# ============================================================

async def finalize_management_proposal(
    bot,
    chat_id,
    state,
    proposal_id
):

    await asyncio.sleep(
        MODIFICATION_REJECTION_SECONDS
    )

    pending = state.get(
        "pending_proposal"
    )

    if not pending:
        return

    if pending["id"] != proposal_id:
        return

    result = pending["result"]

    # The proposal expired exactly now.
    state["pending_proposal"] = None

    action = result["action"]

    if action in [
        "KEEP",
        "CLOSE_PROFIT",
        "CLOSE_LOSS",
        "CLOSE_NOW"
    ]:
        return

    new_sl = (
        result["new_sl"]
        if result["new_sl"] is not None
        else state["sl"]
    )

    new_tp = (
        result["new_tp"]
        if result["new_tp"] is not None
        else state["tp"]
    )

    # Check current price BEFORE applying.
    try:

        current_price = await asyncio.to_thread(
            get_live_gold_price
        )

    except Exception as e:

        await send_debug(
            bot,
            chat_id,
            "FINALIZE_PRICE",
            repr(e)
        )

        return

    # If price already hit old SL/TP,
    # the old levels win.
    if trade_hit_sl_or_tp(
        state,
        current_price
    ):

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ انتهت الصفقة قبل تنفيذ التعديل.\n"
                f"السعر الحالي: {fmt_price(current_price)}"
            )
        )

        return

    valid, reason = validate_management(
        state,
        current_price,
        result
    )

    if not valid:

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ لم يتم تنفيذ التعديل.\n"
                f"السبب: {reason}"
            )
        )

        return

    state["sl"] = new_sl
    state["tp"] = new_tp

    await bot.send_message(
        chat_id=chat_id,
        text=(
            "✅ تم تنفيذ تعديل الصفقة تلقائيًا\n\n"
            f"SL الجديد: {fmt_price(new_sl)}\n"
            f"TP الجديد: {fmt_price(new_tp)}\n\n"
            f"السبب:\n{result['reason']}"
        )
    )


# ============================================================
# REJECTION
# ============================================================

async def reject_proposal(
    query,
    proposal_id
):

    chat_id = query.message.chat_id

    state = trade_states.get(
        chat_id
    )

    if not state:
        await query.answer(
            "لا توجد صفقة مفتوحة.",
            show_alert=True
        )
        return

    pending = state.get(
        "pending_proposal"
    )

    if not pending:
        await query.answer(
            "انتهت صلاحية الاقتراح.",
            show_alert=True
        )
        return

    if pending["id"] != proposal_id:

        await query.answer(
            "هذا الاقتراح قديم.",
            show_alert=True
        )

        return

    age = (
        time.time()
        - pending["created_at"]
    )

    if age >= MODIFICATION_REJECTION_SECONDS:

        state["pending_proposal"] = None

        await query.answer(
            "انتهت صلاحية الاقتراح.",
            show_alert=True
        )

        try:
            await query.edit_message_reply_markup(
                reply_markup=None
            )
        except Exception:
            pass

        return

    state["pending_proposal"] = None

    try:
        await query.edit_message_reply_markup(
            reply_markup=None
        )
    except Exception:
        pass

    await query.answer(
        "تم رفض التعديل.",
        show_alert=True
    )

    await query.message.reply_text(
        "❌ تم رفض التعديل.\n"
        "تم الإبقاء على SL و TP الحاليين."
    )


# ============================================================
# HIT CHECK
# ============================================================

def trade_hit_sl_or_tp(
    state,
    price
):

    direction = state["direction"]

    sl = state["sl"]

    tp = state["tp"]

    if direction == "BUY":

        if price <= sl:
            return "SL"

        if price >= tp:
            return "TP"

    else:

        if price >= sl:
            return "SL"

        if price <= tp:
            return "TP"

    return None


# ============================================================
# CLOSE TRADE
# ============================================================

async def close_trade(
    bot,
    chat_id,
    state,
    exit_price,
    reason
):

    pnl = calculate_pnl(
        state,
        exit_price
    )

    duration_minutes = (
        time.time() - state["opened_at"]
    ) / 60

    state["closed"] = True

    state["exit_price"] = exit_price

    state["pnl"] = pnl

    state["close_reason"] = reason

    pending = state.get(
        "pending_proposal"
    )

    state["pending_proposal"] = None

    text = (
        "🏁 إغلاق الصفقة\n\n"
        f"الاتجاه: {state['direction']}\n"
        f"ENTRY الفعلي: {fmt_price(state['entry_price'])}\n"
        f"EXIT الفعلي: {fmt_price(exit_price)}\n"
        f"SL الأخير: {fmt_price(state['sl'])}\n"
        f"TP الأخير: {fmt_price(state['tp'])}\n\n"
        f"النتيجة: {pnl:+.2f}\n"
        f"المدة: {duration_minutes:.1f} دقيقة\n"
        f"السبب: {reason}"
    )

    await bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=main_keyboard()
    )

    return pnl


# ============================================================
# MANAGEMENT REVIEW
# ============================================================

async def perform_management_review(
    bot,
    chat_id,
    state
):

    try:

        current_price = await asyncio.to_thread(
            get_live_gold_price
        )

        df_1h = await asyncio.to_thread(
            get_ohlc,
            "1h",
            120
        )

        df_5m = await asyncio.to_thread(
            get_ohlc,
            "5min",
            150
        )

        df_1m = await asyncio.to_thread(
            get_ohlc,
            "1min",
            150
        )

        chart_1h = await asyncio.to_thread(
            make_chart,
            df_1h,
            "XAU/USD 1H"
        )

        chart_5m = await asyncio.to_thread(
            make_chart,
            df_5m,
            "XAU/USD 5M"
        )

        chart_1m = await asyncio.to_thread(
            make_chart,
            df_1m,
            "XAU/USD 1M"
        )

        prompt = build_management_prompt(
            state,
            current_price,
            df_1h,
            df_5m,
            df_1m
        )

        contents = build_gemini_contents(
            prompt,
            chart_1h,
            chart_5m,
            chart_1m
        )

        # Completely fresh Gemini analysis.
        response = await ask_gemini_fresh(
            contents,
            "Management review"
        )

        result = parse_management(
            response
        )

        if not result:

            await send_debug(
                bot,
                chat_id,
                "MANAGEMENT_PARSE",
                "Could not parse Gemini response",
                response[:3000]
            )

            return

        action = result["action"]

        # Direct close decisions.
        if action in [
            "CLOSE_PROFIT",
            "CLOSE_LOSS",
            "CLOSE_NOW"
        ]:

            await close_trade(
                bot,
                chat_id,
                state,
                current_price,
                (
                    f"Gemini: {action} - "
                    f"{result['reason']}"
                )
            )

            return

        # KEEP
        if action == "KEEP":

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "🔎 مراجعة Gemini للصفقة\n\n"
                    "✅ KEEP\n"
                    f"السعر: {fmt_price(current_price)}\n"
                    f"SL: {fmt_price(state['sl'])}\n"
                    f"TP: {fmt_price(state['tp'])}\n\n"
                    f"{result['reason']}"
                ),
                reply_markup=stop_keyboard()
            )

            return

        valid, reason = validate_management(
            state,
            current_price,
            result
        )

        if not valid:

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ Gemini اقترح تعديلًا "
                    "لكن البوت رفضه للحماية.\n\n"
                    f"السبب: {reason}"
                )
            )

            return

        await create_management_proposal(
            bot,
            chat_id,
            state,
            result
        )

    except Exception as e:

        await send_debug(
            bot,
            chat_id,
            "MANAGEMENT_REVIEW",
            repr(e),
            traceback.format_exc()[-3000:]
        )


# ============================================================
# TRADE MONITOR
# ============================================================

async def monitor_trade(
    bot,
    chat_id,
    state
):

    last_management_review = time.time()

    while not state.get("closed"):

        try:

            current_price = await asyncio.to_thread(
                get_live_gold_price
            )

            # ------------------------------------------------
            # SL / TP are checked independently of Gemini.
            # ------------------------------------------------

            hit = trade_hit_sl_or_tp(
                state,
                current_price
            )

            if hit:

                await close_trade(
                    bot,
                    chat_id,
                    state,
                    current_price,
                    f"Price hit {hit}"
                )

                break

            # ------------------------------------------------
            # Maximum trade duration.
            # ------------------------------------------------

            age = (
                time.time()
                - state["opened_at"]
            )

            if age >= MAX_TRADE_SECONDS:

                await close_trade(
                    bot,
                    chat_id,
                    state,
                    current_price,
                    "Maximum trade duration reached"
                )

                break

            # ------------------------------------------------
            # Gemini management every 15 minutes.
            # ------------------------------------------------

            if (
                time.time()
                - last_management_review
                >= MANAGEMENT_REVIEW_SECONDS
            ):

                # Do not run another proposal review
                # while a previous proposal is waiting.
                if not state.get(
                    "pending_proposal"
                ):

                    last_management_review = (
                        time.time()
                    )

                    await perform_management_review(
                        bot,
                        chat_id,
                        state
                    )

        except Exception as e:

            await send_debug(
                bot,
                chat_id,
                "TRADE_MONITOR",
                repr(e)
            )

        await asyncio.sleep(
            PRICE_POLL_SECONDS
        )


# ============================================================
# MAIN ANALYSIS LOOP
# ============================================================

async def analysis_loop(
    bot,
    chat_id
):

    try:

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "🔎 جاري تحليل XAU/USD...\n"
                "تحليل Gemini جديد ومستقل."
            ),
            reply_markup=stop_keyboard()
        )

        # ----------------------------------------------------
        # Fresh market data
        # ----------------------------------------------------

        price = await asyncio.to_thread(
            get_live_gold_price
        )

        df_1h = await asyncio.to_thread(
            get_ohlc,
            "1h",
            120
        )

        df_5m = await asyncio.to_thread(
            get_ohlc,
            "5min",
            150
        )

        df_1m = await asyncio.to_thread(
            get_ohlc,
            "1min",
            150
        )

        chart_1h = await asyncio.to_thread(
            make_chart,
            df_1h,
            "XAU/USD 1H"
        )

        chart_5m = await asyncio.to_thread(
            make_chart,
            df_5m,
            "XAU/USD 5M"
        )

        chart_1m = await asyncio.to_thread(
            make_chart,
            df_1m,
            "XAU/USD 1M"
        )

        # ----------------------------------------------------
        # Completely fresh Gemini analysis.
        # ----------------------------------------------------

        prompt = build_entry_prompt(
            price,
            df_1h,
            df_5m,
            df_1m
        )

        contents = build_gemini_contents(
            prompt,
            chart_1h,
            chart_5m,
            chart_1m
        )

        response = await ask_gemini_fresh(
            contents,
            "Entry analysis"
        )

        trade = parse_trade(
            response
        )

        if not trade:

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "🚫 لا توجد صفقة مناسبة الآن.\n\n"
                    f"Gemini:\n{response[:3000]}"
                ),
                reply_markup=main_keyboard()
            )

            return

        # ----------------------------------------------------
        # Independent RR protection.
        # ----------------------------------------------------

        valid, reason = validate_trade(
            trade
        )

        if not valid:

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "🚫 تم رفض الصفقة من فلتر البوت.\n\n"
                    f"السبب: {reason}\n\n"
                    "البوت لا يغير TP لإجبار الصفقة "
                    "على تحقيق R:R."
                ),
                reply_markup=main_keyboard()
            )

            return

        # ----------------------------------------------------
        # Show accepted idea.
        # ----------------------------------------------------

        await bot.send_message(
            chat_id=chat_id,
            text=(
                trade_text(
                    trade,
                    price
                )
                + "\n\n"
                "⏳ بانتظار وصول السعر إلى منطقة ENTRY..."
            ),
            reply_markup=stop_keyboard()
        )

        # ----------------------------------------------------
        # Wait for actual entry.
        # ----------------------------------------------------

        actual_entry = await wait_for_entry(
            bot,
            chat_id,
            trade
        )

        if actual_entry is None:

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⌛ انتهت مدة انتظار الدخول.\n"
                    "لم يتم فتح الصفقة."
                ),
                reply_markup=main_keyboard()
            )

            return

        # ----------------------------------------------------
        # Create actual open trade state.
        # ----------------------------------------------------

        state = {
            "direction": trade["direction"],

            "entry_price": actual_entry,

            "sl": trade["sl"],

            "tp": trade["tp"],

            "original_sl": trade["original_sl"],

            "original_tp": trade["original_tp"],

            "opened_at": time.time(),

            "closed": False,

            "pending_proposal": None,

            "rr": trade["rr"],

        }

        trade_states[chat_id] = state

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "🟢 تم اعتبار الصفقة مفتوحة\n\n"
                f"Direction: {state['direction']}\n"
                f"ENTRY الفعلي: {fmt_price(actual_entry)}\n"
                f"SL: {fmt_price(state['sl'])}\n"
                f"TP: {fmt_price(state['tp'])}\n\n"
                "📡 مراقبة السعر كل 5 ثوانٍ.\n"
                "🧠 مراجعة Gemini كل 15 دقيقة."
            ),
            reply_markup=stop_keyboard()
        )

        # ----------------------------------------------------
        # Monitor independently.
        # ----------------------------------------------------

        monitor_tasks[chat_id] = asyncio.create_task(
            monitor_trade(
                bot,
                chat_id,
                state
            )
        )

        await monitor_tasks[chat_id]

    except asyncio.CancelledError:

        try:
            await bot.send_message(
                chat_id=chat_id,
                text="🛑 تم إيقاف التحليل."
            )
        except Exception:
            pass

        raise

    except Exception as e:

        await send_debug(
            bot,
            chat_id,
            "ANALYSIS_LOOP",
            repr(e),
            traceback.format_exc()[-3500:]
        )

        try:

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ حدث خطأ أثناء التحليل.\n\n"
                    "تم إرسال تفاصيل الخطأ إلى هذه الدردشة."
                ),
                reply_markup=main_keyboard()
            )

        except Exception:
            pass

    finally:

        analysis_tasks.pop(
            chat_id,
            None
        )


# ============================================================
# /START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    chat_id = update.effective_chat.id

    await update.message.reply_text(
        "RustyGold جاهز 🟢\n\n"
        "تحليل XAU/USD مع Gemini\n"
        "ومراقبة مستقلة للسعر.",
        reply_markup=main_keyboard()
    )


# ============================================================
# SUMMARY
# ============================================================

async def send_summary(
    query
):

    chat_id = query.message.chat_id

    state = trade_states.get(
        chat_id
    )

    if not state:

        await query.message.reply_text(
            "📊 لا توجد صفقة حالية."
        )

        return

    if state.get("closed"):

        await query.message.reply_text(
            "📊 آخر صفقة مغلقة\n\n"
            f"ENTRY: {fmt_price(state['entry_price'])}\n"
            f"EXIT: {fmt_price(state['exit_price'])}\n"
            f"PnL: {state['pnl']:+.2f}\n"
            f"السبب: {state['close_reason']}"
        )

        return

    age = (
        time.time()
        - state["opened_at"]
    ) / 60

    await query.message.reply_text(
        "📊 الصفقة الحالية\n\n"
        f"Direction: {state['direction']}\n"
        f"ENTRY: {fmt_price(state['entry_price'])}\n"
        f"SL: {fmt_price(state['sl'])}\n"
        f"TP: {fmt_price(state['tp'])}\n"
        f"العمر: {age:.1f} دقيقة\n"
        f"Gemini model: "
        f"{working_gemini_model or 'unknown'}"
    )


# ============================================================
# CALLBACK HANDLER
# ============================================================

async def callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    chat_id = query.message.chat_id

    data = query.data

    # --------------------------------------------------------
    # START ANALYSIS
    # --------------------------------------------------------

    if data == "START_ANALYSIS":

        existing_task = analysis_tasks.get(
            chat_id
        )

        if existing_task and not existing_task.done():

            await query.message.reply_text(
                "⚠️ يوجد تحليل يعمل بالفعل."
            )

            return

        state = trade_states.get(
            chat_id
        )

        if state and not state.get("closed"):

            await query.message.reply_text(
                "⚠️ توجد صفقة مفتوحة حاليًا.\n"
                "لا يمكن بدء تحليل دخول جديد."
            )

            return

        task = asyncio.create_task(
            analysis_loop(
                context.bot,
                chat_id
            )
        )

        analysis_tasks[chat_id] = task

        return

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if data == "STOP_ANALYSIS":

        task = analysis_tasks.get(
            chat_id
        )

        if task and not task.done():

            task.cancel()

        monitor = monitor_tasks.get(
            chat_id
        )

        if monitor and not monitor.done():

            monitor.cancel()

        await query.message.reply_text(
            "🛑 تم إرسال أمر الإيقاف."
        )

        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if data == "SUMMARY":

        await send_summary(
            query
        )

        return

    # --------------------------------------------------------
    # REJECT PROPOSAL
    # --------------------------------------------------------

    if data.startswith("REJECT:"):

        raw_id = data.split(
            ":",
            1
        )[1]

        try:
            proposal_id = int(raw_id)
        except Exception:
            await query.answer(
                "Proposal ID غير صالح.",
                show_alert=True
            )
            return

        await reject_proposal(
            query,
            proposal_id
        )

        return


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update,
    context
):

    print(
        "Telegram error:",
        repr(context.error)
    )

    try:

        if update and update.effective_chat:

            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=(
                    "🛠 حدث خطأ داخلي.\n"
                    f"{type(context.error).__name__}: "
                    f"{context.error}"
                )
            )

    except Exception:
        pass


# ============================================================
# TELEGRAM APP
# ============================================================

def create_bot():

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start_command
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            callback_handler
        )
    )

    application.add_error_handler(
        error_handler
    )

    return application


# ============================================================
# RUN FLASK
# ============================================================

def run_flask():

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("RustyGold starting...")

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    application = create_bot()

    print(
        "RustyGold Telegram bot started."
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()