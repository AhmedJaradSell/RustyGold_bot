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


# =========================================================
# ENVIRONMENT
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")


if not TELEGRAM_TOKEN:
    raise RuntimeError("Missing TELEGRAM_TOKEN")

if not TWELVEDATA_KEY:
    raise RuntimeError("Missing TWELVEDATA_KEY")

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_KEY")


# =========================================================
# GEMINI
# =========================================================

client = genai.Client(api_key=GEMINI_KEY)


# =========================================================
# CONFIG
# =========================================================

SYMBOL = "XAU/USD"

# New minimum R:R
MIN_RR = 1.25

# Price monitoring
PRICE_POLL_SECONDS = 5

# Gemini management review
MANAGEMENT_REVIEW_SECONDS = 15 * 60

# Pending management proposal
MODIFICATION_REJECTION_SECONDS = 60

# Maximum trade duration
MAX_TRADE_SECONDS = 75 * 60

# Maximum time waiting for entry
ENTRY_WAIT_SECONDS = 10 * 60

# Wait before completely fresh analysis
REANALYSIS_WAIT_SECONDS = 5 * 60

# Small tolerance around the proposed entry
ENTRY_TOLERANCE = 0.60

# Used only to avoid hammering APIs when a price is requested repeatedly
GEMINI_CACHE_SECONDS = 300


# =========================================================
# GLOBAL STATE
# =========================================================

analysis_tasks = {}
monitor_tasks = {}
trade_states = {}

proposal_counter = 0

working_gemini_model = None
gemini_models_cache = []
gemini_models_cache_time = 0


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "RustyGold Bot Live", 200


def run_flask():
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000")),
    )


# =========================================================
# TELEGRAM KEYBOARDS
# =========================================================

def main_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🚀 حلل يا جيمني",
                    callback_data="START_ANALYSIS",
                )
            ],
            [
                InlineKeyboardButton(
                    "🛑 إيقاف",
                    callback_data="STOP_ANALYSIS",
                ),
                InlineKeyboardButton(
                    "📊 ملخص",
                    callback_data="SUMMARY",
                ),
            ],
        ]
    )


def stop_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🛑 إيقاف",
                    callback_data="STOP_ANALYSIS",
                )
            ]
        ]
    )


def rejection_keyboard(proposal_id):
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "❌ لا تنفذ التعديل",
                    callback_data=f"REJECT:{proposal_id}",
                )
            ],
            [
                InlineKeyboardButton(
                    "🛑 إيقاف",
                    callback_data="STOP_ANALYSIS",
                )
            ],
        ]
    )


# =========================================================
# TWELVEDATA
# =========================================================

def get_candles(interval, outputsize=120):

    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": SYMBOL,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"TwelveData error: {data}"
        )

    df = pd.DataFrame(data["values"])

    if df.empty:
        raise RuntimeError(
            f"No candles returned for {interval}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    for col in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    df = df.dropna()

    df = df.sort_values(
        "datetime"
    )

    df = df.set_index(
        "datetime"
    )

    return df


# =========================================================
# GOLD API
# =========================================================

def get_live_gold_price():

    url = "https://api.gold-api.com/price/XAU"

    response = requests.get(
        url,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    possible_keys = [
        "price",
        "Price",
        "value",
    ]

    for key in possible_keys:

        if key in data:

            price = float(data[key])

            if price > 0:
                return price

    raise RuntimeError(
        f"Gold API price not found: {data}"
    )


# =========================================================
# CHART GENERATION
# =========================================================

def create_chart(df, title):

    chart_df = df.copy()

    buffer = io.BytesIO()

    mpf.plot(
        chart_df,
        type="candle",
        style="charles",
        title=title,
        volume=False,
        figsize=(12, 7),
        savefig={
            "fname": buffer,
            "dpi": 120,
            "bbox_inches": "tight",
        },
    )

    buffer.seek(0)

    return buffer.getvalue()


# =========================================================
# GEMINI MODEL DISCOVERY
# =========================================================

def normalize_model_name(name):

    if name.startswith("models/"):
        return name[len("models/"):]

    return name


def discover_gemini_models(force=False):

    global gemini_models_cache
    global gemini_models_cache_time

    now = time.time()

    if (
        not force
        and gemini_models_cache
        and now - gemini_models_cache_time
        < GEMINI_CACHE_SECONDS
    ):
        return gemini_models_cache

    models = []

    try:

        listed = client.models.list()

        for model in listed:

            raw_name = getattr(
                model,
                "name",
                None,
            )

            if not raw_name:
                continue

            name = normalize_model_name(
                raw_name
            )

            low = name.lower()

            if any(
                x in low
                for x in [
                    "embedding",
                    "image",
                    "audio",
                    "tts",
                    "aqa",
                ]
            ):
                continue

            models.append(name)

    except Exception:

        models = []

    fallback_models = [
        "gemini-3.8-flash",
        "gemini-3.8-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
    ]

    for model in fallback_models:

        if model not in models:
            models.append(model)

    def model_score(name):

        low = name.lower()

        score = 100

        if "flash" in low:
            score -= 30

        if "pro" in low:
            score -= 10

        if "preview" in low:
            score += 10

        return score

    models.sort(
        key=model_score
    )

    gemini_models_cache = models
    gemini_models_cache_time = now

    return models


# =========================================================
# GEMINI FRESH REQUEST
# =========================================================

def ask_gemini_fresh_sync(
    prompt,
    images=None,
):

    global working_gemini_model

    models = discover_gemini_models()

    if working_gemini_model:

        models = [
            working_gemini_model
        ] + [
            x
            for x in models
            if x != working_gemini_model
        ]

    last_error = None

    for model_name in models:

        try:

            contents = []

            contents.append(
                types.Part.from_text(
                    text=prompt
                )
            )

            if images:

                for image_bytes in images:

                    contents.append(
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/png",
                        )
                    )

            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )

            text = getattr(
                response,
                "text",
                None,
            )

            if not text:

                raise RuntimeError(
                    "Gemini returned empty response"
                )

            working_gemini_model = model_name

            return text.strip()

        except Exception as exc:

            last_error = exc

            continue

    # Refresh model list once
    try:

        models = discover_gemini_models(
            force=True
        )

        for model_name in models:

            try:

                contents = [
                    types.Part.from_text(
                        text=prompt
                    )
                ]

                if images:

                    for image_bytes in images:

                        contents.append(
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/png",
                            )
                        )

                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                )

                text = getattr(
                    response,
                    "text",
                    None,
                )

                if not text:
                    continue

                working_gemini_model = model_name

                return text.strip()

            except Exception as exc:

                last_error = exc

    except Exception as exc:

        last_error = exc

    raise RuntimeError(
        f"Gemini failed: {last_error}"
    )


async def ask_gemini_fresh(
    prompt,
    images=None,
):

    return await asyncio.to_thread(
        ask_gemini_fresh_sync,
        prompt,
        images,
    )


# =========================================================
# INITIAL ANALYSIS PROMPT
# =========================================================

def build_initial_prompt(price):

    return f"""
You are a balanced/aggressive XAU/USD short-term scalping analyst.

This is a COMPLETELY FRESH and INDEPENDENT analysis.

There is:
- NO previous conversation
- NO previous Gemini answer
- NO previous trade
- NO previous setup

Current XAU/USD price:
{price}

You have three fresh candlestick charts:

1H = broad market context
5M = setup and structure
1M = execution timing

Your job is to find the BEST currently tradable setup.

Do NOT be unnecessarily conservative.

Do NOT wait for a perfect textbook ICT setup.

Analyze:

- market direction
- BOS
- CHoCH
- liquidity sweep
- false breakout
- breakout and retest
- displacement
- momentum
- FVG
- order block
- support/resistance
- nearby liquidity
- invalidation
- realistic target

IMPORTANT:

Not every ICT concept must be present.

Do NOT require:
BOS + CHoCH + Sweep + FVG + OB

all at the same time.

A clean setup can be valid with fewer strong confluences.

Examples of acceptable setups:

1. BOS + pullback/retest + momentum
2. Liquidity sweep + strong rejection
3. Breakout + retest
4. CHoCH + displacement + logical pullback
5. Strong continuation from support/resistance
6. Clear reversal with a logical invalidation point

The goal is not to create a trade at any cost.

But also do NOT reject a good setup simply because one ICT concept is missing.

Look for a practical setup that could realistically be traded manually.

RISK/REWARD:

Minimum natural R:R = {MIN_RR}

For BUY:

risk = ENTRY - SL
reward = TP1 - ENTRY

RR = reward / risk

For SELL:

risk = SL - ENTRY
reward = ENTRY - TP1

RR = reward / risk

The TP must be realistic.

DO NOT move TP artificially just to manufacture the required R:R.

The stop loss must have a logical invalidation point.

NO TRADE should be returned only when:

- market structure is genuinely unclear
- price is extremely choppy
- setup is already invalidated
- there is no logical entry
- there is no logical SL
- realistic TP cannot provide at least {MIN_RR} R:R

Do NOT say NO TRADE merely because every ICT concept is not present.

Choose ONLY ONE setup.

Do NOT provide multiple trades.

Do NOT provide alternative entries.

Do NOT provide vague possibilities.

Return EXACTLY one of these formats.

TRADE:

RESULT: TRADE
DIRECTION: BUY
ENTRY: number
SL: number
TP1: number
RR: number
REASON: short explanation

OR:

RESULT: TRADE
DIRECTION: SELL
ENTRY: number
SL: number
TP1: number
RR: number
REASON: short explanation

NO TRADE:

RESULT: NO TRADE
REASON: short explanation
"""


# =========================================================
# PARSE TRADE
# =========================================================

def extract_number(text, label):

    pattern = (
        rf"{re.escape(label)}"
        r"\s*:\s*"
        r"(-?\d+(?:\.\d+)?)"
    )

    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    try:
        return float(
            match.group(1)
        )
    except Exception:
        return None


def parse_trade_response(text):

    if not text:
        return None

    upper = text.upper()

    if "RESULT: NO TRADE" in upper:
        return None

    direction_match = re.search(
        r"DIRECTION\s*:\s*(BUY|SELL)",
        upper,
    )

    if not direction_match:
        return None

    direction = direction_match.group(1)

    entry = extract_number(
        text,
        "ENTRY",
    )

    sl = extract_number(
        text,
        "SL",
    )

    tp1 = extract_number(
        text,
        "TP1",
    )

    rr_reported = extract_number(
        text,
        "RR",
    )

    if (
        entry is None
        or sl is None
        or tp1 is None
    ):
        return None

    if direction == "BUY":

        if not (
            sl < entry < tp1
        ):
            return None

        risk = entry - sl
        reward = tp1 - entry

    else:

        if not (
            sl > entry > tp1
        ):
            return None

        risk = sl - entry
        reward = entry - tp1

    if risk <= 0:
        return None

    rr_calculated = reward / risk

    # Hard Python protection
    if rr_calculated < MIN_RR:
        return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "rr": rr_calculated,
        "rr_reported": rr_reported,
        "reason": extract_reason(text),
        "raw": text,
    }


def extract_reason(text):

    match = re.search(
        r"REASON\s*:\s*(.*)",
        text,
        re.IGNORECASE,
    )

    if not match:
        return "No reason provided."

    return match.group(1).strip()


# =========================================================
# MANAGEMENT PROMPT
# =========================================================

def build_management_prompt(
    trade,
    current_price,
    age_seconds,
):

    return f"""
You are managing an already-open simulated XAU/USD scalping trade.

This is a fresh independent management review.

Do NOT create a new trade.

Do NOT change the entry.

CURRENT TRADE:

Direction:
{trade["direction"]}

Entry:
{trade["entry"]}

Current SL:
{trade["sl"]}

Current TP:
{trade["tp1"]}

Original SL:
{trade["original_sl"]}

Original TP:
{trade["original_tp"]}

Current XAU/USD price:
{current_price}

Trade age:
{int(age_seconds / 60)} minutes

Analyze the fresh 1H, 5M and 1M charts.

Possible actions:

KEEP
MOVE_SL
MOVE_TP
MOVE_SL_AND_TP
CLOSE_PROFIT
CLOSE_LOSS
CLOSE_NOW

Rules:

- Entry is fixed.
- Never create a new entry.
- Never move entry.
- Do not make unnecessary changes.
- A BUY SL must remain below current price.
- A SELL SL must remain above current price.
- A TP must remain logically beyond current price in the trade direction.
- Moving SL to lock profit is allowed when technically justified.
- Do not widen risk without a very strong reason.
- Protect profit when market structure clearly supports it.
- If the trade is losing and structure is invalidated, CLOSE_LOSS or CLOSE_NOW can be used.
- Maximum trade duration is 75 minutes.
- At 60-75 minutes, strongly consider closing if momentum is weak.

Return EXACTLY:

ACTION: KEEP
NEW_SL: current SL
NEW_TP: current TP
REASON: short explanation

OR

ACTION: MOVE_SL
NEW_SL: number
NEW_TP: current TP
REASON: short explanation

OR

ACTION: MOVE_TP
NEW_SL: current SL
NEW_TP: number
REASON: short explanation

OR

ACTION: MOVE_SL_AND_TP
NEW_SL: number
NEW_TP: number
REASON: short explanation

OR

ACTION: CLOSE_PROFIT
NEW_SL: current SL
NEW_TP: current TP
REASON: short explanation

OR

ACTION: CLOSE_LOSS
NEW_SL: current SL
NEW_TP: current TP
REASON: short explanation

OR

ACTION: CLOSE_NOW
NEW_SL: current SL
NEW_TP: current TP
REASON: short explanation
"""


# =========================================================
# MANAGEMENT PARSER
# =========================================================

def parse_management_response(text):

    if not text:
        return None

    action_match = re.search(
        r"ACTION\s*:\s*"
        r"(KEEP|MOVE_SL|MOVE_TP|MOVE_SL_AND_TP|"
        r"CLOSE_PROFIT|CLOSE_LOSS|CLOSE_NOW)",
        text,
        re.IGNORECASE,
    )

    if not action_match:
        return None

    action = action_match.group(1).upper()

    new_sl = extract_number(
        text,
        "NEW_SL",
    )

    new_tp = extract_number(
        text,
        "NEW_TP",
    )

    reason = extract_reason(text)

    return {
        "action": action,
        "new_sl": new_sl,
        "new_tp": new_tp,
        "reason": reason,
        "raw": text,
    }


# =========================================================
# TRADE VALIDATION
# =========================================================

def validate_management(
    trade,
    proposal,
    current_price,
):

    action = proposal["action"]

    if action in [
        "CLOSE_PROFIT",
        "CLOSE_LOSS",
        "CLOSE_NOW",
        "KEEP",
    ]:
        return True, ""

    new_sl = proposal["new_sl"]
    new_tp = proposal["new_tp"]

    if new_sl is None:
        return False, "Missing NEW_SL"

    if new_tp is None:
        return False, "Missing NEW_TP"

    direction = trade["direction"]

    if direction == "BUY":

        if new_sl >= current_price:
            return (
                False,
                "BUY stop loss must remain below current price",
            )

        if new_tp <= current_price:
            return (
                False,
                "BUY take profit must remain above current price",
            )

    else:

        if new_sl <= current_price:
            return (
                False,
                "SELL stop loss must remain above current price",
            )

        if new_tp >= current_price:
            return (
                False,
                "SELL take profit must remain below current price",
            )

    return True, ""


# =========================================================
# ANALYSIS
# =========================================================

async def perform_one_analysis():

    price = await asyncio.to_thread(
        get_live_gold_price
    )

    df_1h = await asyncio.to_thread(
        get_candles,
        "1h",
        120,
    )

    df_5m = await asyncio.to_thread(
        get_candles,
        "5min",
        150,
    )

    df_1m = await asyncio.to_thread(
        get_candles,
        "1min",
        150,
    )

    chart_1h = await asyncio.to_thread(
        create_chart,
        df_1h,
        "XAU/USD 1H",
    )

    chart_5m = await asyncio.to_thread(
        create_chart,
        df_5m,
        "XAU/USD 5M",
    )

    chart_1m = await asyncio.to_thread(
        create_chart,
        df_1m,
        "XAU/USD 1M",
    )

    prompt = build_initial_prompt(
        price
    )

    response = await ask_gemini_fresh(
        prompt,
        images=[
            chart_1h,
            chart_5m,
            chart_1m,
        ],
    )

    trade = parse_trade_response(
        response
    )

    return {
        "price": price,
        "trade": trade,
        "raw": response,
    }


# =========================================================
# WAIT FOR ENTRY
# =========================================================

async def wait_for_entry(
    chat_id,
    trade,
):

    start = time.time()

    last_status = 0

    while (
        time.time() - start
        < ENTRY_WAIT_SECONDS
    ):

        state = trade_states.get(
            chat_id
        )

        if state and state.get(
            "stop_requested"
        ):
            return None

        try:

            price = await asyncio.to_thread(
                get_live_gold_price
            )

        except Exception:

            await asyncio.sleep(
                PRICE_POLL_SECONDS
            )

            continue

        direction = trade[
            "direction"
        ]

        entry = trade[
            "entry"
        ]

        reached = False

        if direction == "BUY":

            if price >= entry - ENTRY_TOLERANCE:
                reached = True

        else:

            if price <= entry + ENTRY_TOLERANCE:
                reached = True

        if reached:

            return price

        if (
            time.time() - last_status
            >= 60
        ):

            elapsed = int(
                time.time() - start
            )

            remaining = max(
                0,
                int(
                    ENTRY_WAIT_SECONDS
                    - elapsed
                ),
            )

            await safe_send_message(
                chat_id,
                (
                    f"⏳ انتظار الدخول\n"
                    f"السعر الحالي: {price:.2f}\n"
                    f"Entry: {entry:.2f}\n"
                    f"المتبقي تقريبًا: "
                    f"{remaining // 60}m "
                    f"{remaining % 60}s"
                ),
                stop_keyboard(),
            )

            last_status = time.time()

        await asyncio.sleep(
            PRICE_POLL_SECONDS
        )

    return None


# =========================================================
# CHECK CURRENT TRADE LEVELS
# =========================================================

def trade_level_hit(
    trade,
    price,
):

    direction = trade["direction"]

    sl = trade["sl"]
    tp = trade["tp1"]

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


# =========================================================
# PROPOSAL APPLY
# =========================================================

async def apply_management_proposal(
    chat_id,
    proposal_id,
):

    await asyncio.sleep(
        MODIFICATION_REJECTION_SECONDS
    )

    state = trade_states.get(
        chat_id
    )

    if not state:
        return

    proposal = state.get(
        "pending_proposal"
    )

    if not proposal:
        return

    if proposal["id"] != proposal_id:
        return

    if state.get(
        "stop_requested"
    ):
        return

    # Check if the proposal is still current
    if state.get(
        "pending_proposal"
    )["id"] != proposal_id:
        return

    trade = state["trade"]

    try:

        current_price = await asyncio.to_thread(
            get_live_gold_price
        )

    except Exception:

        state["pending_proposal"] = None

        await safe_send_message(
            chat_id,
            "⚠️ لم أستطع التحقق من السعر، لذلك لم يتم تنفيذ التعديل.",
            stop_keyboard(),
        )

        return

    # Existing SL/TP always has priority
    hit = trade_level_hit(
        trade,
        current_price,
    )

    if hit:

        state["pending_proposal"] = None

        return

    valid, reason = validate_management(
        trade,
        proposal,
        current_price,
    )

    if not valid:

        state["pending_proposal"] = None

        await safe_send_message(
            chat_id,
            (
                "⚠️ لم يتم تنفيذ تعديل Gemini.\n"
                f"السبب: {reason}"
            ),
            stop_keyboard(),
        )

        return

    action = proposal[
        "action"
    ]

    if action == "KEEP":

        state["pending_proposal"] = None

        await safe_send_message(
            chat_id,
            (
                "✅ Gemini قرر إبقاء الصفقة كما هي.\n"
                f"SL: {trade['sl']:.2f}\n"
                f"TP: {trade['tp1']:.2f}"
            ),
            stop_keyboard(),
        )

        return

    if action in [
        "CLOSE_PROFIT",
        "CLOSE_LOSS",
        "CLOSE_NOW",
    ]:

        state["pending_proposal"] = None
        state["closed"] = True
        state["close_reason"] = action

        await safe_send_message(
            chat_id,
            (
                f"🔒 إغلاق الصفقة بواسطة Gemini\n"
                f"السبب: {proposal['reason']}"
            ),
            stop_keyboard(),
        )

        return

    old_sl = trade["sl"]
    old_tp = trade["tp1"]

    trade["sl"] = proposal[
        "new_sl"
    ]

    trade["tp1"] = proposal[
        "new_tp"
    ]

    state["pending_proposal"] = None

    await safe_send_message(
        chat_id,
        (
            "✅ تم تنفيذ تعديل Gemini بعد 60 ثانية.\n\n"
            f"SL: {old_sl:.2f} → {trade['sl']:.2f}\n"
            f"TP: {old_tp:.2f} → {trade['tp1']:.2f}\n\n"
            f"السبب: {proposal['reason']}"
        ),
        stop_keyboard(),
    )


# =========================================================
# MANAGEMENT REVIEW
# =========================================================

async def run_management_review(
    chat_id,
):

    state = trade_states.get(
        chat_id
    )

    if not state:
        return

    trade = state["trade"]

    if state.get(
        "closed"
    ):
        return

    try:

        current_price = await asyncio.to_thread(
            get_live_gold_price
        )

        df_1h = await asyncio.to_thread(
            get_candles,
            "1h",
            120,
        )

        df_5m = await asyncio.to_thread(
            get_candles,
            "5min",
            150,
        )

        df_1m = await asyncio.to_thread(
            get_candles,
            "1min",
            150,
        )

        chart_1h = await asyncio.to_thread(
            create_chart,
            df_1h,
            "XAU/USD 1H Management",
        )

        chart_5m = await asyncio.to_thread(
            create_chart,
            df_5m,
            "XAU/USD 5M Management",
        )

        chart_1m = await asyncio.to_thread(
            create_chart,
            df_1m,
            "XAU/USD 1M Management",
        )

        age = (
            time.time()
            - state["opened_at"]
        )

        prompt = build_management_prompt(
            trade,
            current_price,
            age,
        )

        response = await ask_gemini_fresh(
            prompt,
            images=[
                chart_1h,
                chart_5m,
                chart_1m,
            ],
        )

        proposal = parse_management_response(
            response
        )

        if not proposal:
            await safe_send_message(
                chat_id,
                (
                    "⚠️ Gemini management review "
                    "لم يعطِ أمرًا صالحًا.\n"
                    "لم يتم تغيير الصفقة."
                ),
                stop_keyboard(),
            )

            return

        # KEEP can be sent directly
        if proposal["action"] == "KEEP":

            await safe_send_message(
                chat_id,
                (
                    "🔎 مراجعة Gemini:\n"
                    "KEEP — لا يوجد تعديل مطلوب."
                ),
                stop_keyboard(),
            )

            return

        global proposal_counter

        proposal_counter += 1

        proposal_id = proposal_counter

        state["pending_proposal"] = {
            "id": proposal_id,
            "action": proposal["action"],
            "new_sl": proposal["new_sl"],
            "new_tp": proposal["new_tp"],
            "reason": proposal["reason"],
            "created_at": time.time(),
        }

        if proposal["action"] in [
            "CLOSE_PROFIT",
            "CLOSE_LOSS",
            "CLOSE_NOW",
        ]:

            text = (
                "⚠️ Gemini يقترح إغلاق الصفقة.\n\n"
                f"الإجراء: {proposal['action']}\n"
                f"السبب: {proposal['reason']}\n\n"
                "سيتم التنفيذ بعد 60 ثانية ما لم تضغط "
                "«❌ لا تنفذ التعديل»."
            )

        else:

            text = (
                "⚠️ Gemini يقترح تعديل الصفقة.\n\n"
                f"SL الجديد: {proposal['new_sl']:.2f}\n"
                f"TP الجديد: {proposal['new_tp']:.2f}\n\n"
                f"السبب: {proposal['reason']}\n\n"
                "سيتم التنفيذ بعد 60 ثانية ما لم تضغط "
                "«❌ لا تنفذ التعديل»."
            )

        await safe_send_message(
            chat_id,
            text,
            rejection_keyboard(
                proposal_id
            ),
        )

        asyncio.create_task(
            apply_management_proposal(
                chat_id,
                proposal_id,
            )
        )

    except Exception as exc:

        await safe_send_message(
            chat_id,
            (
                "⚠️ حدث خطأ أثناء مراجعة Gemini.\n"
                "لم يتم تغيير الصفقة."
            ),
            stop_keyboard(),
        )


# =========================================================
# TRADE MONITOR
# =========================================================

async def monitor_trade(
    chat_id,
):

    state = trade_states.get(
        chat_id
    )

    if not state:
        return

    trade = state["trade"]

    last_management_review = time.time()

    last_status = time.time()

    await safe_send_message(
        chat_id,
        (
            "🟢 تم فتح الصفقة التجريبية.\n\n"
            f"الاتجاه: {trade['direction']}\n"
            f"Entry: {trade['entry']:.2f}\n"
            f"SL: {trade['sl']:.2f}\n"
            f"TP: {trade['tp1']:.2f}\n"
            f"R:R: {trade['rr']:.2f}"
        ),
        stop_keyboard(),
    )

    while True:

        state = trade_states.get(
            chat_id
        )

        if not state:
            return

        if state.get(
            "stop_requested"
        ):

            state["closed"] = True
            state["close_reason"] = "STOPPED"

            await safe_send_message(
                chat_id,
                "🛑 تم إيقاف مراقبة الصفقة.",
                main_keyboard(),
            )

            return

        if state.get(
            "closed"
        ):

            return

        try:

            price = await asyncio.to_thread(
                get_live_gold_price
            )

        except Exception:

            await asyncio.sleep(
                PRICE_POLL_SECONDS
            )

            continue

        # -------------------------------------------------
        # Existing levels always win over pending proposal
        # -------------------------------------------------

        hit = trade_level_hit(
            trade,
            price,
        )

        if hit:

            state["closed"] = True
            state["close_reason"] = hit
            state["close_price"] = price

            if hit == "TP":

                message = (
                    "🎯 TP HIT\n\n"
                    f"السعر: {price:.2f}\n"
                    f"TP: {trade['tp1']:.2f}"
                )

            else:

                message = (
                    "🛑 SL HIT\n\n"
                    f"السعر: {price:.2f}\n"
                    f"SL: {trade['sl']:.2f}"
                )

            state["pending_proposal"] = None

            await safe_send_message(
                chat_id,
                message,
                main_keyboard(),
            )

            return

        # -------------------------------------------------
        # Maximum duration
        # -------------------------------------------------

        age = (
            time.time()
            - state["opened_at"]
        )

        if age >= MAX_TRADE_SECONDS:

            state["closed"] = True
            state["close_reason"] = "TIMEOUT"
            state["close_price"] = price
            state["pending_proposal"] = None

            await safe_send_message(
                chat_id,
                (
                    "⏰ انتهت مدة الصفقة القصوى "
                    "75 دقيقة.\n\n"
                    f"السعر عند الإغلاق: {price:.2f}"
                ),
                main_keyboard(),
            )

            return

        # -------------------------------------------------
        # Periodic status
        # -------------------------------------------------

        if (
            time.time() - last_status
            >= 60
        ):

            remaining = max(
                0,
                MAX_TRADE_SECONDS
                - age,
            )

            await safe_send_message(
                chat_id,
                (
                    "📊 حالة الصفقة\n\n"
                    f"الاتجاه: {trade['direction']}\n"
                    f"السعر: {price:.2f}\n"
                    f"Entry: {trade['entry']:.2f}\n"
                    f"SL: {trade['sl']:.2f}\n"
                    f"TP: {trade['tp1']:.2f}\n"
                    f"العمر: {int(age // 60)} دقيقة\n"
                    f"المتبقي: {int(remaining // 60)} دقيقة"
                ),
                stop_keyboard(),
            )

            last_status = time.time()

        # -------------------------------------------------
        # Gemini management every 15 minutes
        # -------------------------------------------------

        if (
            time.time()
            - last_management_review
            >= MANAGEMENT_REVIEW_SECONDS
        ):

            pending = state.get(
                "pending_proposal"
            )

            if not pending:

                last_management_review = time.time()

                asyncio.create_task(
                    run_management_review(
                        chat_id
                    )
                )

        await asyncio.sleep(
            PRICE_POLL_SECONDS
        )


# =========================================================
# ANALYSIS LOOP
# =========================================================

async def analysis_loop(
    chat_id,
):

    try:

        number = 0

        while True:

            state = analysis_tasks.get(
                chat_id
            )

            if not state:
                return

            if state.get(
                "stop_requested"
            ):
                return

            number += 1

            await safe_send_message(
                chat_id,
                (
                    f"🔎 تحليل مستقل رقم {number}\n\n"
                    "Gemini يبدأ تحليلًا جديدًا "
                    "بدون استخدام أي تحليل سابق."
                ),
                stop_keyboard(),
            )

            try:

                result = await perform_one_analysis()

            except Exception as exc:

                await safe_send_message(
                    chat_id,
                    (
                        "⚠️ فشل التحليل.\n\n"
                        f"{str(exc)[:500]}"
                    ),
                    stop_keyboard(),
                )

                await asyncio.sleep(
                    REANALYSIS_WAIT_SECONDS
                )

                continue

            raw = result["raw"]
            trade = result["trade"]
            price = result["price"]

            if not trade:

                await safe_send_message(
                    chat_id,
                    (
                        "🚫 NO TRADE\n\n"
                        f"السعر: {price:.2f}\n\n"
                        f"{raw[:1500]}"
                    ),
                    stop_keyboard(),
                )

                await asyncio.sleep(
                    REANALYSIS_WAIT_SECONDS
                )

                continue

            # -------------------------------------------------
            # Valid trade found
            # -------------------------------------------------

            await safe_send_message(
                chat_id,
                (
                    "🟢 فرصة محتملة\n\n"
                    f"الاتجاه: {trade['direction']}\n"
                    f"Entry: {trade['entry']:.2f}\n"
                    f"SL: {trade['sl']:.2f}\n"
                    f"TP1: {trade['tp1']:.2f}\n"
                    f"R:R: {trade['rr']:.2f}\n\n"
                    f"السبب:\n{trade['reason']}\n\n"
                    "⏳ سأنتظر وصول السعر إلى Entry "
                    "لمدة أقصاها 10 دقائق."
                ),
                stop_keyboard(),
            )

            # -------------------------------------------------
            # Wait for entry
            # -------------------------------------------------

            entry_price = await wait_for_entry(
                chat_id,
                trade,
            )

            state = analysis_tasks.get(
                chat_id
            )

            if (
                not state
                or state.get(
                    "stop_requested"
                )
            ):
                return

            if entry_price is None:

                await safe_send_message(
                    chat_id,
                    (
                        "⌛ لم يصل السعر إلى Entry "
                        "خلال 10 دقائق.\n\n"
                        "تم إلغاء هذا الـsetup."
                    ),
                    stop_keyboard(),
                )

                await asyncio.sleep(
                    REANALYSIS_WAIT_SECONDS
                )

                continue

            # -------------------------------------------------
            # Open simulated trade
            # -------------------------------------------------

            trade_state = {
                "trade": {
                    "direction": trade["direction"],
                    "entry": trade["entry"],
                    "sl": trade["sl"],
                    "tp1": trade["tp1"],
                    "rr": trade["rr"],
                    "original_sl": trade["sl"],
                    "original_tp": trade["tp1"],
                    "reason": trade["reason"],
                },
                "opened_at": time.time(),
                "closed": False,
                "stop_requested": False,
                "pending_proposal": None,
            }

            trade_states[
                chat_id
            ] = trade_state

            await monitor_trade(
                chat_id
            )

            # -------------------------------------------------
            # Trade ended
            # -------------------------------------------------

            trade_states.pop(
                chat_id,
                None,
            )

            state = analysis_tasks.get(
                chat_id
            )

            if not state:
                return

            if state.get(
                "stop_requested"
            ):
                return

            await safe_send_message(
                chat_id,
                (
                    "🔄 انتهت الصفقة.\n"
                    f"انتظار {REANALYSIS_WAIT_SECONDS // 60} "
                    "دقائق ثم يبدأ تحليل Gemini جديد مستقل تمامًا."
                ),
                main_keyboard(),
            )

            await asyncio.sleep(
                REANALYSIS_WAIT_SECONDS
            )

    except asyncio.CancelledError:

        return

    except Exception as exc:

        traceback.print_exc()

        await safe_send_message(
            chat_id,
            (
                "❌ توقف مسار التحليل بسبب خطأ.\n"
                f"{str(exc)[:500]}"
            ),
            main_keyboard(),
        )


# =========================================================
# SAFE TELEGRAM MESSAGE
# =========================================================

async def safe_send_message(
    chat_id,
    text,
    reply_markup=None,
):

    try:

        application = ApplicationHolder.application

        if not application:
            return

        await application.bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
        )

    except Exception:

        traceback.print_exc()


# =========================================================
# APPLICATION HOLDER
# =========================================================

class ApplicationHolder:

    application = None


# =========================================================
# /START
# =========================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    chat_id = update.effective_chat.id

    await update.message.reply_text(
        "RustyGold جاهز",
        reply_markup=main_keyboard(),
    )


# =========================================================
# STOP
# =========================================================

async def stop_everything(
    chat_id,
):

    state = analysis_tasks.get(
        chat_id
    )

    if state:
        state[
            "stop_requested"
        ] = True

    trade = trade_states.get(
        chat_id
    )

    if trade:
        trade[
            "stop_requested"
        ] = True

        trade[
            "pending_proposal"
        ] = None

    task = analysis_tasks.get(
        chat_id,
        {},
    ).get(
        "task"
    )

    if task:

        current = asyncio.current_task()

        if task != current:
            task.cancel()

    analysis_tasks.pop(
        chat_id,
        None,
    )


# =========================================================
# SUMMARY
# =========================================================

async def send_summary(
    chat_id,
):

    state = trade_states.get(
        chat_id
    )

    analysis = analysis_tasks.get(
        chat_id
    )

    if state:

        trade = state["trade"]

        try:
            price = await asyncio.to_thread(
                get_live_gold_price
            )
        except Exception:
            price = None

        if price is None:

            text = (
                "📊 الصفقة الحالية\n\n"
                f"الاتجاه: {trade['direction']}\n"
                f"Entry: {trade['entry']:.2f}\n"
                f"SL: {trade['sl']:.2f}\n"
                f"TP: {trade['tp1']:.2f}\n"
                f"R:R: {trade['rr']:.2f}"
            )

        else:

            text = (
                "📊 الصفقة الحالية\n\n"
                f"الاتجاه: {trade['direction']}\n"
                f"السعر: {price:.2f}\n"
                f"Entry: {trade['entry']:.2f}\n"
                f"SL: {trade['sl']:.2f}\n"
                f"TP: {trade['tp1']:.2f}\n"
                f"R:R: {trade['rr']:.2f}"
            )

        await safe_send_message(
            chat_id,
            text,
            stop_keyboard(),
        )

        return

    if analysis:

        await safe_send_message(
            chat_id,
            "🔎 RustyGold يعمل حاليًا على التحليل.",
            stop_keyboard(),
        )

        return

    await safe_send_message(
        chat_id,
        "ℹ️ لا توجد صفقة مفتوحة ولا تحليل يعمل حاليًا.",
        main_keyboard(),
    )


# =========================================================
# CALLBACKS
# =========================================================

async def callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    chat_id = query.message.chat_id

    data = query.data

    # -----------------------------------------------------
    # START ANALYSIS
    # -----------------------------------------------------

    if data == "START_ANALYSIS":

        existing = analysis_tasks.get(
            chat_id
        )

        if existing:

            await query.message.reply_text(
                "🔎 التحليل يعمل بالفعل.",
                reply_markup=stop_keyboard(),
            )

            return

        state = {
            "stop_requested": False,
            "task": None,
        }

        analysis_tasks[
            chat_id
        ] = state

        task = asyncio.create_task(
            analysis_loop(
                chat_id
            )
        )

        state[
            "task"
        ] = task

        await query.message.reply_text(
            (
                "🚀 بدأ RustyGold.\n\n"
                f"Minimum R:R = {MIN_RR:.2f}\n"
                "Gemini mode = Balanced/Aggressive\n\n"
                "كل تحليل مستقل عن السابق."
            ),
            reply_markup=stop_keyboard(),
        )

        return

    # -----------------------------------------------------
    # STOP
    # -----------------------------------------------------

    if data == "STOP_ANALYSIS":

        await stop_everything(
            chat_id
        )

        await query.message.reply_text(
            "🛑 تم إيقاف التحليل والمراقبة.",
            reply_markup=main_keyboard(),
        )

        return

    # -----------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------

    if data == "SUMMARY":

        await send_summary(
            chat_id
        )

        return

    # -----------------------------------------------------
    # REJECT PROPOSAL
    # -----------------------------------------------------

    if data.startswith(
        "REJECT:"
    ):

        try:

            proposal_id = int(
                data.split(
                    ":",
                    1,
                )[1]
            )

        except Exception:

            return

        state = trade_states.get(
            chat_id
        )

        if not state:
            return

        proposal = state.get(
            "pending_proposal"
        )

        if not proposal:
            return

        # Old buttons cannot touch newer proposals
        if proposal["id"] != proposal_id:
            await query.message.reply_text(
                "ℹ️ هذا المقترح قديم ولم يعد فعالًا.",
                reply_markup=stop_keyboard(),
            )

            return

        state[
            "pending_proposal"
        ] = None

        await query.message.reply_text(
            (
                "❌ تم رفض تعديل Gemini.\n"
                "الصفقة ستستمر بالمستويات الحالية."
            ),
            reply_markup=stop_keyboard(),
        )

        return


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update,
    context,
):

    print(
        "Telegram error:",
        context.error,
    )

    traceback.print_exc()


# =========================================================
# MAIN
# =========================================================

def main():

    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True,
    )

    flask_thread.start()

    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    ApplicationHolder.application = application

    application.add_handler(
        CommandHandler(
            "start",
            start_command,
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

    print(
        "RustyGold bot starting..."
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()