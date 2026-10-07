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

GEMINI_GLOBAL_TIMEOUT_MS = 90000

client = genai.Client(
    api_key=GEMINI_KEY,
    http_options=types.HttpOptions(
        timeout=GEMINI_GLOBAL_TIMEOUT_MS
    ),
)

GEMINI_REQUEST_TIMEOUT_MS = 45000


# =========================================================
# CONFIG
# =========================================================

SYMBOL = "XAU/USD"

MIN_RR = 1.25

PRICE_POLL_SECONDS = 5

MANAGEMENT_REVIEW_SECONDS = 15 * 60

MODIFICATION_REJECTION_SECONDS = 60

MAX_TRADE_SECONDS = 75 * 60

ENTRY_WAIT_SECONDS = 10 * 60

REANALYSIS_WAIT_SECONDS = 5 * 60

ENTRY_TOLERANCE = 0.60

GEMINI_CACHE_SECONDS = 300

TRADE_STATUS_SECONDS = 5 * 60

# ---------------------------------------------------------
# MANAGEMENT RULES
# ---------------------------------------------------------

# Strong positive score required before modifying.
MIN_SCORE_FOR_SL = 4
MIN_SCORE_FOR_TP = 5

# Losing trade must receive strong negative score twice
# consecutively before Gemini is allowed to close it.
MIN_SCORE_FOR_LOSS_CLOSE = -5
LOSS_CLOSE_CONFIRMATIONS = 2

# Profitable Gemini close also needs strong evidence.
MIN_SCORE_FOR_PROFIT_CLOSE = -2

# Do not allow a new management proposal to be applied
# if the market moved too far while waiting.
MAX_PROPOSAL_PRICE_DRIFT = 3.0


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
# PERFORMANCE
# =========================================================

performance_stats = {
    "total_trades": 0,
    "wins": 0,
    "losses": 0,
    "other_closes": 0,

    "total_r": 0.0,
    "gross_profit_r": 0.0,
    "gross_loss_r": 0.0,

    "adjusted_trades": 0,
    "adjusted_wins": 0,
    "adjusted_losses": 0,
    "adjusted_other": 0,

    "adjusted_total_r": 0.0,
    "adjusted_profit_r": 0.0,
    "adjusted_loss_r": 0.0,

    "avoided_original_sl": 0,

    "total_modifications": 0,
    "sl_modifications": 0,
    "tp_modifications": 0,

    "forced_closes": 0,

    "gemini_closes": 0,
    "gemini_close_losses": 0,
    "gemini_close_profit": 0.0,

    "last_modification": None,
}


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

def get_candles(
    interval,
    outputsize=120,
):

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

    df = pd.DataFrame(
        data["values"]
    )

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

            price = float(
                data[key]
            )

            if price > 0:
                return price

    raise RuntimeError(
        f"Gold API price not found: {data}"
    )


# =========================================================
# CHART GENERATION
# =========================================================

def create_chart(
    df,
    title,
):

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

        return name[
            len("models/"):
        ]

    return name


def discover_gemini_models(
    force=False
):

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

    except Exception as exc:

        print(
            "Gemini model discovery failed:",
            repr(exc),
        )

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
# GEMINI CONTENT BUILDER
# =========================================================

def build_gemini_contents(
    prompt,
    images=None,
):

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

    return contents


# =========================================================
# GEMINI REQUEST
# =========================================================

def ask_one_gemini_model(
    model_name,
    prompt,
    images=None,
):

    contents = build_gemini_contents(
        prompt,
        images,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=contents,
        config={
            "http_options": {
                "timeout": GEMINI_REQUEST_TIMEOUT_MS
            }
        },
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

    return text.strip()


async def ask_gemini_fresh(
    prompt,
    images=None,
):

    global working_gemini_model

    models = await asyncio.to_thread(
        discover_gemini_models
    )

    if working_gemini_model:

        models = [
            working_gemini_model
        ] + [
            x
            for x in models
            if x != working_gemini_model
        ]

    errors = []

    for model_name in models:

        try:

            result = await asyncio.to_thread(
                ask_one_gemini_model,
                model_name,
                prompt,
                images,
            )

            working_gemini_model = model_name

            return result

        except Exception as exc:

            error_text = (
                f"{type(exc).__name__}: {str(exc)}"
            )

            errors.append(
                f"{model_name} -> {error_text}"
            )

            print(
                "Gemini model failed:",
                model_name,
                repr(exc),
            )

            continue

    try:

        refreshed_models = await asyncio.to_thread(
            discover_gemini_models,
            True,
        )

    except Exception as exc:

        refreshed_models = []

        errors.append(
            f"Model refresh -> "
            f"{type(exc).__name__}: {str(exc)}"
        )

    for model_name in refreshed_models:

        if model_name in models:
            continue

        try:

            result = await asyncio.to_thread(
                ask_one_gemini_model,
                model_name,
                prompt,
                images,
            )

            working_gemini_model = model_name

            return result

        except Exception as exc:

            error_text = (
                f"{type(exc).__name__}: {str(exc)}"
            )

            errors.append(
                f"{model_name} -> {error_text}"
            )

            print(
                "Gemini fallback failed:",
                model_name,
                repr(exc),
            )

            continue

    final_error = "\n".join(
        errors[-8:]
    )

    raise RuntimeError(
        "Gemini failed on all available models:\n"
        + final_error
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

OR:

RESULT: NO TRADE
REASON: short explanation
"""


# =========================================================
# PARSE TRADE
# =========================================================

def extract_number(
    text,
    label,
):

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


def extract_reason(text):

    match = re.search(
        r"REASON\s*:\s*(.*)",
        text,
        re.IGNORECASE,
    )

    if not match:
        return "No reason provided."

    return match.group(1).strip()


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

    rr_calculated = (
        reward / risk
    )

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


# =========================================================
# MANAGEMENT PROMPT
# =========================================================

def build_management_prompt(
    trade,
    current_price,
    age_seconds,
):

    return f"""
You are an independent XAU/USD trade management engine.

You are NOT the entry analyst.

You must NOT defend the original trade idea.

You must NOT know why the trade was opened.

Evaluate the CURRENT market from scratch using only:
- current trade levels
- current price
- fresh 1H chart
- fresh 5M chart
- fresh 1M chart

IMPORTANT:

The trade direction is FIXED.

If the trade is BUY, you are NOT allowed to turn it into SELL.

If the trade is SELL, you are NOT allowed to turn it into BUY.

You can only manage the existing position.

CURRENT POSITION:

Direction:
{trade["direction"]}

Entry:
{trade["entry"]}

Current SL:
{trade["sl"]}

Current TP:
{trade["tp1"]}

Current price:
{current_price}

Trade age:
{int(age_seconds / 60)} minutes

Analyze independently:

- current market structure
- BOS
- CHoCH
- displacement
- momentum
- liquidity
- rejection
- continuation
- reversal risk
- distance from current SL
- distance from current TP
- whether current SL/TP are still technically sensible

Do NOT assume the original trade was correct.

Do NOT assume the original trade was wrong.

Judge only the CURRENT situation.

==================================================
STRENGTH SCORE
==================================================

Give one score from -8 to +8.

+8 = extremely strong continuation in favor of the current position
+5 to +7 = strong position
+2 to +4 = moderately favorable
-1 to +1 = unclear / neutral
-2 to -4 = deteriorating
-5 to -7 = strongly invalidating
-8 = extremely strong invalidation

The score must reflect the CURRENT market,
not the original entry thesis.

==================================================
MANAGEMENT
==================================================

Possible actions:

HOLD
MOVE_SL
MOVE_TP
MOVE_SL_AND_TP
CLOSE

Rules:

1. HOLD is the default when evidence is unclear.

2. Do not close merely because price moved temporarily against the trade.

3. A losing trade should only receive CLOSE when there is strong
   evidence that the CURRENT structure is invalidating the position.

4. Do not use emotional reasoning.

5. Do not widen the original risk.

6. A BUY SL may only move UP.
   A BUY TP may only move UP.

7. A SELL SL may only move DOWN.
   A SELL TP may only move DOWN.

8. Never move SL farther away from the current price in a way
   that increases the trade's original risk.

9. SL modification should normally require a score of at least +4.

10. TP extension should normally require a score of at least +5.

11. When score is between -1 and +1, prefer HOLD.

12. If the trade is strongly profitable and structure remains strong,
    protecting profit is preferred over unnecessarily closing the trade.

13. If the trade is losing but structure is not clearly invalidated,
    HOLD rather than panic-close.

14. Maximum trade duration is 75 minutes.
    If the trade reaches 60-75 minutes and momentum is weak,
    CLOSE can be considered.

15. Do not make changes merely to make a change.

==================================================
OUTPUT
==================================================

Return EXACTLY:

SCORE: integer from -8 to +8
ACTION: HOLD
NEW_SL: current SL
NEW_TP: current TP
REASON: short explanation

OR:

SCORE: integer from -8 to +8
ACTION: MOVE_SL
NEW_SL: number
NEW_TP: current TP
REASON: short explanation

OR:

SCORE: integer from -8 to +8
ACTION: MOVE_TP
NEW_SL: current SL
NEW_TP: number
REASON: short explanation

OR:

SCORE: integer from -8 to +8
ACTION: MOVE_SL_AND_TP
NEW_SL: number
NEW_TP: number
REASON: short explanation

OR:

SCORE: integer from -8 to +8
ACTION: CLOSE
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

    score_match = re.search(
        r"SCORE\s*:\s*(-?\d+)",
        text,
        re.IGNORECASE,
    )

    action_match = re.search(
        r"ACTION\s*:\s*"
        r"(HOLD|MOVE_SL|MOVE_TP|MOVE_SL_AND_TP|CLOSE)",
        text,
        re.IGNORECASE,
    )

    if not score_match or not action_match:
        return None

    try:
        score = int(
            score_match.group(1)
        )
    except Exception:
        return None

    score = max(
        -8,
        min(8, score)
    )

    action = action_match.group(1).upper()

    new_sl = extract_number(
        text,
        "NEW_SL",
    )

    new_tp = extract_number(
        text,
        "NEW_TP",
    )

    reason = extract_reason(
        text
    )

    return {
        "score": score,
        "action": action,
        "new_sl": new_sl,
        "new_tp": new_tp,
        "reason": reason,
        "raw": text,
    }


# =========================================================
# MANAGEMENT VALIDATION
# =========================================================

def validate_management(
    trade,
    proposal,
    current_price,
):

    action = proposal["action"]

    score = proposal["score"]

    old_sl = trade["sl"]
    old_tp = trade["tp1"]

    new_sl = proposal["new_sl"]
    new_tp = proposal["new_tp"]

    direction = trade["direction"]

    # -----------------------------------------------------
    # HOLD
    # -----------------------------------------------------

    if action == "HOLD":

        return True, ""

    # -----------------------------------------------------
    # CLOSE
    # -----------------------------------------------------

    if action == "CLOSE":

        return True, ""

    # -----------------------------------------------------
    # SCORE GATES
    # -----------------------------------------------------

    if action == "MOVE_SL":

        if score < MIN_SCORE_FOR_SL:

            return (
                False,
                "Score too weak for SL modification",
            )

    if action == "MOVE_TP":

        if score < MIN_SCORE_FOR_TP:

            return (
                False,
                "Score too weak for TP extension",
            )

    if action == "MOVE_SL_AND_TP":

        if (
            score < MIN_SCORE_FOR_SL
            and score < MIN_SCORE_FOR_TP
        ):

            return (
                False,
                "Score too weak for modification",
            )

    if new_sl is None:
        return False, "Missing NEW_SL"

    if new_tp is None:
        return False, "Missing NEW_TP"

    # -----------------------------------------------------
    # CURRENT PRICE POSITION
    # -----------------------------------------------------

    if direction == "BUY":

        if new_sl >= current_price:

            return (
                False,
                "BUY SL must remain below current price",
            )

        if new_tp <= current_price:

            return (
                False,
                "BUY TP must remain above current price",
            )

        # SL may only improve.
        if new_sl < old_sl:

            return (
                False,
                "BUY SL cannot be widened",
            )

        # TP may only extend.
        if new_tp < old_tp:

            return (
                False,
                "BUY TP cannot be moved closer",
            )

        # At least one actual improvement.
        if (
            abs(new_sl - old_sl) < 0.00001
            and abs(new_tp - old_tp) < 0.00001
        ):

            return (
                False,
                "No actual level improvement",
            )

    else:

        if new_sl <= current_price:

            return (
                False,
                "SELL SL must remain above current price",
            )

        if new_tp >= current_price:

            return (
                False,
                "SELL TP must remain below current price",
            )

        # SL may only improve.
        if new_sl > old_sl:

            return (
                False,
                "SELL SL cannot be widened",
            )

        # TP may only extend.
        if new_tp > old_tp:

            return (
                False,
                "SELL TP cannot be moved closer",
            )

        if (
            abs(new_sl - old_sl) < 0.00001
            and abs(new_tp - old_tp) < 0.00001
        ):

            return (
                False,
                "No actual level improvement",
            )

    return True, ""


# =========================================================
# R CALCULATIONS
# =========================================================

def calculate_trade_r(
    trade,
    close_price,
):

    direction = trade["direction"]

    entry = trade["entry"]

    original_sl = trade[
        "original_sl"
    ]

    risk = abs(
        entry - original_sl
    )

    if risk <= 0:
        return 0.0

    if direction == "BUY":

        return (
            close_price - entry
        ) / risk

    return (
        entry - close_price
    ) / risk


# =========================================================
# RECORD CLOSED TRADE
# =========================================================

def record_closed_trade(
    state,
    close_price,
):

    trade = state["trade"]

    close_reason = state.get(
        "close_reason",
        "UNKNOWN",
    )

    original_r = calculate_trade_r(
        trade,
        close_price,
    )

    state[
        "final_original_r"
    ] = original_r

    # -----------------------------------------------------
    # FORCED CLOSE
    # -----------------------------------------------------

    if close_reason in [
        "STOPPED",
        "FORCED",
        "MANUAL",
    ]:

        performance_stats[
            "forced_closes"
        ] += 1

        state[
            "counted_in_performance"
        ] = False

        return

    # -----------------------------------------------------
    # NORMAL PERFORMANCE
    # -----------------------------------------------------

    performance_stats[
        "total_trades"
    ] += 1

    performance_stats[
        "total_r"
    ] += original_r

    state[
        "counted_in_performance"
    ] = True

    if original_r > 0:

        performance_stats[
            "wins"
        ] += 1

        performance_stats[
            "gross_profit_r"
        ] += original_r

    elif original_r < 0:

        performance_stats[
            "losses"
        ] += 1

        performance_stats[
            "gross_loss_r"
        ] += abs(original_r)

    else:

        performance_stats[
            "other_closes"
        ] += 1

    # -----------------------------------------------------
    # MODIFIED TRADE
    # -----------------------------------------------------

    if state.get(
        "was_modified"
    ):

        adjusted_r = original_r

        state[
            "adjusted_r"
        ] = adjusted_r

        performance_stats[
            "adjusted_trades"
        ] += 1

        performance_stats[
            "adjusted_total_r"
        ] += adjusted_r

        if adjusted_r > 0:

            performance_stats[
                "adjusted_wins"
            ] += 1

            performance_stats[
                "adjusted_profit_r"
            ] += adjusted_r

        elif adjusted_r < 0:

            performance_stats[
                "adjusted_losses"
            ] += 1

            performance_stats[
                "adjusted_loss_r"
            ] += abs(adjusted_r)

        else:

            performance_stats[
                "adjusted_other"
            ] += 1

    else:

        state[
            "adjusted_r"
        ] = None


# =========================================================
# ORIGINAL SL AVOIDANCE
# =========================================================

def check_original_sl_avoided(
    state,
    close_price,
):

    if not state.get(
        "counted_in_performance"
    ):
        return

    trade = state["trade"]

    if not state.get(
        "was_modified"
    ):
        return

    direction = trade["direction"]

    original_sl = trade[
        "original_sl"
    ]

    close_reason = state.get(
        "close_reason"
    )

    if close_reason == "SL":
        return

    if direction == "BUY":

        if close_price > original_sl:

            performance_stats[
                "avoided_original_sl"
            ] += 1

    else:

        if close_price < original_sl:

            performance_stats[
                "avoided_original_sl"
            ] += 1


# =========================================================
# FORMAT R
# =========================================================

def format_r(value):

    return f"{value:+.2f}R"


# =========================================================
# TRADE RESULT SUMMARY
# =========================================================

def build_closed_trade_summary(
    state,
):

    trade = state["trade"]

    close_price = state.get(
        "close_price",
        trade["entry"],
    )

    close_reason = state.get(
        "close_reason",
        "UNKNOWN",
    )

    result_r = state.get(
        "final_original_r",
        calculate_trade_r(
            trade,
            close_price,
        ),
    )

    if close_reason in [
        "STOPPED",
        "FORCED",
        "MANUAL",
    ]:

        result_text = "⚠️ إغلاق إجباري"
        performance_text = "غير محتسب"

    elif result_r > 0:

        result_text = (
            f"✅ WIN {format_r(result_r)}"
        )

        performance_text = "محتسب"

    elif result_r < 0:

        result_text = (
            f"❌ LOSS {format_r(result_r)}"
        )

        performance_text = "محتسب"

    else:

        result_text = "⚪ BREAKEVEN"
        performance_text = "محتسب"

    duration = int(
        (
            time.time()
            - state["opened_at"]
        ) / 60
    )

    if close_reason == "TP":
        reason_text = "TP"
    elif close_reason == "SL":
        reason_text = "SL"
    elif close_reason == "TIMEOUT":
        reason_text = "TIMEOUT"
    elif close_reason in [
        "CLOSE",
        "GEMINI_CLOSE",
    ]:
        reason_text = "Gemini CLOSE"
    elif close_reason in [
        "STOPPED",
        "FORCED",
        "MANUAL",
    ]:
        reason_text = "Forced"
    else:
        reason_text = close_reason

    modification_text = (
        "نعم"
        if state.get(
            "was_modified"
        )
        else "لا"
    )

    text = (
        "📊 ملخص نهاية الصفقة\n\n"

        f"الاتجاه: {trade['direction']}\n"
        f"Entry: {trade['entry']:.2f}\n"
        f"الإغلاق: {close_price:.2f}\n"
        f"سبب الإغلاق: {reason_text}\n\n"

        f"النتيجة: {result_text}\n"
        f"الإحصائيات: {performance_text}\n"
        f"مدة الصفقة: {duration} دقيقة\n\n"

        "🔧 إدارة الصفقة\n"
        f"تم تعديلها: {modification_text}\n"
        f"عدد التعديلات: "
        f"{state.get('modification_count', 0)}\n"
    )

    if state.get(
        "last_management_score"
    ) is not None:

        text += (
            f"آخر Score: "
            f"{state['last_management_score']:+d}\n"
        )

    # -----------------------------------------------------
    # CURRENT PERFORMANCE
    # -----------------------------------------------------

    total = performance_stats[
        "total_trades"
    ]

    wins = performance_stats[
        "wins"
    ]

    losses = performance_stats[
        "losses"
    ]

    total_r = performance_stats[
        "total_r"
    ]

    if total > 0:

        win_rate = (
            wins / total
        ) * 100

    else:

        win_rate = 0.0

    text += (
        "\n📈 الأداء الحالي\n"
        f"الصفقات المحتسبة: {total}\n"
        f"فوز: {wins}\n"
        f"خسارة: {losses}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Total P/L: {format_r(total_r)}\n"
    )

    return text


# =========================================================
# SEND TRADE RESULT SUMMARY
# =========================================================

async def send_closed_trade_summary(
    chat_id,
    state,
):

    await safe_send_message(
        chat_id,
        build_closed_trade_summary(
            state
        ),
        main_keyboard(),
    )


# =========================================================
# ANALYSIS
# =========================================================

async def perform_one_analysis(
    chat_id,
):

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

        state = analysis_tasks.get(
            chat_id
        )

        if (
            state
            and state.get(
                "stop_requested"
            )
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

            if price >= (
                entry - ENTRY_TOLERANCE
            ):

                reached = True

        else:

            if price <= (
                entry + ENTRY_TOLERANCE
            ):

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

    direction = trade[
        "direction"
    ]

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
# BUILD TRADE STATUS
# =========================================================

def build_trade_status(
    trade,
    price,
    age,
    state,
):

    direction = trade[
        "direction"
    ]

    entry = trade[
        "entry"
    ]

    sl = trade[
        "sl"
    ]

    tp = trade[
        "tp1"
    ]

    risk = abs(
        trade["original_sl"]
        - entry
    )

    if risk > 0:

        if direction == "BUY":

            unrealized_r = (
                price - entry
            ) / risk

        else:

            unrealized_r = (
                entry - price
            ) / risk

    else:

        unrealized_r = 0.0

    modification_text = (
        "نعم"
        if state.get(
            "was_modified"
        )
        else "لا"
    )

    score = state.get(
        "last_management_score"
    )

    score_text = (
        f"{score:+d}"
        if score is not None
        else "لم تتم مراجعة الإدارة بعد"
    )

    return (
        "📊 حالة الصفقة\n\n"
        f"الاتجاه: {direction}\n"
        f"السعر: {price:.2f}\n"
        f"Entry: {entry:.2f}\n"
        f"SL الحالي: {sl:.2f}\n"
        f"TP الحالي: {tp:.2f}\n"
        f"R:R الأصلي: {trade['rr']:.2f}\n"
        f"R الحالي التقريبي: {unrealized_r:+.2f}R\n"
        f"العمر: {int(age // 60)} دقيقة\n"
        f"آخر قوة إدارة: {score_text}\n"
        f"تم تعديل الصفقة: {modification_text}\n"
        f"عدد التعديلات: "
        f"{state.get('modification_count', 0)}"
    )


# =========================================================
# APPLY MANAGEMENT PROPOSAL
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

    if state.get(
        "closed"
    ):
        return

    trade = state[
        "trade"
    ]

    try:

        current_price = await asyncio.to_thread(
            get_live_gold_price
        )

    except Exception:

        state[
            "pending_proposal"
        ] = None

        return

    hit = trade_level_hit(
        trade,
        current_price,
    )

    if hit:

        state[
            "pending_proposal"
        ] = None

        return

    # -----------------------------------------------------
    # PRICE DRIFT PROTECTION
    # -----------------------------------------------------

    proposal_price = proposal.get(
        "price_at_proposal"
    )

    if proposal_price is not None:

        if abs(
            current_price
            - proposal_price
        ) > MAX_PROPOSAL_PRICE_DRIFT:

            state[
                "pending_proposal"
            ] = None

            return

    # -----------------------------------------------------
    # CLOSE
    # -----------------------------------------------------

    if proposal["action"] == "CLOSE":

        score = proposal[
            "score"
        ]

        # Losing close requires confirmation.
        current_r = calculate_trade_r(
            trade,
            current_price,
        )

        if current_r < 0:

            if score > MIN_SCORE_FOR_LOSS_CLOSE:

                state[
                    "pending_proposal"
                ] = None

                return

            confirmations = state.get(
                "negative_close_confirmations",
                0,
            )

            if (
                confirmations
                < LOSS_CLOSE_CONFIRMATIONS
            ):

                state[
                    "pending_proposal"
                ] = None

                return

        else:

            if score > MIN_SCORE_FOR_PROFIT_CLOSE:

                state[
                    "pending_proposal"
                ] = None

                return

        state[
            "pending_proposal"
        ] = None

        state[
            "closed"
        ] = True

        state[
            "close_reason"
        ] = "CLOSE"

        state[
            "close_price"
        ] = current_price

        performance_stats[
            "gemini_closes"
        ] += 1

        if current_r < 0:

            performance_stats[
                "gemini_close_losses"
            ] += 1

        else:

            performance_stats[
                "gemini_close_profit"
            ] += current_r

        record_closed_trade(
            state,
            current_price,
        )

        check_original_sl_avoided(
            state,
            current_price,
        )

        await send_closed_trade_summary(
            chat_id,
            state,
        )

        return

    # -----------------------------------------------------
    # HOLD
    # -----------------------------------------------------

    if proposal["action"] == "HOLD":

        state[
            "pending_proposal"
        ] = None

        return

    # -----------------------------------------------------
    # VALIDATE MODIFICATION
    # -----------------------------------------------------

    valid, reason = validate_management(
        trade,
        proposal,
        current_price,
    )

    if not valid:

        state[
            "pending_proposal"
        ] = None

        return

    old_sl = trade["sl"]

    old_tp = trade["tp1"]

    new_sl = proposal[
        "new_sl"
    ]

    new_tp = proposal[
        "new_tp"
    ]

    sl_changed = (
        abs(
            new_sl - old_sl
        ) > 0.00001
    )

    tp_changed = (
        abs(
            new_tp - old_tp
        ) > 0.00001
    )

    if not sl_changed and not tp_changed:

        state[
            "pending_proposal"
        ] = None

        return

    # -----------------------------------------------------
    # APPLY
    # -----------------------------------------------------

    if sl_changed:

        performance_stats[
            "sl_modifications"
        ] += 1

    if tp_changed:

        performance_stats[
            "tp_modifications"
        ] += 1

    performance_stats[
        "total_modifications"
    ] += 1

    state[
        "was_modified"
    ] = True

    state[
        "modification_count"
    ] = (
        state.get(
            "modification_count",
            0
        ) + 1
    )

    state[
        "last_management_score"
    ] = proposal[
        "score"
    ]

    performance_stats[
        "last_modification"
    ] = time.time()

    trade["sl"] = new_sl
    trade["tp1"] = new_tp

    state[
        "pending_proposal"
    ] = None

    await safe_send_message(
        chat_id,
        (
            "🔧 تم تعديل الصفقة\n\n"
            f"قوة الإدارة: "
            f"{proposal['score']:+d}/8\n"
            f"SL: {old_sl:.2f} → {new_sl:.2f}\n"
            f"TP: {old_tp:.2f} → {new_tp:.2f}\n\n"
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

    if state.get(
        "closed"
    ):
        return

    trade = state[
        "trade"
    ]

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

            return

        score = proposal[
            "score"
        ]

        state[
            "last_management_score"
        ] = score

        # -------------------------------------------------
        # CLOSE CONFIRMATION TRACKING
        # -------------------------------------------------

        current_r = calculate_trade_r(
            trade,
            current_price,
        )

        if (
            proposal["action"] == "CLOSE"
            and current_r < 0
            and score <= MIN_SCORE_FOR_LOSS_CLOSE
        ):

            state[
                "negative_close_confirmations"
            ] = (
                state.get(
                    "negative_close_confirmations",
                    0,
                ) + 1
            )

        else:

            state[
                "negative_close_confirmations"
            ] = 0

        # -------------------------------------------------
        # HOLD
        # -------------------------------------------------

        if proposal["action"] == "HOLD":

            return

        # -------------------------------------------------
        # MODIFICATION / CLOSE PROPOSAL
        # -------------------------------------------------

        global proposal_counter

        proposal_counter += 1

        proposal_id = proposal_counter

        state[
            "pending_proposal"
        ] = {
            "id": proposal_id,
            "action": proposal["action"],
            "score": proposal["score"],
            "new_sl": proposal["new_sl"],
            "new_tp": proposal["new_tp"],
            "reason": proposal["reason"],
            "created_at": time.time(),
            "price_at_proposal": current_price,
        }

        # -------------------------------------------------
        # CLOSE
        # -------------------------------------------------

        if proposal["action"] == "CLOSE":

            # Only create a real close proposal when
            # the internal gates have passed.

            if current_r < 0:

                if (
                    score > MIN_SCORE_FOR_LOSS_CLOSE
                    or state.get(
                        "negative_close_confirmations",
                        0,
                    ) < LOSS_CLOSE_CONFIRMATIONS
                ):

                    state[
                        "pending_proposal"
                    ] = None

                    return

            else:

                if score > MIN_SCORE_FOR_PROFIT_CLOSE:

                    state[
                        "pending_proposal"
                    ] = None

                    return

            asyncio.create_task(
                apply_management_proposal(
                    chat_id,
                    proposal_id,
                )
            )

            return

        # -------------------------------------------------
        # MODIFICATION
        # -------------------------------------------------

        valid, reason = validate_management(
            trade,
            proposal,
            current_price,
        )

        if not valid:

            state[
                "pending_proposal"
            ] = None

            return

        asyncio.create_task(
            apply_management_proposal(
                chat_id,
                proposal_id,
            )
        )

    except Exception as exc:

        print(
            "Management review error:",
            repr(exc),
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

    trade = state[
        "trade"
    ]

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

            state[
                "closed"
            ] = True

            state[
                "close_reason"
            ] = "STOPPED"

            try:

                price = await asyncio.to_thread(
                    get_live_gold_price
                )

            except Exception:

                price = trade["entry"]

            state[
                "close_price"
            ] = price

            state[
                "pending_proposal"
            ] = None

            record_closed_trade(
                state,
                price,
            )

            check_original_sl_avoided(
                state,
                price,
            )

            await send_closed_trade_summary(
                chat_id,
                state,
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

        hit = trade_level_hit(
            trade,
            price,
        )

        if hit:

            state[
                "closed"
            ] = True

            state[
                "close_reason"
            ] = hit

            state[
                "close_price"
            ] = price

            state[
                "pending_proposal"
            ] = None

            record_closed_trade(
                state,
                price,
            )

            check_original_sl_avoided(
                state,
                price,
            )

            await send_closed_trade_summary(
                chat_id,
                state,
            )

            return

        age = (
            time.time()
            - state["opened_at"]
        )

        # -------------------------------------------------
        # MAX DURATION
        # -------------------------------------------------

        if age >= MAX_TRADE_SECONDS:

            state[
                "closed"
            ] = True

            state[
                "close_reason"
            ] = "TIMEOUT"

            state[
                "close_price"
            ] = price

            state[
                "pending_proposal"
            ] = None

            record_closed_trade(
                state,
                price,
            )

            check_original_sl_avoided(
                state,
                price,
            )

            await send_closed_trade_summary(
                chat_id,
                state,
            )

            return

        # -------------------------------------------------
        # 5-MINUTE STATUS
        # -------------------------------------------------

        if (
            time.time() - last_status
            >= TRADE_STATUS_SECONDS
        ):

            remaining = max(
                0,
                MAX_TRADE_SECONDS
                - age,
            )

            await safe_send_message(
                chat_id,
                build_trade_status(
                    trade,
                    price,
                    age,
                    state,
                ),
                stop_keyboard(),
            )

            last_status = time.time()

        # -------------------------------------------------
        # 15-MINUTE MANAGEMENT
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

            # Do not start a new analysis while
            # an existing trade is being monitored.
            if trade_states.get(
                chat_id
            ):

                await asyncio.sleep(
                    PRICE_POLL_SECONDS
                )

                continue

            number += 1

            try:

                result = await perform_one_analysis(
                    chat_id
                )

            except Exception as exc:

                print(
                    "Analysis error:",
                    repr(exc),
                )

                await safe_send_message(
                    chat_id,
                    (
                        "⚠️ فشل التحليل.\n\n"
                        f"{type(exc).__name__}: "
                        f"{str(exc)[:500]}"
                    ),
                    stop_keyboard(),
                )

                await asyncio.sleep(
                    REANALYSIS_WAIT_SECONDS
                )

                continue

            raw = result[
                "raw"
            ]

            trade = result[
                "trade"
            ]

            price = result[
                "price"
            ]

            if not trade:

                await safe_send_message(
                    chat_id,
                    (
                        "🚫 NO TRADE\n\n"
                        f"السعر: {price:.2f}\n"
                        f"{extract_reason(raw)}"
                    ),
                    stop_keyboard(),
                )

                await asyncio.sleep(
                    REANALYSIS_WAIT_SECONDS
                )

                continue

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
                    "⏳ انتظار وصول السعر إلى Entry "
                    "لمدة أقصاها 10 دقائق."
                ),
                stop_keyboard(),
            )

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
                        "خلال 10 دقائق.\n"
                        "تم إلغاء الـsetup."
                    ),
                    stop_keyboard(),
                )

                await asyncio.sleep(
                    REANALYSIS_WAIT_SECONDS
                )

                continue

            trade_state = {
                "trade": {
                    "direction": trade[
                        "direction"
                    ],
                    "entry": trade[
                        "entry"
                    ],
                    "sl": trade[
                        "sl"
                    ],
                    "tp1": trade[
                        "tp1"
                    ],
                    "rr": trade[
                        "rr"
                    ],
                    "original_sl": trade[
                        "sl"
                    ],
                    "original_tp": trade[
                        "tp1"
                    ],

                    # Kept for record only.
                    # Management model does NOT receive it.
                    "reason": trade[
                        "reason"
                    ],
                },

                "opened_at": time.time(),

                "closed": False,

                "stop_requested": False,

                "pending_proposal": None,

                "was_modified": False,

                "modification_count": 0,

                "adjusted_r": None,

                "final_original_r": None,

                "counted_in_performance": False,

                "last_management_score": None,

                "negative_close_confirmations": 0,
            }

            trade_states[
                chat_id
            ] = trade_state

            await monitor_trade(
                chat_id
            )

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

        application = (
            ApplicationHolder.application
        )

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

    task = (
        analysis_tasks.get(
            chat_id,
            {},
        ).get(
            "task"
        )
    )

    if task:

        current = asyncio.current_task()

        if task != current:

            task.cancel()

    # Do NOT remove trade state here.
    # monitor_trade must receive the stop request
    # and record the forced closure correctly.

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

    # -----------------------------------------------------
    # CURRENT TRADE
    # -----------------------------------------------------

    if state:

        trade = state[
            "trade"
        ]

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

        score = state.get(
            "last_management_score"
        )

        if score is not None:

            text += (
                f"\nقوة الإدارة: {score:+d}/8"
            )

        text += (
            "\n\n🟢 الصفقة تحت المراقبة."
        )

        await safe_send_message(
            chat_id,
            text,
            stop_keyboard(),
        )

        return

    # -----------------------------------------------------
    # PERFORMANCE
    # -----------------------------------------------------

    total = performance_stats[
        "total_trades"
    ]

    wins = performance_stats[
        "wins"
    ]

    losses = performance_stats[
        "losses"
    ]

    other = performance_stats[
        "other_closes"
    ]

    total_r = performance_stats[
        "total_r"
    ]

    if total > 0:

        win_rate = (
            wins / total
        ) * 100

    else:

        win_rate = 0.0

    adjusted_total = performance_stats[
        "adjusted_trades"
    ]

    adjusted_total_r = performance_stats[
        "adjusted_total_r"
    ]

    avoided = performance_stats[
        "avoided_original_sl"
    ]

    modifications = performance_stats[
        "total_modifications"
    ]

    sl_modifications = performance_stats[
        "sl_modifications"
    ]

    tp_modifications = performance_stats[
        "tp_modifications"
    ]

    forced_closes = performance_stats[
        "forced_closes"
    ]

    gemini_closes = performance_stats[
        "gemini_closes"
    ]

    gemini_close_losses = performance_stats[
        "gemini_close_losses"
    ]

    gemini_close_profit = performance_stats[
        "gemini_close_profit"
    ]

    text = (
        "📊 RustyGold Performance\n\n"

        f"الصفقات المحتسبة: {total}\n"
        f"الفوز: {wins}\n"
        f"الخسارة: {losses}\n"
        f"إغلاقات أخرى: {other}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Total P/L: {format_r(total_r)}\n\n"

        "🔧 إدارة Gemini\n\n"
        f"صفقات تم تعديلها: {adjusted_total}\n"
        f"إجمالي التعديلات: {modifications}\n"
        f"تعديلات SL: {sl_modifications}\n"
        f"تعديلات TP: {tp_modifications}\n"
        f"P/L الصفقات المعدلة: "
        f"{format_r(adjusted_total_r)}\n\n"

        "🧠 إغلاقات Gemini\n\n"
        f"إغلاقات Gemini: {gemini_closes}\n"
        f"إغلاقات Gemini الخاسرة: "
        f"{gemini_close_losses}\n"
        f"P/L إغلاقات Gemini الرابحة: "
        f"{format_r(gemini_close_profit)}\n\n"

        "🛡️ ما تم تجنبه\n\n"
        f"صفقات تجنبت SL الأصلي: {avoided}\n"
        f"إغلاقات إجبارية غير محتسبة: "
        f"{forced_closes}\n"
    )

    if analysis:

        text += (
            "\n🔎 التحليل يعمل حاليًا."
        )

    else:

        text += (
            "\nℹ️ لا يوجد تحليل يعمل حاليًا."
        )

    await safe_send_message(
        chat_id,
        text,
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
    # START
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
                "Gemini mode = Balanced/Aggressive\n"
                "Analysis = 1H + 5M + 1M\n\n"
                "تحليل الإدارة يعمل داخليًا بدون "
                "رسائل Gemini المرحلية."
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

        if trade_states.get(
            chat_id
        ):

            await query.message.reply_text(
                "🛑 تم طلب إيقاف الصفقة والمراقبة.",
                reply_markup=stop_keyboard(),
            )

        else:

            await query.message.reply_text(
                "🛑 تم إيقاف التحليل.",
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
    # REJECT MANAGEMENT
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

    ApplicationHolder.application = (
        application
    )

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

    print(
        "Gemini timeout:",
        GEMINI_REQUEST_TIMEOUT_MS,
        "ms",
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":

    main()