import os
import io
import re
import time
import asyncio
import threading
import requests
import pandas as pd
import mplfinance as mpf

from flask import Flask
from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardMarkup,
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
from google import genai


# ============================================================
# CONFIG
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


# Minimum acceptable Risk/Reward ratio.
# Example:
# Risk = 10
# Reward = 5
# RR = 0.50 -> rejected
MIN_RR = 1.50

# Price polling interval.
PRICE_POLL_SECONDS = 5

# Gemini trade-management review interval.
MANAGEMENT_REVIEW_SECONDS = 15 * 60

# Time allowed to reject a proposed SL/TP modification.
MODIFICATION_REJECTION_SECONDS = 60

# Maximum trade duration.
MAX_TRADE_SECONDS = 75 * 60

# How close price must be to suggested entry to consider entry detected.
ENTRY_TOLERANCE = 0.60

# Maximum time to wait for entry.
ENTRY_WAIT_SECONDS = 5 * 60


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(api_key=GEMINI_KEY)

gemini_model = None


def discover_gemini_model():
    global gemini_model

    preferred = [
        "gemini-3.8-flash",
        "gemini-3.8-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    try:
        models = list(client.models.list())

        available = []

        for model in models:
            name = getattr(model, "name", "")

            if not name:
                continue

            if name.startswith("models/"):
                name = name.replace("models/", "", 1)

            available.append(name)

        for wanted in preferred:
            for available_model in available:
                if available_model == wanted:
                    gemini_model = available_model
                    return gemini_model

        for available_model in available:
            low = available_model.lower()

            if "flash" in low:
                gemini_model = available_model
                return gemini_model

        if available:
            gemini_model = available[0]
            return gemini_model

    except Exception as e:
        print("Gemini model discovery error:", e)

    gemini_model = "gemini-3.8-flash"
    return gemini_model


discover_gemini_model()


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "OK - RustyGold Bot Live"


@app.route("/health")
def health():
    return "healthy"


def run_flask():
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


# ============================================================
# GLOBAL STATE
# ============================================================

analysis_tasks = {}
trade_states = {}
management_proposals = {}

stats = {
    "ideas": 0,
    "accepted_ideas": 0,
    "rejected_rr": 0,
    "entered": 0,
    "wins": 0,
    "losses": 0,
}


# ============================================================
# TELEGRAM KEYBOARDS
# ============================================================

main_keyboard = ReplyKeyboardMarkup(
    [
        ["🚀 حلل يا جيمني"],
        ["🛑 إيقاف", "📊 ملخص"],
    ],
    resize_keyboard=True,
)


def stop_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🛑 إيقاف التحليل",
                    callback_data="stop_analysis",
                )
            ]
        ]
    )


def modification_keyboard(proposal_id):
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "❌ لا تنفذ التعديل",
                    callback_data=f"reject_mod:{proposal_id}",
                )
            ]
        ]
    )


# ============================================================
# TWELVEDATA
# ============================================================

def get_ohlc(interval, outputsize):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
    }

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"TwelveData error: {data}")

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(df["datetime"])

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("datetime")

    df = df.set_index("datetime")

    return df[["open", "high", "low", "close"]]


# ============================================================
# CURRENT GOLD PRICE
# ============================================================

def get_gold_price():
    url = "https://api.gold-api.com/price/XAU"

    response = requests.get(url, timeout=15)
    data = response.json()

    price = data.get("price")

    if price is None:
        raise RuntimeError(f"Gold API error: {data}")

    return float(price)


# ============================================================
# CHART CREATION
# ============================================================

def make_chart(df, title):
    buffer = io.BytesIO()

    chart_df = df.tail(100).copy()

    mpf.plot(
        chart_df,
        type="candle",
        style="charles",
        volume=False,
        title=title,
        figsize=(12, 6),
        savefig=dict(
            fname=buffer,
            dpi=120,
            bbox_inches="tight",
        ),
    )

    buffer.seek(0)

    return buffer


# ============================================================
# DATA FORMAT FOR GEMINI
# ============================================================

def dataframe_to_text(df, limit=100):
    data = df.tail(limit).copy()

    lines = []

    for index, row in data.iterrows():
        lines.append(
            f"{index} | "
            f"O={row['open']:.2f} "
            f"H={row['high']:.2f} "
            f"L={row['low']:.2f} "
            f"C={row['close']:.2f}"
        )

    return "\n".join(lines)


# ============================================================
# GEMINI CALL
# ============================================================

def ask_gemini(contents):
    global gemini_model

    try:
        response = client.models.generate_content(
            model=gemini_model,
            contents=contents,
        )

        return response.text.strip()

    except Exception as e:
        error_text = str(e)

        print("Gemini error:", error_text)

        # Try discovering another model once.
        discover_gemini_model()

        try:
            response = client.models.generate_content(
                model=gemini_model,
                contents=contents,
            )

            return response.text.strip()

        except Exception as retry_error:
            print("Gemini retry error:", retry_error)
            raise


# ============================================================
# INITIAL TRADE ANALYSIS PROMPT
# ============================================================

TRADE_PROMPT = """
You are RustyGold, an XAU/USD short-term trading analyst.

Analyze the market using price action and ICT concepts.

TIMEFRAME HIERARCHY:
1H = overall context
5M = market structure
1M = entry timing

Look for:
- BOS
- CHoCH
- liquidity sweep
- false breakout
- FVG
- order blocks
- displacement
- support/resistance
- continuation or reversal

Do NOT force a trade.

A trade is valid only when there is a reasonable combination of confirmations.

IMPORTANT RISK RULE:

The trade must have a minimum Risk/Reward ratio of 1.5.

Calculate:

For BUY:
Risk = ENTRY - SL
Reward = TP1 - ENTRY

For SELL:
Risk = SL - ENTRY
Reward = ENTRY - TP1

If Risk <= 0 or Reward <= 0, output NO TRADE.

If Reward / Risk < 1.5, output NO TRADE.

DO NOT artificially move TP just to satisfy the ratio.

Do not invent a trade simply because the market is moving.

Return exactly one of:

TRADE
DIRECTION: BUY or SELL
ENTRY: number
SL: number
TP1: number
RR: number
REASON: short explanation

OR:

NO TRADE
REASON: short explanation
"""


# ============================================================
# TRANSLATION
# ============================================================

TRANSLATION_PROMPT = """
Translate the following trading analysis into natural Arabic.

Keep these technical words in English:
BOS
CHoCH
FVG
OB
ENTRY
SL
TP
RR
BUY
SELL
TRADE
NO TRADE
KEEP
MOVE_SL
MOVE_TP
MOVE_SL_AND_TP
CLOSE_PROFIT
CLOSE_LOSS
CLOSE_NOW

Do not change any numbers.

Return only the translated analysis.

TEXT:
"""


def translate_analysis(text):
    try:
        return ask_gemini(
            TRANSLATION_PROMPT + "\n" + text
        )
    except Exception:
        return text


# ============================================================
# TRADE PARSER
# ============================================================

def extract_number(text, key):
    pattern = rf"{key}\s*:\s*(-?\d+(?:\.\d+)?)"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE,
    )

    if not match:
        return None

    return float(match.group(1))


def parse_trade(text):
    if not re.search(r"\bTRADE\b", text, re.IGNORECASE):
        return None

    if re.search(r"\bNO\s+TRADE\b", text, re.IGNORECASE):
        return None

    direction_match = re.search(
        r"DIRECTION\s*:\s*(BUY|SELL)",
        text,
        re.IGNORECASE,
    )

    if not direction_match:
        return None

    direction = direction_match.group(1).upper()

    entry = extract_number(text, "ENTRY")
    sl = extract_number(text, "SL")
    tp = extract_number(text, "TP1")

    if entry is None or sl is None or tp is None:
        return None

    if direction == "BUY":

        if sl >= entry:
            return None

        if tp <= entry:
            return None

        risk = entry - sl
        reward = tp - entry

    else:

        if sl <= entry:
            return None

        if tp >= entry:
            return None

        risk = sl - entry
        reward = entry - tp

    if risk <= 0 or reward <= 0:
        return None

    rr = reward / risk

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "risk": risk,
        "reward": reward,
        "rr": rr,
        "raw": text,
    }


# ============================================================
# R:R FILTER
# ============================================================

def passes_rr_filter(trade):
    if not trade:
        return False

    rr = trade["rr"]

    print(
        f"Trade RR = {rr:.2f} | "
        f"Minimum RR = {MIN_RR:.2f}"
    )

    return rr >= MIN_RR


# ============================================================
# TRADE MANAGEMENT PROMPT
# ============================================================

MANAGEMENT_PROMPT = """
You are managing an EXISTING XAU/USD scalp trade.

This is NOT a new trade.

Your job is to reassess the existing position every 15 minutes.

Do NOT automatically close or modify a trade just because you can.

Compare the CURRENT market structure against the ORIGINAL trade thesis.

Possible actions:

KEEP
MOVE_SL
MOVE_TP
MOVE_SL_AND_TP
CLOSE_PROFIT
CLOSE_LOSS
CLOSE_NOW

Rules:

1. KEEP means current SL and TP remain unchanged.

2. MOVE_SL means propose a new SL.
The new SL must make structural sense.

3. MOVE_TP means propose a new TP.
Do not move TP farther merely to chase price.

4. MOVE_SL_AND_TP means propose both.

5. CLOSE_PROFIT means close while profitable.

6. CLOSE_LOSS means close to prevent further deterioration.

7. CLOSE_NOW means close immediately regardless of profit/loss.

8. Do not propose a modification merely for activity.

9. Maximum intended trade duration is 75 minutes.

10. Once trade age reaches around 60-75 minutes,
strongly consider closing unless there is a very strong reason to continue.

11. Protect profits when market structure changes.

12. Do not turn a winning trade into a large losing trade.

13. Do not widen SL merely to avoid being stopped.

14. Do not create a new trade.

Return exactly:

ACTION: KEEP
NEW_SL: current
NEW_TP: current
REASON: explanation

OR

ACTION: MOVE_SL
NEW_SL: number
NEW_TP: current
REASON: explanation

OR

ACTION: MOVE_TP
NEW_SL: current
NEW_TP: number
REASON: explanation

OR

ACTION: MOVE_SL_AND_TP
NEW_SL: number
NEW_TP: number
REASON: explanation

OR

ACTION: CLOSE_PROFIT
NEW_SL: current
NEW_TP: current
REASON: explanation

OR

ACTION: CLOSE_LOSS
NEW_SL: current
NEW_TP: current
REASON: explanation

OR

ACTION: CLOSE_NOW
NEW_SL: current
NEW_TP: current
REASON: explanation
"""


# ============================================================
# MANAGEMENT PARSER
# ============================================================

def parse_management(text):
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

    new_sl = extract_number(text, "NEW_SL")
    new_tp = extract_number(text, "NEW_TP")

    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    reason = (
        reason_match.group(1).strip()
        if reason_match
        else ""
    )

    return {
        "action": action,
        "new_sl": new_sl,
        "new_tp": new_tp,
        "reason": reason,
        "raw": text,
    }


# ============================================================
# VALIDATE MANAGEMENT
# ============================================================

def validate_management(management, trade):
    if not management:
        return False

    action = management["action"]

    if action == "KEEP":
        return True

    if action in {
        "CLOSE_PROFIT",
        "CLOSE_LOSS",
        "CLOSE_NOW",
    }:
        return True

    new_sl = management["new_sl"]
    new_tp = management["new_tp"]

    if action == "MOVE_SL" and new_sl is None:
        return False

    if action == "MOVE_TP" and new_tp is None:
        return False

    if action == "MOVE_SL_AND_TP":
        if new_sl is None or new_tp is None:
            return False

    direction = trade["direction"]
    entry = trade["entry"]

    if new_sl is None:
        new_sl = trade["sl"]

    if new_tp is None:
        new_tp = trade["tp"]

    if direction == "BUY":

        if new_sl >= entry:
            # Do not allow the management system
            # to turn a normal protective SL into
            # an invalid level.
            return False

        if new_tp <= entry:
            return False

        risk = entry - new_sl
        reward = new_tp - entry

    else:

        if new_sl <= entry:
            return False

        if new_tp >= entry:
            return False

        risk = new_sl - entry
        reward = entry - new_tp

    if risk <= 0 or reward <= 0:
        return False

    return True


# ============================================================
# PNL
# ============================================================

def calculate_pnl(trade, exit_price):
    entry = trade["entry"]
    direction = trade["direction"]

    if direction == "BUY":
        return exit_price - entry

    return entry - exit_price


def format_pnl(pnl):
    if pnl > 0:
        return f"+{pnl:.2f}"

    return f"{pnl:.2f}"


# ============================================================
# SEND TRADE
# ============================================================

async def send_trade_message(bot, chat_id, trade):
    text = (
        "🚨 صفقة مقترحة\n\n"
        f"📌 الاتجاه: {trade['direction']}\n"
        f"🎯 ENTRY: {trade['entry']:.2f}\n"
        f"🛑 SL: {trade['sl']:.2f}\n"
        f"💰 TP1: {trade['tp']:.2f}\n"
        f"📊 R:R: {trade['rr']:.2f}\n\n"
        "✅ الصفقة اجتازت فلتر المخاطرة.\n"
        "⚠️ البوت لا يفتح الصفقة في حسابك.\n"
        "المتابعة تبدأ عند اقتراب السعر من ENTRY."
    )

    await bot.send_message(
        chat_id=chat_id,
        text=text,
        reply_markup=stop_keyboard(),
    )


# ============================================================
# WAIT FOR ENTRY
# ============================================================

async def wait_for_entry(bot, chat_id, trade, state):
    start = time.time()

    while time.time() - start < ENTRY_WAIT_SECONDS:

        if state["stop_requested"]:
            return None

        try:
            price = await asyncio.to_thread(
                get_gold_price
            )
        except Exception as e:
            print("Entry price error:", e)
            await asyncio.sleep(PRICE_POLL_SECONDS)
            continue

        state["last_price"] = price

        distance = abs(price - trade["entry"])

        if distance <= ENTRY_TOLERANCE:

            state["entered"] = True
            state["entry_time"] = time.time()
            state["entry_price"] = price

            stats["entered"] += 1

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "🟢 تم رصد الدخول\n\n"
                    f"📌 الاتجاه: {trade['direction']}\n"
                    f"💵 سعر الدخول الفعلي: {price:.2f}\n"
                    f"🛑 SL: {state['current_sl']:.2f}\n"
                    f"🎯 TP: {state['current_tp']:.2f}\n\n"
                    "🔄 بدأت مراقبة الصفقة."
                ),
                reply_markup=stop_keyboard(),
            )

            return price

        await asyncio.sleep(PRICE_POLL_SECONDS)

    await bot.send_message(
        chat_id=chat_id,
        text=(
            "⌛ انتهت مدة انتظار الدخول.\n"
            "لم يقترب السعر من ENTRY، لذلك لن يتم تتبع الصفقة."
        ),
    )

    return None


# ============================================================
# APPLY MANAGEMENT
# ============================================================

async def apply_management(
    bot,
    chat_id,
    state,
    management,
):
    action = management["action"]

    if action == "KEEP":
        return

    if action in {
        "CLOSE_PROFIT",
        "CLOSE_LOSS",
        "CLOSE_NOW",
    }:

        try:
            price = await asyncio.to_thread(
                get_gold_price
            )
        except Exception:
            price = state["last_price"]

        pnl = calculate_pnl(
            state,
            price,
        )

        state["closed"] = True
        state["close_price"] = price
        state["close_reason"] = action
        state["pnl"] = pnl

        if pnl >= 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "🔴 تم إغلاق الصفقة حسب مراجعة Gemini\n\n"
                f"📌 السبب: {action}\n"
                f"💵 السعر: {price:.2f}\n"
                f"📊 النتيجة: {format_pnl(pnl)}"
            ),
        )

        return

    new_sl = management["new_sl"]
    new_tp = management["new_tp"]

    if new_sl is None:
        new_sl = state["current_sl"]

    if new_tp is None:
        new_tp = state["current_tp"]

    old_sl = state["current_sl"]
    old_tp = state["current_tp"]

    state["current_sl"] = new_sl
    state["current_tp"] = new_tp

    await bot.send_message(
        chat_id=chat_id,
        text=(
            "⚙️ تم تطبيق تعديل Gemini تلقائيًا\n\n"
            f"📌 الإجراء: {action}\n\n"
            f"🛑 SL السابق: {old_sl:.2f}\n"
            f"🛑 SL الجديد: {new_sl:.2f}\n\n"
            f"🎯 TP السابق: {old_tp:.2f}\n"
            f"🎯 TP الجديد: {new_tp:.2f}\n\n"
            f"💡 {management['reason']}\n\n"
            "👁️ المراقبة الآن على المستويات الجديدة."
        ),
        reply_markup=stop_keyboard(),
    )


# ============================================================
# MANAGEMENT PROPOSAL
# ============================================================

async def create_management_proposal(
    bot,
    chat_id,
    state,
    management,
):
    proposal_id = (
        f"{chat_id}_{int(time.time() * 1000)}"
    )

    proposal = {
        "id": proposal_id,
        "chat_id": chat_id,
        "state": state,
        "management": management,
        "created": time.time(),
        "active": True,
    }

    management_proposals[proposal_id] = proposal

    action = management["action"]

    if action == "KEEP":

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "🔎 مراجعة Gemini بعد 15 دقيقة\n\n"
                "✅ القرار: KEEP\n\n"
                "لا يوجد تعديل على SL/TP."
            ),
            reply_markup=stop_keyboard(),
        )

        return

    if action in {
        "CLOSE_PROFIT",
        "CLOSE_LOSS",
        "CLOSE_NOW",
    }:

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ Gemini يقترح إغلاق الصفقة\n\n"
                f"📌 القرار: {action}\n"
                f"💡 {management['reason']}\n\n"
                "سيتم الإغلاق حسب قرار الإدارة بعد انتهاء "
                "مهلة الرفض."
            ),
            reply_markup=modification_keyboard(proposal_id),
        )

    else:

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "🔎 مراجعة Gemini\n\n"
                f"📌 القرار: {action}\n\n"
                f"🛑 SL الحالي: {state['current_sl']:.2f}\n"
                f"🎯 TP الحالي: {state['current_tp']:.2f}\n\n"
                f"🛑 SL المقترح: "
                f"{management['new_sl']:.2f}\n"
                f"🎯 TP المقترح: "
                f"{management['new_tp']:.2f}\n\n"
                f"💡 {management['reason']}\n\n"
                "⏳ لديك 60 ثانية لرفض التعديل.\n"
                "إذا لم تضغط الزر، سيُطبَّق تلقائيًا."
            ),
            reply_markup=modification_keyboard(proposal_id),
        )

    asyncio.create_task(
        finalize_management_proposal(
            bot,
            proposal_id,
        )
    )


async def finalize_management_proposal(
    bot,
    proposal_id,
):
    await asyncio.sleep(
        MODIFICATION_REJECTION_SECONDS
    )

    proposal = management_proposals.get(
        proposal_id
    )

    if not proposal:
        return

    if not proposal["active"]:
        return

    proposal["active"] = False

    state = proposal["state"]
    management = proposal["management"]
    chat_id = proposal["chat_id"]

    if state["closed"] or state["stop_requested"]:
        return

    await apply_management(
        bot,
        chat_id,
        state,
        management,
    )

    try:
        await bot.edit_message_reply_markup(
            chat_id=chat_id,
            message_id=state.get(
                "proposal_message_id"
            ),
            reply_markup=None,
        )
    except Exception:
        pass


# ============================================================
# MANAGEMENT ANALYSIS
# ============================================================

async def perform_management_review(
    bot,
    chat_id,
    state,
):
    if state["closed"]:
        return

    if state["stop_requested"]:
        return

    try:
        price = await asyncio.to_thread(
            get_gold_price
        )

        df_1h = await asyncio.to_thread(
            get_ohlc,
            "1h",
            120,
        )

        df_5m = await asyncio.to_thread(
            get_ohlc,
            "5min",
            150,
        )

        df_1m = await asyncio.to_thread(
            get_ohlc,
            "1min",
            150,
        )

    except Exception as e:
        print("Management data error:", e)

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ تعذر جلب بيانات المراجعة.\n"
                "ستستمر مراقبة SL/TP بالسعر الحالي."
            ),
        )

        return

    state["last_price"] = price

    age_seconds = (
        time.time() - state["entry_time"]
    )

    age_minutes = age_seconds / 60

    pnl = calculate_pnl(
        state,
        price,
    )

    prompt = (
        MANAGEMENT_PROMPT
        + "\n\n"
        + "CURRENT TRADE:\n"
        + f"Direction: {state['direction']}\n"
        + f"Original Entry: {state['entry']:.2f}\n"
        + f"Original SL: {state['original_sl']:.2f}\n"
        + f"Original TP: {state['original_tp']:.2f}\n"
        + f"Current SL: {state['current_sl']:.2f}\n"
        + f"Current TP: {state['current_tp']:.2f}\n"
        + f"Current Price: {price:.2f}\n"
        + f"Current PnL: {pnl:.2f}\n"
        + f"Trade Age Minutes: {age_minutes:.1f}\n\n"
        + "1H DATA:\n"
        + dataframe_to_text(df_1h, 80)
        + "\n\n"
        + "5M DATA:\n"
        + dataframe_to_text(df_5m, 100)
        + "\n\n"
        + "1M DATA:\n"
        + dataframe_to_text(df_1m, 100)
    )

    try:
        raw = await asyncio.to_thread(
            ask_gemini,
            prompt,
        )

        management = parse_management(raw)

    except Exception as e:
        print("Management Gemini error:", e)
        return

    if not validate_management(
        management,
        state,
    ):
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ Gemini أعطى تعديلًا غير صالح.\n"
                "تم تجاهله والإبقاء على SL/TP الحاليين."
            ),
        )

        return

    # Maximum duration protection.
    # At 60 minutes, tell Gemini to strongly consider closing.
    if age_seconds >= MAX_TRADE_SECONDS:

        try:
            price = await asyncio.to_thread(
                get_gold_price
            )
        except Exception:
            price = state["last_price"]

        pnl = calculate_pnl(
            state,
            price,
        )

        state["closed"] = True
        state["close_price"] = price
        state["close_reason"] = "MAX_TIME"
        state["pnl"] = pnl

        if pnl >= 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⏰ انتهى الحد الأقصى لمدة الصفقة.\n\n"
                f"💵 السعر: {price:.2f}\n"
                f"📊 النتيجة: {format_pnl(pnl)}\n\n"
                "🔴 تم إنهاء المتابعة."
            ),
        )

        return

    await create_management_proposal(
        bot,
        chat_id,
        state,
        management,
    )


# ============================================================
# PRICE MONITOR
# ============================================================

async def monitor_trade(
    bot,
    chat_id,
    state,
):
    next_review = (
        time.time()
        + MANAGEMENT_REVIEW_SECONDS
    )

    while not state["closed"]:

        if state["stop_requested"]:
            break

        try:
            price = await asyncio.to_thread(
                get_gold_price
            )

            state["last_price"] = price

        except Exception as e:
            print("Monitoring price error:", e)

            await asyncio.sleep(
                PRICE_POLL_SECONDS
            )

            continue

        direction = state["direction"]
        sl = state["current_sl"]
        tp = state["current_tp"]

        hit = None

        if direction == "BUY":

            if price <= sl:
                hit = "SL"

            elif price >= tp:
                hit = "TP"

        else:

            if price >= sl:
                hit = "SL"

            elif price <= tp:
                hit = "TP"

        if hit:

            pnl = calculate_pnl(
                state,
                price,
            )

            state["closed"] = True
            state["close_price"] = price
            state["close_reason"] = hit
            state["pnl"] = pnl

            if pnl >= 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"{'🎯' if hit == 'TP' else '🛑'} "
                    f"تم ضرب {hit}\n\n"
                    f"📌 الاتجاه: {direction}\n"
                    f"💵 سعر الخروج: {price:.2f}\n"
                    f"📊 PnL: {format_pnl(pnl)}\n\n"
                    "🔴 انتهت متابعة الصفقة."
                ),
            )

            break

        # ====================================================
        # 15-MINUTE GEMINI REVIEW
        # ====================================================

        if time.time() >= next_review:

            await perform_management_review(
                bot,
                chat_id,
                state,
            )

            if state["closed"]:
                break

            next_review = (
                time.time()
                + MANAGEMENT_REVIEW_SECONDS
            )

        # ====================================================
        # MAX DURATION CHECK
        # ====================================================

        age = (
            time.time()
            - state["entry_time"]
        )

        if age >= MAX_TRADE_SECONDS:

            try:
                exit_price = await asyncio.to_thread(
                    get_gold_price
                )
            except Exception:
                exit_price = price

            pnl = calculate_pnl(
                state,
                exit_price,
            )

            state["closed"] = True
            state["close_price"] = exit_price
            state["close_reason"] = "MAX_TIME"
            state["pnl"] = pnl

            if pnl >= 0:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⏰ وصلت الصفقة للحد الأقصى للمدة.\n\n"
                    f"💵 سعر الإغلاق: {exit_price:.2f}\n"
                    f"📊 PnL: {format_pnl(pnl)}\n\n"
                    "🔴 تم إغلاق المتابعة."
                ),
            )

            break

        await asyncio.sleep(
            PRICE_POLL_SECONDS
        )


# ============================================================
# ANALYSIS LOOP
# ============================================================

async def analysis_loop(
    bot,
    chat_id,
):
    try:

        while True:

            if chat_id not in analysis_tasks:
                break

            state_task = analysis_tasks.get(
                chat_id
            )

            if state_task is None:
                break

            if state_task["stop_requested"]:
                break

            try:

                current_price = await asyncio.to_thread(
                    get_gold_price
                )

                df_1h = await asyncio.to_thread(
                    get_ohlc,
                    "1h",
                    120,
                )

                df_5m = await asyncio.to_thread(
                    get_ohlc,
                    "5min",
                    150,
                )

                df_1m = await asyncio.to_thread(
                    get_ohlc,
                    "1min",
                    150,
                )

                chart_1h = await asyncio.to_thread(
                    make_chart,
                    df_1h,
                    "XAU/USD 1H",
                )

                chart_5m = await asyncio.to_thread(
                    make_chart,
                    df_5m,
                    "XAU/USD 5M",
                )

                chart_1m = await asyncio.to_thread(
                    make_chart,
                    df_1m,
                    "XAU/USD 1M",
                )

                prompt = (
                    TRADE_PROMPT
                    + "\n\n"
                    + f"CURRENT GOLD PRICE: "
                    f"{current_price:.2f}\n\n"
                    + "1H OHLC:\n"
                    + dataframe_to_text(df_1h, 100)
                    + "\n\n"
                    + "5M OHLC:\n"
                    + dataframe_to_text(df_5m, 120)
                    + "\n\n"
                    + "1M OHLC:\n"
                    + dataframe_to_text(df_1m, 120)
                )

                contents = [
                    prompt,
                    {
                        "mime_type": "image/png",
                        "data": chart_1h.getvalue(),
                    },
                    {
                        "mime_type": "image/png",
                        "data": chart_5m.getvalue(),
                    },
                    {
                        "mime_type": "image/png",
                        "data": chart_1m.getvalue(),
                    },
                ]

                raw = await asyncio.to_thread(
                    ask_gemini,
                    contents,
                )

                stats["ideas"] += 1

                trade = parse_trade(raw)

                if not trade:

                    arabic = translate_analysis(
                        raw
                    )

                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "🔎 تحليل جديد\n\n"
                            + arabic
                        ),
                        reply_markup=stop_keyboard(),
                    )

                    await asyncio.sleep(
                        MANAGEMENT_REVIEW_SECONDS
                    )

                    continue

                # ====================================================
                # PYTHON R:R FILTER
                # ====================================================

                if not passes_rr_filter(trade):

                    stats["rejected_rr"] += 1

                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "🚫 تم رفض الصفقة تلقائيًا\n\n"
                            f"📌 الاتجاه: {trade['direction']}\n"
                            f"ENTRY: {trade['entry']:.2f}\n"
                            f"SL: {trade['sl']:.2f}\n"
                            f"TP: {trade['tp']:.2f}\n"
                            f"📊 R:R = {trade['rr']:.2f}\n\n"
                            f"❌ الحد الأدنى المطلوب: "
                            f"{MIN_RR:.2f}\n\n"
                            "السبب: المخاطرة أعلى من العائد."
                        ),
                        reply_markup=stop_keyboard(),
                    )

                    await asyncio.sleep(
                        60
                    )

                    continue

                stats["accepted_ideas"] += 1

                arabic = translate_analysis(
                    raw
                )

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "🧠 تحليل Gemini\n\n"
                        + arabic
                        + "\n\n"
                        + f"📊 R:R محسوب: "
                        f"{trade['rr']:.2f}\n"
                        + "✅ اجتازت الصفقة فلتر المخاطرة."
                    ),
                    reply_markup=stop_keyboard(),
                )

                await send_trade_message(
                    bot,
                    chat_id,
                    trade,
                )

                # ====================================================
                # CREATE TRADE STATE
                # ====================================================

                state = {
                    "direction": trade["direction"],

                    "entry": trade["entry"],
                    "original_sl": trade["sl"],
                    "original_tp": trade["tp"],

                    "current_sl": trade["sl"],
                    "current_tp": trade["tp"],

                    "entry_price": None,
                    "entry_time": None,

                    "last_price": current_price,

                    "entered": False,
                    "closed": False,
                    "stop_requested": False,

                    "close_price": None,
                    "close_reason": None,
                    "pnl": None,
                }

                trade_states[chat_id] = state

                # ====================================================
                # WAIT FOR PRICE TO REACH ENTRY
                # ====================================================

                entry_price = await wait_for_entry(
                    bot,
                    chat_id,
                    trade,
                    state,
                )

                if entry_price is None:

                    trade_states.pop(
                        chat_id,
                        None,
                    )

                    if state["stop_requested"]:
                        break

                    continue

                # ====================================================
                # MONITOR EXISTING TRADE
                # ====================================================

                await monitor_trade(
                    bot,
                    chat_id,
                    state,
                )

                trade_states.pop(
                    chat_id,
                    None,
                )

                if state["stop_requested"]:
                    break

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "🔄 انتهت متابعة الصفقة.\n"
                        "يمكن بدء تحليل جديد من الزر."
                    ),
                )

                break

            except Exception as e:

                print(
                    "Analysis loop error:",
                    repr(e),
                )

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚠️ حدث خطأ أثناء التحليل.\n"
                        "سيتم المحاولة مرة أخرى."
                    ),
                )

                await asyncio.sleep(30)

    finally:

        analysis_tasks.pop(
            chat_id,
            None,
        )


# ============================================================
# START COMMAND
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "🟢 RustyGold جاهز",
        reply_markup=main_keyboard,
    )


# ============================================================
# START ANALYSIS
# ============================================================

async def start_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = update.effective_chat.id

    if chat_id in analysis_tasks:

        await update.message.reply_text(
            "⚠️ يوجد تحليل يعمل بالفعل."
        )

        return

    if chat_id in trade_states:

        await update.message.reply_text(
            "⚠️ توجد صفقة تحت المتابعة حاليًا."
        )

        return

    state = {
        "stop_requested": False,
    }

    analysis_tasks[chat_id] = state

    await update.message.reply_text(
        "🚀 بدأ التحليل...\n"
        "سأبحث فقط عن الصفقات التي تحقق R:R مناسب.",
        reply_markup=stop_keyboard(),
    )

    task = asyncio.create_task(
        analysis_loop(
            context.bot,
            chat_id,
        )
    )

    state["task"] = task


# ============================================================
# STOP ANALYSIS
# ============================================================

async def stop_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = update.effective_chat.id

    task_state = analysis_tasks.get(
        chat_id
    )

    trade_state = trade_states.get(
        chat_id
    )

    if task_state:
        task_state["stop_requested"] = True

        task = task_state.get("task")

        if task:
            task.cancel()

        analysis_tasks.pop(
            chat_id,
            None,
        )

    if trade_state:
        trade_state["stop_requested"] = True

    await update.message.reply_text(
        "🛑 تم إيقاف التحليل والمتابعة.",
        reply_markup=main_keyboard,
    )


# ============================================================
# CALLBACKS
# ============================================================

async def callback_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    await query.answer()

    chat_id = query.message.chat_id

    data = query.data

    # ========================================================
    # STOP
    # ========================================================

    if data == "stop_analysis":

        task_state = analysis_tasks.get(
            chat_id
        )

        trade_state = trade_states.get(
            chat_id
        )

        if task_state:
            task_state["stop_requested"] = True

            task = task_state.get("task")

            if task:
                task.cancel()

            analysis_tasks.pop(
                chat_id,
                None,
            )

        if trade_state:
            trade_state["stop_requested"] = True

        try:
            await query.edit_message_reply_markup(
                reply_markup=None
            )
        except Exception:
            pass

        await context.bot.send_message(
            chat_id=chat_id,
            text="🛑 تم إيقاف التحليل والمتابعة.",
            reply_markup=main_keyboard,
        )

        return

    # ========================================================
    # REJECT MANAGEMENT MODIFICATION
    # ========================================================

    if data.startswith("reject_mod:"):

        proposal_id = data.split(
            "reject_mod:",
            1,
        )[1]

        proposal = management_proposals.get(
            proposal_id
        )

        if not proposal:

            await query.answer(
                "انتهت صلاحية التعديل.",
                show_alert=True,
            )

            return

        if not proposal["active"]:

            await query.answer(
                "انتهت مهلة التعديل.",
                show_alert=True,
            )

            return

        proposal["active"] = False

        state = proposal["state"]

        try:
            await query.edit_message_reply_markup(
                reply_markup=None
            )
        except Exception:
            pass

        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                "❌ تم رفض التعديل.\n\n"
                f"🛑 SL يبقى: "
                f"{state['current_sl']:.2f}\n"
                f"🎯 TP يبقى: "
                f"{state['current_tp']:.2f}"
            ),
            reply_markup=stop_keyboard(),
        )


# ============================================================
# SUMMARY
# ============================================================

async def summary_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    total_closed = (
        stats["wins"]
        + stats["losses"]
    )

    if total_closed > 0:
        win_rate = (
            stats["wins"]
            / total_closed
            * 100
        )
    else:
        win_rate = 0

    text = (
        "📊 RustyGold Summary\n\n"
        f"🔎 التحليلات: {stats['ideas']}\n"
        f"✅ الصفقات المقبولة: "
        f"{stats['accepted_ideas']}\n"
        f"🚫 المرفوضة بسبب R:R: "
        f"{stats['rejected_rr']}\n"
        f"🟢 صفقات دخلت المتابعة: "
        f"{stats['entered']}\n"
        f"🏆 Wins: {stats['wins']}\n"
        f"❌ Losses: {stats['losses']}\n"
        f"📈 Win Rate: {win_rate:.1f}%"
    )

    await update.message.reply_text(
        text,
        reply_markup=main_keyboard,
    )


# ============================================================
# TEXT HANDLER
# ============================================================

async def text_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    text = update.message.text

    if text == "🚀 حلل يا جيمني":
        await start_analysis(
            update,
            context,
        )

    elif text == "🛑 إيقاف":
        await stop_analysis(
            update,
            context,
        )

    elif text == "📊 ملخص":
        await summary_command(
            update,
            context,
        )


# ============================================================
# ERROR HANDLER
# ============================================================

async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
):
    print(
        "Telegram error:",
        repr(context.error),
    )


# ============================================================
# MAIN
# ============================================================

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

    application.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT
            & ~filters.COMMAND,
            text_handler,
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            callback_handler,
        )
    )

    application.add_error_handler(
        error_handler
    )

    print(
        "RustyGold bot started."
    )

    print(
        "Gemini model:",
        gemini_model,
    )

    print(
        "Minimum RR:",
        MIN_RR,
    )

    application.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()