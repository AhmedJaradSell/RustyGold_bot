import os
import io
import re
import asyncio
import threading
import time

import requests
import pandas as pd
import mplfinance as mpf

from flask import Flask
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

from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is missing")

if not TWELVEDATA_KEY:
    raise RuntimeError("TWELVEDATA_KEY is missing")

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_KEY is missing")


# ============================================================
# GEMINI
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_KEY)


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "RustyGold is running"


@app.route("/health")
def health():
    return "OK"


def run_flask():
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", "10000"))
    )


# ============================================================
# GLOBAL STATE
# ============================================================

analysis_tasks = {}

stats = {
    "trades": 0,
    "wins": 0,
    "losses": 0,
    "pnl": 0.0,
}


# ============================================================
# KEYBOARDS
# ============================================================

def main_keyboard():
    keyboard = [
        [
            InlineKeyboardButton(
                "🚀 حلل يا جيمني",
                callback_data="start_analysis"
            )
        ],
        [
            InlineKeyboardButton(
                "🛑 إيقاف",
                callback_data="stop_analysis"
            ),
            InlineKeyboardButton(
                "📊 ملخص",
                callback_data="summary"
            ),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def stop_keyboard():
    keyboard = [
        [
            InlineKeyboardButton(
                "🛑 إيقاف التحليل",
                callback_data="stop_analysis"
            )
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


# ============================================================
# AGGRESSIVE BALANCED PROMPT
# ============================================================

ANALYSIS_PROMPT = """
You are an XAU/USD aggressive-balanced scalping analyst using ICT and Price Action.

Your objective is to detect MORE legitimate short-term trading opportunities while still avoiding random or low-quality entries.

Do not force trades.
Do not require perfect alignment between all timeframes.
Do not require every ICT concept.
Do not wait for a textbook-perfect setup when the market provides a clear and tradable structure.

TIMEFRAME HIERARCHY:

1H = CONTEXT
5M = MAIN STRUCTURE
1M = ENTRY AND EXECUTION

1H — MARKET CONTEXT
Analyze:
- Overall direction
- Major swing highs/lows
- Weekly high/low
- Major liquidity
- Trending vs ranging conditions
- Important support/resistance

The 1H timeframe is NOT a hard filter.
Do NOT reject a trade simply because the 1H direction disagrees with the 5M/1M setup.
A strong lower-timeframe reversal is allowed if there is clear evidence.

5M — MAIN STRUCTURE
Use 5M as the primary intraday structure.
Look for:
- BOS
- CHoCH
- Higher highs / higher lows
- Lower highs / lower lows
- Previous highs/lows
- Equal highs/lows
- Buy-side liquidity
- Sell-side liquidity
- Liquidity sweeps
- Displacement
- Important reaction zones
The 5M structure should normally support the trade.
However, a 5M reversal can also be traded when liquidity is swept and 1M confirms the reversal.

1M — SCALPING ENGINE
Use 1M to identify actual entry opportunities.
Look for:
- Liquidity sweep
- CHoCH
- BOS
- Displacement
- FVG
- Order Block
- Break and retest
- Rejection
- Momentum shift
- Short-term structure change
Do not require all of these.
A combination of 2–3 strong pieces of evidence can be enough.

CONTINUATION SETUPS
Prefer continuation when:
- 5M structure is clear
- Price pulls back toward a meaningful area
- 1M confirms continuation
A continuation trade does NOT require a perfect 1H alignment.

REVERSAL SETUPS
Reversals are allowed.
A reversal becomes interesting when:
- Price reaches an important high/low or liquidity pool
- Liquidity is swept
- Price strongly rejects the area
- 1M produces CHoCH/BOS
- Displacement confirms the change
A strong 1M reversal after a meaningful liquidity sweep can be traded even when 1H is still pointing in the opposite direction.

AGGRESSIVE OPPORTUNITY RULE
Do not wait for every ICT confirmation.
If the market gives a clear setup with approximately 2–3 coherent confirmations, it may qualify as a TRADE.
Examples:
1. Liquidity sweep + CHoCH + displacement
2. Strong BOS + retest
3. Key level + rejection + 1M structure shift
4. 5M liquidity sweep + 1M reversal confirmation
5. 5M trend + 1M pullback + continuation BOS
The absence of one element such as FVG or Order Block does NOT invalidate the setup.

WHEN TO SAY NO TRADE
Return NO TRADE when:
- Market structure is genuinely unclear
- Price is extremely choppy
- There is no logical entry location
- Entry would be based only on guessing
- SL cannot be placed at a meaningful invalidation point
- TP1 has no logical target
- Price has already moved too far and chasing would be required
Do NOT say NO TRADE merely because:
- 1H and 5M disagree
- FVG is absent
- Order Block is absent
- the setup is not textbook-perfect

ENTRY
ENTRY must be based on an actual price level visible in the supplied data.
Possible entries:
- FVG retest
- Order Block retest
- Broken structure retest
- Liquidity reaction
- Support/resistance reaction
- Current price after confirmation
Do not invent arbitrary prices.
If the setup is already confirmed, an entry near the current market price is acceptable when justified.

STOP LOSS
Place SL beyond the structural invalidation point.
BUY:
SL < ENTRY
SELL:
SL > ENTRY
Do not place SL randomly.
Prefer the nearest logical invalidation point that gives the setup enough room to breathe.

TAKE PROFIT
TP1 should target the nearest meaningful liquidity or logical price objective.
BUY:
TP1 > ENTRY
SELL:
TP1 < ENTRY
Prefer realistic scalp targets.
Do not demand a very large move when the nearest liquidity target is closer.
Do not choose a TP simply to make the reward/risk ratio look good.

TRADE MANAGEMENT LOGIC
Favor setups where:
- Entry is close to the invalidation level
- The target is realistically reachable
- Price is not already exhausted
- The setup has room to move
Avoid chasing after a large impulsive candle.
If price has already made most of the expected move, prefer NO TRADE.

MARKET CONDITIONS
TRENDING MARKET:
Look for continuation and pullback entries.
RANGING MARKET:
Look for liquidity sweeps and reversals near range extremes.
CHOPPY MARKET:
Reduce confidence and trade only if structure becomes clear.
HIGH MOMENTUM:
Do not blindly chase.
Wait for a pullback, retest, rejection, or structure confirmation when possible.

DECISION
Return exactly:
TRADE
or
NO TRADE
If TRADE:
DIRECTION: BUY or SELL
ENTRY: <price>
SL: <price>
TP1: <price>
Then briefly explain:
1. 1H context
2. 5M structure
3. Liquidity event
4. 1M confirmation
5. Entry reason
6. SL reason
7. TP1 reason
If NO TRADE:
NO TRADE
Reason: <brief explanation>

FINAL OBJECTIVE
You are an AGGRESSIVE-BALANCED scalping analyst.
Look actively for opportunities.
Do not be excessively conservative.
Do not require perfect ICT setups.
Allow both continuation and reversal trades.
Give priority to:
LIQUIDITY → STRUCTURE → CONFIRMATION → ENTRY
But never manufacture a trade when the evidence is unclear.
The goal is to capture legitimate XAU/USD intraday moves, not to maximize the number of trades.
"""


# ============================================================
# TRANSLATION PROMPT
# ============================================================

TRANSLATION_PROMPT = """
Translate the following trading analysis from English into clear natural Arabic.

IMPORTANT:
- Do NOT change any numbers.
- Do NOT change prices.
- Do NOT translate these exact technical decision words:
  TRADE
  NO TRADE
  BUY
  SELL
  DIRECTION
  ENTRY
  SL
  TP1
- Keep all price values exactly as written.
- Translate only the explanatory text.
- Do not add new analysis.
- Do not change the trading decision.
- Do not invent information.

Return only the Arabic translation.

Original analysis:
"""


# ============================================================
# GEMINI MODELS
# ============================================================

def get_gemini_models():
    models = []

    try:
        for model in gemini_client.models.list():
            name = getattr(model, "name", "")

            if not name:
                continue

            if "gemini" not in name.lower():
                continue

            models.append(name)

    except Exception as e:
        print("Gemini model list error:", e)

    def model_priority(name):
        low = name.lower()

        if "flash" in low:
            return 0

        if "pro" in low:
            return 1

        return 2

    models.sort(key=model_priority)

    return models


# ============================================================
# TWELVE DATA
# ============================================================

def get_twelvedata(interval, outputsize):
    url = "https://api.twelvedata.com/time_series"

    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_KEY,
        "format": "JSON",
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
        "close",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = df.sort_values("datetime")

    df = df.set_index("datetime")

    return df


# ============================================================
# GOLD API
# ============================================================

def get_gold_price():
    url = "https://api.gold-api.com/price/XAU"

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    data = response.json()

    price = data.get("price")

    if price is None:
        raise RuntimeError(
            f"Gold API error: {data}"
        )

    return float(price)


# ============================================================
# OHLC FORMAT
# ============================================================

def format_ohlc(df):
    rows = []

    for index, row in df.iterrows():
        rows.append(
            f"{index} | "
            f"O={row['open']:.2f} "
            f"H={row['high']:.2f} "
            f"L={row['low']:.2f} "
            f"C={row['close']:.2f}"
        )

    return "\n".join(rows)


# ============================================================
# CHART
# ============================================================

def create_chart(df, title):
    chart_df = df.copy()

    chart_df.index = pd.to_datetime(
        chart_df.index
    )

    chart_df = chart_df[
        [
            "open",
            "high",
            "low",
            "close",
        ]
    ]

    chart_df.columns = [
        "Open",
        "High",
        "Low",
        "Close",
    ]

    buffer = io.BytesIO()

    mpf.plot(
        chart_df,
        type="candle",
        style="charles",
        title=title,
        figsize=(12, 7),
        volume=False,
        savefig=dict(
            fname=buffer,
            dpi=120,
            bbox_inches="tight",
        ),
    )

    buffer.seek(0)

    return buffer


# ============================================================
# GEMINI ANALYSIS
# ============================================================

async def call_gemini_analysis(
    df_1h,
    df_5m,
    df_1m,
    current_price,
):
    chart_1h = create_chart(
        df_1h,
        "XAU/USD 1H"
    )

    chart_5m = create_chart(
        df_5m,
        "XAU/USD 5M"
    )

    chart_1m = create_chart(
        df_1m.tail(240),
        "XAU/USD 1M"
    )

    contents = [
        types.Part.from_text(
            text=ANALYSIS_PROMPT
        ),

        types.Part.from_text(
            text="\nCURRENT LIVE GOLD PRICE:\n"
            f"{current_price:.2f}"
        ),

        types.Part.from_text(
            text="\n1H OHLC:\n"
            + format_ohlc(df_1h)
        ),

        types.Part.from_text(
            text="\n5M OHLC:\n"
            + format_ohlc(df_5m)
        ),

        types.Part.from_text(
            text="\nLATEST 1M OHLC:\n"
            + format_ohlc(df_1m.tail(180))
        ),

        types.Part.from_bytes(
            data=chart_1h.getvalue(),
            mime_type="image/png",
        ),

        types.Part.from_bytes(
            data=chart_5m.getvalue(),
            mime_type="image/png",
        ),

        types.Part.from_bytes(
            data=chart_1m.getvalue(),
            mime_type="image/png",
        ),
    ]

    models = get_gemini_models()

    if not models:
        raise RuntimeError(
            "No Gemini models available"
        )

    last_error = None

    for model_name in models:
        try:
            print(
                "Trying Gemini model:",
                model_name
            )

            response = gemini_client.models.generate_content(
                model=model_name,
                contents=contents,
            )

            text = getattr(
                response,
                "text",
                None
            )

            if text:
                print(
                    "Gemini success:",
                    model_name
                )

                return text.strip()

        except Exception as e:
            print(
                "Gemini model failed:",
                model_name,
                e
            )

            last_error = e

    raise RuntimeError(
        f"All Gemini models failed: {last_error}"
    )


# ============================================================
# TRANSLATION
# ============================================================

async def translate_analysis(text):
    models = get_gemini_models()

    if not models:
        return text

    prompt = (
        TRANSLATION_PROMPT
        + "\n"
        + text
    )

    for model_name in models:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_text(
                        text=prompt
                    )
                ],
            )

            translated = getattr(
                response,
                "text",
                None
            )

            if translated:
                return translated.strip()

        except Exception as e:
            print(
                "Translation failed:",
                model_name,
                e
            )

    return text


# ============================================================
# PARSE TRADE
# ============================================================

def parse_trade(text):
    if not text:
        return None

    upper = text.upper()

    if "NO TRADE" in upper:
        return None

    if "TRADE" not in upper:
        return None

    direction_match = re.search(
        r"DIRECTION\s*:\s*(BUY|SELL)",
        upper
    )

    entry_match = re.search(
        r"ENTRY\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        upper
    )

    sl_match = re.search(
        r"SL\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        upper
    )

    tp_match = re.search(
        r"TP1\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        upper
    )

    if not all(
        [
            direction_match,
            entry_match,
            sl_match,
            tp_match,
        ]
    ):
        return None

    direction = direction_match.group(1)

    entry = float(
        entry_match.group(1)
    )

    sl = float(
        sl_match.group(1)
    )

    tp1 = float(
        tp_match.group(1)
    )

    if direction == "BUY":
        if not (
            sl < entry < tp1
        ):
            return None

    elif direction == "SELL":
        if not (
            tp1 < entry < sl
        ):
            return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "analysis": text,
    }


# ============================================================
# LONG MESSAGE
# ============================================================

async def send_long_message(
    bot,
    chat_id,
    text,
    reply_markup=None,
):
    max_length = 4000

    if reply_markup is None:
        reply_markup = stop_keyboard()

    if len(text) <= max_length:
        await bot.send_message(
            chat_id=chat_id,
            text=text,
            reply_markup=reply_markup,
        )
        return

    parts = []

    while text:
        parts.append(
            text[:max_length]
        )
        text = text[max_length:]

    for index, part in enumerate(parts):
        await bot.send_message(
            chat_id=chat_id,
            text=part,
            reply_markup=(
                reply_markup
                if index == len(parts) - 1
                else None
            ),
        )


# ============================================================
# WAIT FOR ENTRY
# ============================================================

async def wait_for_entry(
    bot,
    chat_id,
    trade,
    task_state,
):
    start = time.monotonic()

    entry = trade["entry"]

    while True:

        if task_state.get("cancelled"):
            return False

        elapsed = (
            time.monotonic()
            - start
        )

        if elapsed >= 300:
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⏰ انتهت مهلة انتظار الدخول.\n\n"
                    f"DIRECTION: {trade['direction']}\n"
                    f"ENTRY: {trade['entry']:.2f}\n"
                    f"SL: {trade['sl']:.2f}\n"
                    f"TP1: {trade['tp1']:.2f}\n\n"
                    "لم يصل السعر إلى منطقة ENTRY خلال 5 دقائق."
                ),
                reply_markup=stop_keyboard(),
            )

            return False

        try:
            price = get_gold_price()

            distance = abs(
                price - entry
            )

            if distance <= 0.60:

                task_state["entered"] = True
                task_state["entry_price"] = price
                task_state["trade_start"] = time.monotonic()

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "🟢 تم دخول الصفقة\n\n"
                        f"DIRECTION: {trade['direction']}\n"
                        f"ENTRY: {trade['entry']:.2f}\n"
                        f"سعر الدخول الفعلي: {price:.2f}\n"
                        f"SL: {trade['sl']:.2f}\n"
                        f"TP1: {trade['tp1']:.2f}\n\n"
                        "👀 بدأت مراقبة الصفقة كل 5 ثواني.\n"
                        "📩 سيتم إرسال تحديث كل 5 دقائق."
                    ),
                    reply_markup=stop_keyboard(),
                )

                return True

        except Exception as e:
            print(
                "Entry price error:",
                e
            )

        await asyncio.sleep(5)


# ============================================================
# TRADE STATUS
# ============================================================

def build_trade_status(
    trade,
    current_price,
    minutes_open,
):
    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]

    if direction == "BUY":
        pnl = current_price - entry
    else:
        pnl = entry - current_price

    if pnl > 0:
        pnl_text = f"+{pnl:.2f}"
    else:
        pnl_text = f"{pnl:.2f}"

    return (
        "👀 تحديث الصفقة\n\n"
        f"بعد {minutes_open} دقائق\n\n"
        f"DIRECTION: {direction}\n"
        f"ENTRY: {entry:.2f}\n"
        f"السعر الحالي: {current_price:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP1: {tp1:.2f}\n\n"
        "الحالة: الصفقة ما زالت مفتوحة\n"
        f"النتيجة الحالية: {pnl_text} من نقطة الدخول."
    )


# ============================================================
# MONITOR TRADE
# ============================================================

async def monitor_trade(
    bot,
    chat_id,
    trade,
    task_state,
):
    direction = trade["direction"]

    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]

    # IMPORTANT:
    # This is the independent user-report timer.
    next_report_at = (
        time.monotonic()
        + 300
    )

    trade_start = time.monotonic()

    task_state["trade_open"] = True

    while True:

        # ----------------------------------------------------
        # STOP
        # ----------------------------------------------------

        if task_state.get("cancelled"):
            return


        # ----------------------------------------------------
        # PRICE CHECK EVERY 5 SECONDS
        # ----------------------------------------------------

        try:
            price = get_gold_price()

        except Exception as e:
            print(
                "Trade monitor price error:",
                e
            )

            await asyncio.sleep(5)
            continue


        # ----------------------------------------------------
        # SL / TP CHECK FIRST
        # ----------------------------------------------------

        hit_result = None

        if direction == "BUY":

            if price >= tp1:
                hit_result = "TP1"

            elif price <= sl:
                hit_result = "SL"

        else:

            if price <= tp1:
                hit_result = "TP1"

            elif price >= sl:
                hit_result = "SL"


        # ----------------------------------------------------
        # IMMEDIATE RESULT
        # ----------------------------------------------------

        if hit_result:

            if hit_result == "TP1":

                if direction == "BUY":
                    pnl = price - entry
                else:
                    pnl = entry - price

                stats["wins"] += 1
                stats["pnl"] += pnl

                result_text = (
                    "🎯 TP1 HIT"
                )

            else:

                if direction == "BUY":
                    pnl = price - entry
                else:
                    pnl = entry - price

                stats["losses"] += 1
                stats["pnl"] += pnl

                result_text = (
                    "🛑 SL HIT"
                )


            elapsed_minutes = int(
                (
                    time.monotonic()
                    - trade_start
                ) / 60
            )

            if pnl >= 0:
                pnl_text = f"+{pnl:.2f}"
            else:
                pnl_text = f"{pnl:.2f}"


            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"{result_text}\n\n"
                    f"DIRECTION: {direction}\n"
                    f"ENTRY: {entry:.2f}\n"
                    f"السعر النهائي: {price:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}\n\n"
                    f"مدة الصفقة: {elapsed_minutes} دقائق\n"
                    f"Simulated PnL: {pnl_text}\n\n"
                    "📊 انتهت الصفقة."
                ),
                reply_markup=stop_keyboard(),
            )

            task_state["trade_open"] = False
            return


        # ----------------------------------------------------
        # FIVE MINUTE REPORT
        # ----------------------------------------------------

        now = time.monotonic()

        if now >= next_report_at:

            elapsed_minutes = int(
                (
                    now
                    - trade_start
                ) / 60
            )

            if elapsed_minutes < 5:
                elapsed_minutes = 5

            status_text = build_trade_status(
                trade=trade,
                current_price=price,
                minutes_open=elapsed_minutes,
            )

            await bot.send_message(
                chat_id=chat_id,
                text=status_text,
                reply_markup=stop_keyboard(),
            )

            # Keep the reporting schedule independent.
            # If sending was delayed, do not restart the
            # five-minute timer from zero.
            while next_report_at <= now:
                next_report_at += 300


        # ----------------------------------------------------
        # NEXT INTERNAL CHECK
        # ----------------------------------------------------

        await asyncio.sleep(5)


# ============================================================
# ANALYSIS LOOP
# ============================================================

async def analysis_loop(
    bot,
    chat_id,
):
    task_state = {
        "cancelled": False,
        "entered": False,
        "trade_open": False,
    }

    analysis_tasks[chat_id] = task_state

    try:

        while not task_state["cancelled"]:

            try:

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "🔎 جاري تحليل XAU/USD...\n"
                        "1H + 5M + 1M"
                    ),
                    reply_markup=stop_keyboard(),
                )

                # ------------------------------------------------
                # DATA
                # ------------------------------------------------

                df_1h = get_twelvedata(
                    "1h",
                    168
                )

                df_5m = get_twelvedata(
                    "5min",
                    288
                )

                df_1m_full = get_twelvedata(
                    "1min",
                    720
                )

                df_1m = df_1m_full.tail(
                    240
                )

                current_price = get_gold_price()


                # ------------------------------------------------
                # GEMINI
                # ------------------------------------------------

                analysis = await call_gemini_analysis(
                    df_1h=df_1h,
                    df_5m=df_5m,
                    df_1m=df_1m,
                    current_price=current_price,
                )


                # ------------------------------------------------
                # TRANSLATION
                # ------------------------------------------------

                arabic_analysis = await translate_analysis(
                    analysis
                )


                # ------------------------------------------------
                # SEND ANALYSIS
                # ------------------------------------------------

                await send_long_message(
                    bot,
                    chat_id,
                    arabic_analysis,
                    reply_markup=stop_keyboard(),
                )


                # ------------------------------------------------
                # PARSE
                # ------------------------------------------------

                trade = parse_trade(
                    analysis
                )


                # ------------------------------------------------
                # NO TRADE
                # ------------------------------------------------

                if trade is None:

                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "⏳ NO TRADE\n\n"
                            "سيتم إعادة التحليل بعد 5 دقائق."
                        ),
                        reply_markup=stop_keyboard(),
                    )

                    await asyncio.sleep(300)

                    continue


                # ------------------------------------------------
                # TRADE FOUND
                # ------------------------------------------------

                stats["trades"] += 1

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "🚨 TRADE FOUND\n\n"
                        f"DIRECTION: {trade['direction']}\n"
                        f"ENTRY: {trade['entry']:.2f}\n"
                        f"SL: {trade['sl']:.2f}\n"
                        f"TP1: {trade['tp1']:.2f}\n\n"
                        "👀 في انتظار وصول السعر إلى ENTRY."
                    ),
                    reply_markup=stop_keyboard(),
                )


                # ------------------------------------------------
                # WAIT FOR ENTRY
                # ------------------------------------------------

                entered = await wait_for_entry(
                    bot,
                    chat_id,
                    trade,
                    task_state,
                )

                if not entered:

                    if task_state.get("cancelled"):
                        return

                    await asyncio.sleep(10)
                    continue


                # ------------------------------------------------
                # MONITOR OPEN TRADE
                # ------------------------------------------------

                await monitor_trade(
                    bot,
                    chat_id,
                    trade,
                    task_state,
                )


                if task_state.get("cancelled"):
                    return


                # ------------------------------------------------
                # AFTER TRADE
                # ------------------------------------------------

                await asyncio.sleep(10)


            except asyncio.CancelledError:
                return

            except Exception as e:

                print(
                    "Analysis loop error:",
                    e
                )

                if task_state.get("cancelled"):
                    return

                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "⚠️ حدث خطأ أثناء التحليل.\n\n"
                            f"{str(e)}\n\n"
                            "سيتم إعادة المحاولة بعد دقيقة."
                        ),
                        reply_markup=stop_keyboard(),
                    )

                except Exception as send_error:
                    print(
                        "Error sending error message:",
                        send_error
                    )

                await asyncio.sleep(60)


    finally:

        if analysis_tasks.get(chat_id) is task_state:
            del analysis_tasks[chat_id]


# ============================================================
# START COMMAND
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "🤖 RustyGold جاهز",
        reply_markup=main_keyboard(),
    )


# ============================================================
# START ANALYSIS
# ============================================================

async def start_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    await query.answer()

    chat_id = query.message.chat_id

    existing = analysis_tasks.get(
        chat_id
    )

    if existing:
        if not existing.get("cancelled"):
            await query.message.reply_text(
                "⚠️ التحليل يعمل بالفعل.",
                reply_markup=stop_keyboard(),
            )
            return

    task = asyncio.create_task(
        analysis_loop(
            context.bot,
            chat_id,
        )
    )

    # The actual state is registered by analysis_loop.
    # This task is intentionally not awaited here.


# ============================================================
# STOP ANALYSIS
# ============================================================

async def stop_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    await query.answer()

    chat_id = query.message.chat_id

    task_state = analysis_tasks.get(
        chat_id
    )

    if task_state:

        task_state["cancelled"] = True

        analysis_tasks.pop(
            chat_id,
            None
        )

        await query.message.reply_text(
            "🛑 تم إيقاف التحليل.",
            reply_markup=main_keyboard(),
        )

    else:

        await query.message.reply_text(
            "ℹ️ لا يوجد تحليل يعمل حاليًا.",
            reply_markup=main_keyboard(),
        )


# ============================================================
# SUMMARY
# ============================================================

async def summary(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    await query.answer()

    total = stats["trades"]
    wins = stats["wins"]
    losses = stats["losses"]

    if total > 0:
        win_rate = (
            wins / total
        ) * 100
    else:
        win_rate = 0.0

    pnl = stats["pnl"]

    if pnl >= 0:
        pnl_text = f"+{pnl:.2f}"
    else:
        pnl_text = f"{pnl:.2f}"

    await query.message.reply_text(
        (
            "📊 ملخص RustyGold\n\n"
            f"Trades: {total}\n"
            f"Wins: {wins}\n"
            f"Losses: {losses}\n"
            f"Win rate: {win_rate:.2f}%\n"
            f"Simulated PnL: {pnl_text}"
        ),
        reply_markup=main_keyboard(),
    )


# ============================================================
# BUTTON HANDLER
# ============================================================

async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    data = query.data

    if data == "start_analysis":
        await start_analysis(
            update,
            context,
        )

    elif data == "stop_analysis":
        await stop_analysis(
            update,
            context,
        )

    elif data == "summary":
        await summary(
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
        context.error
    )


# ============================================================
# MAIN
# ============================================================

def main():
    threading.Thread(
        target=run_flask,
        daemon=True,
    ).start()

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
        CallbackQueryHandler(
            button_handler
        )
    )

    application.add_error_handler(
        error_handler
    )

    print(
        "RustyGold bot started"
    )

    application.run_polling()


if __name__ == "__main__":
    main()