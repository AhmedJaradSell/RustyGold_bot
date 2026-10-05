import os
import io
import re
import asyncio
import threading
from datetime import datetime

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


# =========================================================
# ENVIRONMENT
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN is missing")

if not TWELVEDATA_KEY:
    raise RuntimeError("TWELVEDATA_KEY is missing")

if not GEMINI_KEY:
    raise RuntimeError("GEMINI_KEY is missing")


# =========================================================
# GEMINI
# =========================================================

gemini_client = genai.Client(api_key=GEMINI_KEY)


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "RustyGold is running."


@app.route("/health")
def health():
    return "OK"


def run_flask():
    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 10000)),
    )


# =========================================================
# GLOBAL STATE
# =========================================================

analysis_tasks = {}

stats = {
    "trades": 0,
    "wins": 0,
    "losses": 0,
    "pnl": 0.0,
}


# =========================================================
# KEYBOARDS
# =========================================================

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


# =========================================================
# GEMINI PROMPT
# =========================================================

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


# =========================================================
# TRANSLATION PROMPT
# =========================================================

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


# =========================================================
# GEMINI MODELS
# =========================================================

def get_gemini_models():
    models = []

    try:
        for model in gemini_client.models.list():
            name = getattr(model, "name", "")

            if not name:
                continue

            name_lower = name.lower()

            if "gemini" not in name_lower:
                continue

            models.append(name)

    except Exception as e:
        print("Gemini model list error:", e)

    # Prefer Flash models
    flash_models = [
        x for x in models
        if "flash" in x.lower()
    ]

    other_models = [
        x for x in models
        if x not in flash_models
    ]

    return flash_models + other_models


# =========================================================
# TWELVEDATA
# =========================================================

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
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"TwelveData error: {data}"
        )

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(df["datetime"])

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.dropna()

    df = df.sort_values("datetime")

    df = df.set_index("datetime")

    return df


# =========================================================
# LIVE GOLD PRICE
# =========================================================

def get_gold_price():
    url = "https://api.gold-api.com/price/XAU"

    response = requests.get(
        url,
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    price = data.get("price")

    if price is None:
        raise RuntimeError(
            f"Gold API error: {data}"
        )

    return float(price)


# =========================================================
# OHLC FORMAT
# =========================================================

def format_ohlc(df):
    lines = []

    for index, row in df.iterrows():
        lines.append(
            f"{index.strftime('%Y-%m-%d %H:%M')} | "
            f"O={row['open']:.2f} "
            f"H={row['high']:.2f} "
            f"L={row['low']:.2f} "
            f"C={row['close']:.2f}"
        )

    return "\n".join(lines)


# =========================================================
# CHART
# =========================================================

def create_chart(df, title):
    chart_df = df.copy()

    chart_df = chart_df[
        ["open", "high", "low", "close"]
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
        volume=False,
        figsize=(12, 7),
        savefig=dict(
            fname=buffer,
            dpi=120,
            bbox_inches="tight",
        ),
    )

    buffer.seek(0)

    return buffer


# =========================================================
# GEMINI ANALYSIS
# =========================================================

async def call_gemini_analysis(
    one_hour,
    five_min,
    one_min,
    current_price,
    chart_1h,
    chart_5m,
    chart_1m,
):
    ohlc_1h = format_ohlc(one_hour)

    ohlc_5m = format_ohlc(five_min)

    latest_1m = one_min.tail(180)

    ohlc_1m = format_ohlc(latest_1m)

    full_prompt = f"""
{ANALYSIS_PROMPT}

CURRENT LIVE GOLD PRICE:
{current_price:.2f}

1H OHLC DATA:
{ohlc_1h}

5M OHLC DATA:
{ohlc_5m}

LATEST 1M OHLC DATA:
{ohlc_1m}

Analyze the supplied charts and OHLC data.
"""

    parts = [
        types.Part.from_text(text=full_prompt),
    ]

    for chart in [
        chart_1h,
        chart_5m,
        chart_1m,
    ]:
        image_bytes = chart.getvalue()

        parts.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/png",
            )
        )

    contents = [
        types.Content(
            role="user",
            parts=parts,
        )
    ]

    models = get_gemini_models()

    if not models:
        raise RuntimeError(
            "No Gemini models available."
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
                None,
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
                e,
            )

            last_error = e

    raise RuntimeError(
        f"All Gemini models failed: {last_error}"
    )


# =========================================================
# TRANSLATE ANALYSIS
# =========================================================

async def translate_analysis(text):
    prompt = (
        TRANSLATION_PROMPT
        + "\n"
        + text
    )

    models = get_gemini_models()

    last_error = None

    for model_name in models:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            result = getattr(
                response,
                "text",
                None,
            )

            if result:
                return result.strip()

        except Exception as e:
            print(
                "Translation model failed:",
                model_name,
                e,
            )

            last_error = e

    print(
        "Translation failed:",
        last_error
    )

    return text


# =========================================================
# PARSE TRADE
# =========================================================

def parse_trade(text):
    if not text:
        return None

    cleaned = text.strip()

    first_line = cleaned.splitlines()[0].strip().upper()

    if first_line.startswith("NO TRADE"):
        return None

    if not first_line.startswith("TRADE"):
        match = re.search(
            r"\bTRADE\b",
            cleaned,
            re.IGNORECASE,
        )

        no_trade_match = re.search(
            r"\bNO TRADE\b",
            cleaned,
            re.IGNORECASE,
        )

        if no_trade_match and (
            not match
            or no_trade_match.start() < match.start()
        ):
            return None

        if not match:
            return None

    direction_match = re.search(
        r"DIRECTION\s*:\s*(BUY|SELL)",
        cleaned,
        re.IGNORECASE,
    )

    entry_match = re.search(
        r"ENTRY\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        cleaned,
        re.IGNORECASE,
    )

    sl_match = re.search(
        r"SL\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        cleaned,
        re.IGNORECASE,
    )

    tp_match = re.search(
        r"TP1\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        cleaned,
        re.IGNORECASE,
    )

    if not all([
        direction_match,
        entry_match,
        sl_match,
        tp_match,
    ]):
        return None

    direction = direction_match.group(1).upper()

    entry = float(entry_match.group(1))

    sl = float(sl_match.group(1))

    tp1 = float(tp_match.group(1))

    # Validate logical direction
    if direction == "BUY":
        if not (sl < entry < tp1):
            return None

    elif direction == "SELL":
        if not (tp1 < entry < sl):
            return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
        "raw": cleaned,
    }


# =========================================================
# SEND LONG MESSAGE
# =========================================================

async def send_long_message(
    bot,
    chat_id,
    text,
    reply_markup=None,
):
    max_length = 4000

    chunks = []

    while len(text) > max_length:
        split_at = text.rfind(
            "\n",
            0,
            max_length,
        )

        if split_at <= 0:
            split_at = max_length

        chunks.append(
            text[:split_at]
        )

        text = text[split_at:]

    if text:
        chunks.append(text)

    for index, chunk in enumerate(chunks):
        if index == len(chunks) - 1:
            await bot.send_message(
                chat_id=chat_id,
                text=chunk,
                reply_markup=reply_markup,
            )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=chunk,
            )


# =========================================================
# WAIT FOR ENTRY
# =========================================================

async def wait_for_entry(
    bot,
    chat_id,
    trade,
    task,
):
    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]

    start_time = asyncio.get_running_loop().time()

    while True:
        if task.get("cancelled"):
            return None

        elapsed = (
            asyncio.get_running_loop().time()
            - start_time
        )

        if elapsed >= 300:
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⏳ لم يصل السعر إلى ENTRY خلال 5 دقائق.\n\n"
                    f"DIRECTION: {direction}\n"
                    f"ENTRY: {entry:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}\n\n"
                    "🔄 سيتم إعادة التحليل."
                ),
                reply_markup=stop_keyboard(),
            )

            return None

        try:
            price = get_gold_price()

            if abs(price - entry) <= 0.60:

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "🎯 تم الوصول إلى منطقة ENTRY\n\n"
                        f"DIRECTION: {direction}\n"
                        f"ENTRY: {entry:.2f}\n"
                        f"السعر الحالي: {price:.2f}\n"
                        f"SL: {sl:.2f}\n"
                        f"TP1: {tp1:.2f}\n\n"
                        "▶️ بدأت مراقبة الصفقة."
                    ),
                    reply_markup=stop_keyboard(),
                )

                return price

        except Exception as e:
            print(
                "Entry monitoring error:",
                e
            )

        await asyncio.sleep(5)


# =========================================================
# TRADE STATUS MESSAGE
# =========================================================

def build_trade_status(
    trade,
    current_price,
    elapsed_seconds,
):
    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]

    if direction == "BUY":
        pnl = current_price - entry
    else:
        pnl = entry - current_price

    minutes = int(elapsed_seconds // 60)

    if pnl > 0:
        pnl_text = f"+{pnl:.2f}"
    else:
        pnl_text = f"{pnl:.2f}"

    return (
        f"👀 تحديث الصفقة — بعد {minutes} دقيقة\n\n"
        f"DIRECTION: {direction}\n"
        f"ENTRY: {entry:.2f}\n"
        f"السعر الحالي: {current_price:.2f}\n"
        f"SL: {sl:.2f}\n"
        f"TP1: {tp1:.2f}\n\n"
        f"الحالة: الصفقة ما زالت مفتوحة، "
        f"{pnl_text} من نقطة الدخول.\n\n"
        "📌 المراقبة مستمرة داخليًا كل 5 ثوانٍ."
    )


# =========================================================
# MONITOR TRADE
# =========================================================

async def monitor_trade(
    bot,
    chat_id,
    trade,
    task,
):
    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]

    start_time = asyncio.get_running_loop().time()

    # User-facing report timer
    last_report_time = start_time

    # Maximum trade monitoring time
    max_monitoring_seconds = 30 * 60

    while True:

        # -------------------------------------------------
        # STOP CHECK
        # -------------------------------------------------

        if task.get("cancelled"):
            return

        # -------------------------------------------------
        # TIME
        # -------------------------------------------------

        now = asyncio.get_running_loop().time()

        elapsed = now - start_time

        # -------------------------------------------------
        # MAXIMUM TRADE TIME
        # -------------------------------------------------

        if elapsed >= max_monitoring_seconds:

            try:
                price = get_gold_price()

                if direction == "BUY":
                    pnl = price - entry
                else:
                    pnl = entry - price

                if pnl > 0:
                    pnl_text = f"+{pnl:.2f}"
                else:
                    pnl_text = f"{pnl:.2f}"

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⏰ انتهت مدة مراقبة الصفقة.\n\n"
                        f"DIRECTION: {direction}\n"
                        f"ENTRY: {entry:.2f}\n"
                        f"السعر النهائي: {price:.2f}\n"
                        f"SL: {sl:.2f}\n"
                        f"TP1: {tp1:.2f}\n\n"
                        f"الحالة: انتهت المهلة.\n"
                        f"Simulated PnL: {pnl_text}"
                    ),
                    reply_markup=stop_keyboard(),
                )

            except Exception as e:
                print(
                    "Timeout price error:",
                    e
                )

            return

        # -------------------------------------------------
        # GET PRICE
        # -------------------------------------------------

        try:
            price = get_gold_price()

        except Exception as e:
            print(
                "Trade monitoring error:",
                e
            )

            await asyncio.sleep(5)
            continue

        # -------------------------------------------------
        # CHECK SL / TP FIRST
        # -------------------------------------------------
        # Important:
        # SL/TP are checked BEFORE the 5-minute report.
        # Therefore, if SL/TP is reached, the result is
        # sent immediately.
        # -------------------------------------------------

        result = None

        if direction == "BUY":

            if price >= tp1:
                result = "TP1"

            elif price <= sl:
                result = "SL"

        else:

            if price <= tp1:
                result = "TP1"

            elif price >= sl:
                result = "SL"

        # -------------------------------------------------
        # IMMEDIATE SL / TP RESULT
        # -------------------------------------------------

        if result:

            if result == "TP1":
                stats["wins"] += 1

                pnl = abs(tp1 - entry)

                result_text = (
                    "🎯 TP1 HIT — الصفقة رابحة"
                )

            else:
                stats["losses"] += 1

                pnl = -abs(sl - entry)

                result_text = (
                    "🛑 SL HIT — الصفقة خاسرة"
                )

            stats["pnl"] += pnl

            if pnl > 0:
                pnl_text = f"+{pnl:.2f}"
            else:
                pnl_text = f"{pnl:.2f}"

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"{result_text}\n\n"
                    f"DIRECTION: {direction}\n"
                    f"ENTRY: {entry:.2f}\n"
                    f"السعر: {price:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}\n\n"
                    f"Simulated PnL: {pnl_text}\n\n"
                    "🔄 سيتم إعادة التحليل."
                ),
                reply_markup=stop_keyboard(),
            )

            return

        # -------------------------------------------------
        # 5-MINUTE USER REPORT
        # -------------------------------------------------

        if now - last_report_time >= 300:

            status_message = build_trade_status(
                trade=trade,
                current_price=price,
                elapsed_seconds=elapsed,
            )

            await bot.send_message(
                chat_id=chat_id,
                text=status_message,
                reply_markup=stop_keyboard(),
            )

            last_report_time = now

        # -------------------------------------------------
        # INTERNAL MONITORING
        # -------------------------------------------------
        # The bot continues checking every 5 seconds.
        # No message is sent during these checks unless
        # SL/TP is reached or the 5-minute report is due.
        # -------------------------------------------------

        await asyncio.sleep(5)


# =========================================================
# ANALYSIS LOOP
# =========================================================

async def analysis_loop(
    bot,
    chat_id,
    task,
):
    while not task.get("cancelled"):

        try:
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "🔎 جاري تحليل XAU/USD...\n\n"
                    "1H + 5M + 1M\n"
                    "ICT + Price Action"
                ),
            )

            # ---------------------------------------------
            # GET DATA
            # ---------------------------------------------

            one_hour = get_twelvedata(
                "1h",
                168,
            )

            five_min = get_twelvedata(
                "5min",
                288,
            )

            one_min_full = get_twelvedata(
                "1min",
                720,
            )

            one_min = one_min_full.tail(240)

            current_price = get_gold_price()

            # ---------------------------------------------
            # CREATE CHARTS
            # ---------------------------------------------

            chart_1h = create_chart(
                one_hour,
                "XAU/USD — 1H"
            )

            chart_5m = create_chart(
                five_min,
                "XAU/USD — 5M"
            )

            chart_1m = create_chart(
                one_min,
                "XAU/USD — 1M"
            )

            # ---------------------------------------------
            # GEMINI
            # ---------------------------------------------

            raw_analysis = await call_gemini_analysis(
                one_hour=one_hour,
                five_min=five_min,
                one_min=one_min_full,
                current_price=current_price,
                chart_1h=chart_1h,
                chart_5m=chart_5m,
                chart_1m=chart_1m,
            )

            print(
                "RAW GEMINI ANALYSIS:\n",
                raw_analysis
            )

            # ---------------------------------------------
            # PARSE BEFORE TRANSLATION
            # ---------------------------------------------

            trade = parse_trade(
                raw_analysis
            )

            # ---------------------------------------------
            # TRANSLATION
            # ---------------------------------------------

            arabic_analysis = await translate_analysis(
                raw_analysis
            )

            # ---------------------------------------------
            # NO TRADE
            # ---------------------------------------------

            if trade is None:

                await send_long_message(
                    bot=bot,
                    chat_id=chat_id,
                    text=(
                        "📊 نتيجة التحليل\n\n"
                        + arabic_analysis
                        + "\n\n"
                        "⏳ لا توجد صفقة حاليًا.\n"
                        "🔄 سأعيد التحليل بعد 5 دقائق."
                    ),
                    reply_markup=stop_keyboard(),
                )

                # Wait 5 minutes
                # But still allow STOP to cancel.
                for _ in range(60):

                    if task.get("cancelled"):
                        return

                    await asyncio.sleep(5)

                continue

            # ---------------------------------------------
            # TRADE FOUND
            # ---------------------------------------------

            stats["trades"] += 1

            direction = trade["direction"]
            entry = trade["entry"]
            sl = trade["sl"]
            tp1 = trade["tp1"]

            trade_message = (
                "🚨 TRADE FOUND\n\n"
                f"DIRECTION: {direction}\n"
                f"ENTRY: {entry:.2f}\n"
                f"SL: {sl:.2f}\n"
                f"TP1: {tp1:.2f}\n\n"
                "📋 التحليل:\n"
                f"{arabic_analysis}\n\n"
                "👀 الآن سأراقب ENTRY كل 5 ثوانٍ."
            )

            await send_long_message(
                bot=bot,
                chat_id=chat_id,
                text=trade_message,
                reply_markup=stop_keyboard(),
            )

            # ---------------------------------------------
            # WAIT FOR ENTRY
            # ---------------------------------------------

            entry_price = await wait_for_entry(
                bot=bot,
                chat_id=chat_id,
                trade=trade,
                task=task,
            )

            if task.get("cancelled"):
                return

            if entry_price is None:

                # No entry reached
                # Continue to next analysis
                for _ in range(60):

                    if task.get("cancelled"):
                        return

                    await asyncio.sleep(5)

                continue

            # ---------------------------------------------
            # MONITOR OPEN TRADE
            # ---------------------------------------------

            await monitor_trade(
                bot=bot,
                chat_id=chat_id,
                trade=trade,
                task=task,
            )

            if task.get("cancelled"):
                return

            # ---------------------------------------------
            # SHORT DELAY BEFORE REANALYSIS
            # ---------------------------------------------

            for _ in range(12):

                if task.get("cancelled"):
                    return

                await asyncio.sleep(5)

        except asyncio.CancelledError:
            return

        except Exception as e:

            print(
                "Analysis loop error:",
                repr(e)
            )

            if task.get("cancelled"):
                return

            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚠️ حدث خطأ مؤقت أثناء التحليل.\n\n"
                        "🔄 سأحاول مرة أخرى بعد 1 دقيقة."
                    ),
                    reply_markup=stop_keyboard(),
                )

            except Exception as send_error:
                print(
                    "Error sending error message:",
                    send_error
                )

            # Retry after 1 minute
            for _ in range(12):

                if task.get("cancelled"):
                    return

                await asyncio.sleep(5)


# =========================================================
# START COMMAND
# =========================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    await update.message.reply_text(
        "🤖 RustyGold جاهز",
        reply_markup=main_keyboard(),
    )


# =========================================================
# START ANALYSIS
# =========================================================

async def start_analysis(
    query,
):
    chat_id = query.message.chat_id

    # Prevent duplicate analysis
    if chat_id in analysis_tasks:

        task = analysis_tasks[chat_id]

        if not task.get("cancelled"):

            await query.answer(
                "التحليل يعمل بالفعل."
            )

            return

    task_data = {
        "cancelled": False,
    }

    analysis_tasks[chat_id] = task_data

    await query.answer(
        "بدأ التحليل."
    )

    await query.message.reply_text(
        "🚀 تم تشغيل RustyGold.\n\n"
        "سيتم تحليل XAU/USD الآن.",
        reply_markup=stop_keyboard(),
    )

    asyncio.create_task(
        analysis_loop(
            bot=query.get_bot(),
            chat_id=chat_id,
            task=task_data,
        )
    )


# =========================================================
# STOP ANALYSIS
# =========================================================

async def stop_analysis(
    query,
):
    chat_id = query.message.chat_id

    task = analysis_tasks.get(chat_id)

    if task:

        # Set cancellation immediately
        task["cancelled"] = True

        # Remove immediately from active tasks
        analysis_tasks.pop(
            chat_id,
            None,
        )

        await query.answer(
            "تم إيقاف التحليل."
        )

        await query.message.reply_text(
            "🛑 تم إيقاف التحليل.",
            reply_markup=main_keyboard(),
        )

        return

    await query.answer(
        "لا يوجد تحليل يعمل حاليًا."
    )

    await query.message.reply_text(
        "ℹ️ لا يوجد تحليل يعمل حاليًا.",
        reply_markup=main_keyboard(),
    )


# =========================================================
# SUMMARY
# =========================================================

async def summary(
    query,
):
    trades = stats["trades"]
    wins = stats["wins"]
    losses = stats["losses"]
    pnl = stats["pnl"]

    if trades > 0:
        win_rate = (
            wins / trades
        ) * 100
    else:
        win_rate = 0

    if pnl > 0:
        pnl_text = f"+{pnl:.2f}"
    else:
        pnl_text = f"{pnl:.2f}"

    message = (
        "📊 ملخص RustyGold\n\n"
        f"الصفقات: {trades}\n"
        f"الرابحة: {wins}\n"
        f"الخاسرة: {losses}\n"
        f"نسبة الفوز: {win_rate:.2f}%\n"
        f"Simulated PnL: {pnl_text}\n\n"
        "⚠️ هذه أرقام محاكاة وليست تداولًا حقيقيًا."
    )

    await query.answer()

    await query.message.reply_text(
        message,
        reply_markup=main_keyboard(),
    )


# =========================================================
# BUTTON HANDLER
# =========================================================

async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    data = query.data

    if data == "start_analysis":

        await start_analysis(
            query
        )

    elif data == "stop_analysis":

        await stop_analysis(
            query
        )

    elif data == "summary":

        await summary(
            query
        )


# =========================================================
# ERROR HANDLER
# =========================================================

async def error_handler(
    update,
    context,
):
    print(
        "Telegram error:",
        repr(context.error)
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # Flask health server
    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True,
    )

    flask_thread.start()

    # Telegram application
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
            button_handler,
        )
    )

    application.add_error_handler(
        error_handler
    )

    print(
        "RustyGold Telegram bot started."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()

هذا هو التغيير المهم بالضبط:

- السعر يُفحص كل 5 ثوانٍ في الخلفية.
- لا يرسل رسالة كل 5 ثوانٍ.
- بعد 5 دقائق من بقاء الصفقة مفتوحة يرسل تحديثًا مثل:
  "👀 تحديث الصفقة — بعد 5 دقيقة"
- يعرض ENTRY + السعر الحالي + SL + TP1 + الربح/الخسارة من الدخول.
- إذا لمس السعر TP1 أو SL في أي لحظة، يرسل النتيجة فورًا ولا ينتظر تقرير الـ5 دقائق.
- زر 🛑 إيقاف التحليل يبقى موجودًا.
- لم أغيّر البرومبت أو شكل الأزرار.
- الـ"PnL" هنا محاكاة بفارق السعر وليس مبلغًا بالدولار.
