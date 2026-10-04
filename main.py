import os
import re
import io
import asyncio
import threading

import requests
import pandas as pd
import mplfinance as mpf

from flask import Flask
from PIL import Image

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from google import genai
from google.genai import types


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

missing = []

if not TELEGRAM_TOKEN:
    missing.append("TELEGRAM_TOKEN")

if not TWELVEDATA_KEY:
    missing.append("TWELVEDATA_KEY")

if not GEMINI_KEY:
    missing.append("GEMINI_KEY")

if missing:
    raise RuntimeError(
        "Missing environment variables: " + ", ".join(missing)
    )


# ============================================================
# CLIENTS
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_KEY)

app = Flask(__name__)


# ============================================================
# GLOBAL STATE
# ============================================================

tasks = {}

stats = {
    "wins": 0,
    "losses": 0,
    "trades": 0,
    "pnl": 0.0,
}


# ============================================================
# GEMINI PROMPT
# ============================================================

ICT_PROMPT = r"""
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

Examples:

Liquidity sweep + CHoCH + displacement

BOS + retest + momentum

Liquidity sweep + rejection + BOS

FVG + displacement + structure confirmation

Order Block + CHoCH + retest


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

Return exactly one of:

TRADE

or

NO TRADE

If TRADE, the first four lines MUST be exactly:

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

If NO TRADE, return:

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
# GEMINI MODELS
# ============================================================

def load_gemini_models():
    models = []

    try:
        for model in gemini_client.models.list():
            name = getattr(model, "name", None)

            if not name:
                continue

            name_lower = name.lower()

            if "gemini" not in name_lower:
                continue

            supported = getattr(
                model,
                "supported_actions",
                None
            )

            if supported:
                supported_text = str(supported).lower()

                if (
                    "generatecontent" not in supported_text
                    and "generate_content" not in supported_text
                ):
                    continue

            models.append(name)

    except Exception as e:
        print("Gemini model list error:", e)

    # Remove duplicates
    models = list(dict.fromkeys(models))

    # Prefer Flash models
    models.sort(
        key=lambda x: (
            0 if "flash" in x.lower() else 1,
            x
        )
    )

    return models


# ============================================================
# TWELVEDATA
# ============================================================

def fetch_candles(interval, outputsize):
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
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "status" in data and data["status"] == "error":
        raise RuntimeError(
            "TwelveData error: "
            + str(data.get("message", data))
        )

    values = data.get("values")

    if not values:
        raise RuntimeError(
            f"No candle data returned for {interval}"
        )

    df = pd.DataFrame(values)

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
# LIVE GOLD PRICE
# ============================================================

def fetch_gold_price():
    url = "https://api.gold-api.com/price/XAU"

    response = requests.get(
        url,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    price = data.get("price")

    if price is None:
        raise RuntimeError(
            "Gold API did not return a price."
        )

    return float(price)


# ============================================================
# DATA -> TEXT
# ============================================================

def dataframe_to_ohlc_text(df, max_rows=None):
    temp = df.copy()

    if max_rows:
        temp = temp.tail(max_rows)

    lines = [
        "datetime,open,high,low,close"
    ]

    for timestamp, row in temp.iterrows():

        lines.append(
            f"{timestamp},"
            f"{row['open']:.2f},"
            f"{row['high']:.2f},"
            f"{row['low']:.2f},"
            f"{row['close']:.2f}"
        )

    return "\n".join(lines)


# ============================================================
# CHART
# ============================================================

def create_chart(
    df,
    title,
    max_rows=None
):
    temp = df.copy()

    if max_rows:
        temp = temp.tail(max_rows)

    if len(temp) < 5:
        raise RuntimeError(
            f"Not enough candles for chart: {title}"
        )

    buffer = io.BytesIO()

    mpf.plot(
        temp,
        type="candle",
        style="charles",
        title=title,
        figsize=(14, 7),
        volume=False,
        savefig=dict(
            fname=buffer,
            dpi=130,
            bbox_inches="tight"
        )
    )

    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")

    return image


# ============================================================
# GEMINI CALL
# ============================================================

def call_gemini(
    model_name,
    prompt,
    image_1h,
    image_5m,
    image_1m
):

    def image_part(image):
        buffer = io.BytesIO()

        image.save(
            buffer,
            format="PNG"
        )

        return types.Part.from_bytes(
            data=buffer.getvalue(),
            mime_type="image/png"
        )

    contents = [
        prompt,

        "\nIMAGE 1 — 1H WEEK\n",
        image_part(image_1h),

        "\nIMAGE 2 — 5M LAST 24 HOURS\n",
        image_part(image_5m),

        "\nIMAGE 3 — 1M LAST 4 HOURS\n",
        image_part(image_1m),
    ]

    response = gemini_client.models.generate_content(
        model=model_name,
        contents=contents
    )

    text = getattr(
        response,
        "text",
        None
    )

    if not text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return text


# ============================================================
# ANALYZE WITH GEMINI
# ============================================================

def analyze_with_gemini(
    ohlc_1h,
    ohlc_5m,
    ohlc_1m,
    image_1h,
    image_5m,
    image_1m,
    current_price
):

    prompt = f"""
{ICT_PROMPT}

CURRENT XAU/USD PRICE:
{current_price:.2f}

==================================================
1H OHLC — APPROXIMATELY ONE WEEK
==================================================

{ohlc_1h}

==================================================
5M OHLC — APPROXIMATELY 24 HOURS
==================================================

{ohlc_5m}

==================================================
1M OHLC — LATEST DATA
==================================================

{ohlc_1m}

==================================================

Analyze the charts and OHLC data together.

Remember:

1H = context
5M = main structure
1M = entry confirmation

Return a trade only when the setup is actually coherent.
"""

    models = load_gemini_models()

    if not models:
        raise RuntimeError(
            "No usable Gemini models were found."
        )

    last_error = None

    for model_name in models:

        try:
            print(
                f"Trying Gemini model: {model_name}"
            )

            result = call_gemini(
                model_name,
                prompt,
                image_1h,
                image_5m,
                image_1m
            )

            print(
                f"Gemini success: {model_name}"
            )

            return result

        except Exception as e:

            last_error = e

            print(
                f"Gemini failed on {model_name}: {e}"
            )

    raise RuntimeError(
        "All Gemini models failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# PARSE TRADE
# ============================================================

def parse_trade(text):

    if not text:
        return None

    upper = text.upper()

    if "NO TRADE" in upper:
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

    if not all([
        direction_match,
        entry_match,
        sl_match,
        tp_match,
    ]):
        return None

    direction = direction_match.group(1)

    entry = float(entry_match.group(1))
    sl = float(sl_match.group(1))
    tp1 = float(tp_match.group(1))

    # Validate structure
    if direction == "BUY":

        if not (
            sl < entry < tp1
        ):
            print(
                "Invalid BUY setup:",
                entry,
                sl,
                tp1
            )
            return None

    elif direction == "SELL":

        if not (
            tp1 < entry < sl
        ):
            print(
                "Invalid SELL setup:",
                entry,
                sl,
                tp1
            )
            return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
    }


# ============================================================
# WAIT FOR ENTRY
# ============================================================

async def wait_for_entry(
    direction,
    entry,
    tolerance=0.60,
    max_seconds=300
):

    checks = int(max_seconds / 5)

    for _ in range(checks):

        try:
            price = await asyncio.to_thread(
                fetch_gold_price
            )

            print(
                f"Waiting entry | "
                f"Direction={direction} "
                f"Entry={entry:.2f} "
                f"Price={price:.2f}"
            )

            if abs(price - entry) <= tolerance:

                return price

        except Exception as e:

            print(
                "Entry price error:",
                e
            )

        await asyncio.sleep(5)

    return None


# ============================================================
# MONITOR SIMULATED TRADE
# ============================================================

async def monitor_trade(
    direction,
    entry,
    sl,
    tp1,
    max_seconds=3600
):

    checks = int(max_seconds / 5)

    for _ in range(checks):

        try:
            price = await asyncio.to_thread(
                fetch_gold_price
            )

            print(
                f"Trade monitor | "
                f"{direction} | "
                f"Entry={entry:.2f} | "
                f"SL={sl:.2f} | "
                f"TP={tp1:.2f} | "
                f"Price={price:.2f}"
            )

            if direction == "BUY":

                if price <= sl:
                    pnl = sl - entry

                    return {
                        "result": "LOSS",
                        "exit": sl,
                        "pnl": pnl,
                    }

                if price >= tp1:
                    pnl = tp1 - entry

                    return {
                        "result": "WIN",
                        "exit": tp1,
                        "pnl": pnl,
                    }

            else:

                if price >= sl:
                    pnl = entry - sl

                    return {
                        "result": "LOSS",
                        "exit": sl,
                        "pnl": pnl,
                    }

                if price <= tp1:
                    pnl = entry - tp1

                    return {
                        "result": "WIN",
                        "exit": tp1,
                        "pnl": pnl,
                    }

        except Exception as e:

            print(
                "Trade monitoring error:",
                e
            )

        await asyncio.sleep(5)

    return {
        "result": "TIMEOUT",
        "exit": None,
        "pnl": 0.0,
    }


# ============================================================
# SEND LONG TEXT SAFELY
# ============================================================

async def send_long_message(
    bot,
    chat_id,
    text
):

    max_length = 4000

    if len(text) <= max_length:

        await bot.send_message(
            chat_id=chat_id,
            text=text
        )

        return

    for i in range(
        0,
        len(text),
        max_length
    ):

        chunk = text[
            i:i + max_length
        ]

        await bot.send_message(
            chat_id=chat_id,
            text=chunk
        )


# ============================================================
# ANALYSIS LOOP
# ============================================================

async def analysis_loop(
    chat_id,
    bot
):

    await bot.send_message(
        chat_id=chat_id,
        text=(
            "🚀 RustyGold بدأ التحليل.\n\n"
            "1H = سياق\n"
            "5M = الهيكل الرئيسي\n"
            "1M = تأكيد الدخول\n\n"
            "التداول محاكاة فقط."
        )
    )

    while True:

        try:

            # --------------------------------------------
            # FETCH DATA
            # --------------------------------------------

            print("Fetching market data...")

            df_1h = await asyncio.to_thread(
                fetch_candles,
                "1h",
                168
            )

            df_5m = await asyncio.to_thread(
                fetch_candles,
                "5min",
                288
            )

            df_1m = await asyncio.to_thread(
                fetch_candles,
                "1min",
                720
            )

            current_price = await asyncio.to_thread(
                fetch_gold_price
            )

            print(
                f"Current XAU/USD: "
                f"{current_price:.2f}"
            )

            # --------------------------------------------
            # CREATE CHARTS
            # --------------------------------------------

            image_1h = await asyncio.to_thread(
                create_chart,
                df_1h,
                "XAU/USD — 1H — 1 Week",
                168
            )

            image_5m = await asyncio.to_thread(
                create_chart,
                df_5m,
                "XAU/USD — 5M — 24 Hours",
                288
            )

            image_1m = await asyncio.to_thread(
                create_chart,
                df_1m,
                "XAU/USD — 1M — Last 4 Hours",
                240
            )

            # --------------------------------------------
            # PREPARE OHLC
            # --------------------------------------------

            ohlc_1h = dataframe_to_ohlc_text(
                df_1h,
                168
            )

            ohlc_5m = dataframe_to_ohlc_text(
                df_5m,
                288
            )

            # Only latest 180 candles are sent as raw 1M text.
            # The chart still shows the latest 4 hours.
            ohlc_1m = dataframe_to_ohlc_text(
                df_1m,
                180
            )

            # --------------------------------------------
            # SEND MARKET SNAPSHOT
            # --------------------------------------------

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "🔎 تحليل XAU/USD...\n\n"
                    f"السعر الحالي: {current_price:.2f}\n\n"
                    "📊 1H: أسبوع\n"
                    "📊 5M: آخر 24 ساعة\n"
                    "📊 1M: تأكيد السكالب"
                )
            )

            # --------------------------------------------
            # GEMINI
            # --------------------------------------------

            gemini_result = await asyncio.to_thread(
                analyze_with_gemini,
                ohlc_1h,
                ohlc_5m,
                ohlc_1m,
                image_1h,
                image_5m,
                image_1m,
                current_price
            )

            print(
                "Gemini result:\n",
                gemini_result
            )

            # --------------------------------------------
            # PARSE
            # --------------------------------------------

            trade = parse_trade(
                gemini_result
            )

            # --------------------------------------------
            # NO TRADE
            # --------------------------------------------

            if trade is None:

                await send_long_message(
                    bot,
                    chat_id,
                    (
                        "⚪ NO TRADE\n\n"
                        + gemini_result
                        + "\n\n"
                        "⏳ سأعيد التحليل بعد 5 دقائق."
                    )
                )

                await asyncio.sleep(300)

                continue

            # --------------------------------------------
            # TRADE FOUND
            # --------------------------------------------

            direction = trade["direction"]
            entry = trade["entry"]
            sl = trade["sl"]
            tp1 = trade["tp1"]

            setup_message = (
                "🎯 TRADE SETUP — SIMULATION\n\n"
                f"📌 Direction: {direction}\n"
                f"🎯 Entry: {entry:.2f}\n"
                f"🛑 SL: {sl:.2f}\n"
                f"💰 TP1: {tp1:.2f}\n\n"
                "⏳ أنتظر وصول السعر لمنطقة الدخول...\n"
                "⚠️ محاكاة فقط — لا يوجد أمر حقيقي."
            )

            await bot.send_message(
                chat_id=chat_id,
                text=setup_message
            )

            # --------------------------------------------
            # WAIT FOR ENTRY
            # --------------------------------------------

            entry_price = await wait_for_entry(
                direction,
                entry,
                tolerance=0.60,
                max_seconds=300
            )

            # --------------------------------------------
            # ENTRY NOT REACHED
            # --------------------------------------------

            if entry_price is None:

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⌛ لم يصل السعر إلى منطقة الدخول "
                        "خلال 5 دقائق.\n\n"
                        "❌ تم إلغاء الصفقة المحاكاة.\n"
                        "🔄 سأعيد التحليل."
                    )
                )

                continue

            # --------------------------------------------
            # SIMULATED ENTRY
            # --------------------------------------------

            stats["trades"] += 1

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "🟢 SIMULATED ENTRY\n\n"
                    f"Direction: {direction}\n"
                    f"Entry: {entry_price:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}\n\n"
                    "📡 بدأت مراقبة الصفقة."
                )
            )

            # --------------------------------------------
            # MONITOR
            # --------------------------------------------

            result = await monitor_trade(
                direction,
                entry_price,
                sl,
                tp1
            )

            result_type = result["result"]
            pnl = result["pnl"]
            exit_price = result["exit"]

            # --------------------------------------------
            # UPDATE STATS
            # --------------------------------------------

            stats["pnl"] += pnl

            if result_type == "WIN":

                stats["wins"] += 1

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "✅ SIMULATED WIN\n\n"
                        f"Exit: {exit_price:.2f}\n"
                        f"Simulated PnL: {pnl:+.2f}\n\n"
                        "🔄 إعادة التحليل..."
                    )
                )

            elif result_type == "LOSS":

                stats["losses"] += 1

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "❌ SIMULATED LOSS\n\n"
                        f"Exit: {exit_price:.2f}\n"
                        f"Simulated PnL: {pnl:+.2f}\n\n"
                        "🔄 إعادة التحليل..."
                    )
                )

            else:

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⌛ انتهت مدة مراقبة الصفقة.\n\n"
                        "لم يتم احتسابها Win/Loss.\n"
                        "🔄 إعادة التحليل..."
                    )
                )

        except asyncio.CancelledError:

            print(
                f"Analysis task cancelled: {chat_id}"
            )

            raise

        except Exception as e:

            print(
                "ANALYSIS LOOP ERROR:",
                repr(e)
            )

            try:

                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚠️ حدث خطأ أثناء التحليل:\n\n"
                        f"{str(e)[:1500]}\n\n"
                        "⏳ سأحاول مرة أخرى بعد دقيقة."
                    )
                )

            except Exception as send_error:

                print(
                    "Telegram error:",
                    send_error
                )

            await asyncio.sleep(60)


# ============================================================
# TELEGRAM START
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

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
            )
        ],
    ]

    reply_markup = InlineKeyboardMarkup(
        keyboard
    )

    await update.message.reply_text(
        "🤖 RustyGold جاهز.\n\n"
        "XAU/USD — ICT / Price Action\n"
        "التداول محاكاة فقط.",
        reply_markup=reply_markup
    )


# ============================================================
# BUTTON HANDLER
# ============================================================

async def button_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
):

    query = update.callback_query

    await query.answer()

    chat_id = query.message.chat_id

    # --------------------------------------------
    # START
    # --------------------------------------------

    if query.data == "start_analysis":

        existing_task = tasks.get(chat_id)

        if (
            existing_task
            and not existing_task.done()
        ):

            await query.message.reply_text(
                "⚠️ التحليل يعمل بالفعل."
            )

            return

        task = asyncio.create_task(
            analysis_loop(
                chat_id,
                context.bot
            )
        )

        tasks[chat_id] = task

        await query.message.reply_text(
            "🚀 تم تشغيل RustyGold."
        )

    # --------------------------------------------
    # STOP
    # --------------------------------------------

    elif query.data == "stop_analysis":

        task = tasks.get(chat_id)

        if task and not task.done():

            task.cancel()

            # Remove immediately.
            # We intentionally do not await the task here.
            tasks.pop(chat_id, None)

            await query.message.reply_text(
                "🛑 تم إيقاف التحليل."
            )

        else:

            tasks.pop(chat_id, None)

            await query.message.reply_text(
                "ℹ️ لا يوجد تحليل يعمل حاليًا."
            )

    # --------------------------------------------
    # SUMMARY
    # --------------------------------------------

    elif query.data == "summary":

        trades = stats["trades"]
        wins = stats["wins"]
        losses = stats["losses"]
        pnl = stats["pnl"]

        if trades > 0:
            win_rate = (
                wins / trades
            ) * 100
        else:
            win_rate = 0.0

        summary = (
            "📊 RustyGold Summary\n\n"
            f"Trades: {trades}\n"
            f"Wins: {wins}\n"
            f"Losses: {losses}\n"
            f"Win Rate: {win_rate:.2f}%\n"
            f"Simulated PnL: {pnl:+.2f}\n\n"
            "⚠️ PnL هنا فرق سعري محاكى "
            "وليس أرباحًا بالدولار.\n"
            "⚠️ الإحصائيات تُحفظ في الذاكرة "
            "وتُصفّر عند إعادة تشغيل السيرفر."
        )

        await query.message.reply_text(
            summary
        )


# ============================================================
# FLASK HEALTH CHECK
# ============================================================

@app.route("/")
def home():

    return "RustyGold is running."


def run_flask():

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                10000
            )
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("Starting RustyGold...")

    # Render health server
    flask_thread = threading.Thread(
        target=run_flask,
        daemon=True
    )

    flask_thread.start()

    print("Flask health server started.")

    telegram_app = (
        Application
        .builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    telegram_app.add_handler(
        CommandHandler(
            "start",
            start_command
        )
    )

    telegram_app.add_handler(
        CallbackQueryHandler(
            button_handler
        )
    )

    print("Telegram bot starting...")

    telegram_app.run_polling(
        drop_pending_updates=True
    )


if __name__ == "__main__":
    main()
