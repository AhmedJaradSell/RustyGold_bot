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


# ============================================================
# ENVIRONMENT
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
# GEMINI
# ============================================================

gemini_client = genai.Client(api_key=GEMINI_KEY)


# ============================================================
# FLASK HEALTH SERVER
# ============================================================

app = Flask(__name__)


@app.route("/")
def home():
    return "RustyGold is running."


@app.route("/health")
def health():
    return "OK"


def run_flask():
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)


# ============================================================
# GLOBAL STATE
# ============================================================

analysis_tasks = {}

stats = {}


def get_stats(chat_id):
    if chat_id not in stats:
        stats[chat_id] = {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pnl": 0.0,
        }

    return stats[chat_id]


# ============================================================
# TELEGRAM KEYBOARDS
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
    """
    This keyboard is attached to the LAST analysis message.
    """

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
# GEMINI MODEL DISCOVERY
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

    # Prefer Flash models
    models.sort(
        key=lambda x: (
            0 if "flash" in x.lower() else 1,
            x
        )
    )

    # Fallbacks
    if not models:
        models = [
            "gemini-2.5-flash",
            "gemini-2.0-flash",
        ]

    print("Gemini models:", models)

    return models


# ============================================================
# DATA
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

    df["datetime"] = pd.to_datetime(df["datetime"])

    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna()

    df = df.sort_values("datetime")

    df = df.set_index("datetime")

    return df


def get_gold_price():
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

    chart_df.index = pd.to_datetime(chart_df.index)

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
            bbox_inches="tight"
        )
    )

    buffer.seek(0)

    return buffer.read()


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def call_gemini_analysis(
    chart_1h,
    chart_5m,
    chart_1m,
    ohlc_1h,
    ohlc_5m,
    ohlc_1m,
    current_price,
):
    prompt = f"""
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
Use the live price as the current market reference.

Remember:
Return TRADE or NO TRADE first.
If TRADE, provide DIRECTION, ENTRY, SL and TP1.
"""

    parts = [
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
        types.Part.from_text(
            text=prompt
        ),
    ]

    models = get_gemini_models()

    last_error = None

    for model_name in models:
        try:
            print("Trying Gemini:", model_name)

            response = gemini_client.models.generate_content(
                model=model_name,
                contents=parts,
            )

            text = getattr(response, "text", None)

            if text and text.strip():
                print("Gemini success:", model_name)
                return text.strip()

        except Exception as e:
            print(
                f"Gemini model failed {model_name}:",
                e
            )

            last_error = e

    raise RuntimeError(
        f"All Gemini models failed: {last_error}"
    )


# ============================================================
# TRANSLATE ANALYSIS
# ============================================================

def translate_analysis(text):
    prompt = TRANSLATION_PROMPT + "\n" + text

    models = get_gemini_models()

    last_error = None

    for model_name in models:
        try:
            response = gemini_client.models.generate_content(
                model=model_name,
                contents=prompt,
            )

            translated = getattr(
                response,
                "text",
                None
            )

            if translated and translated.strip():
                return translated.strip()

        except Exception as e:
            print(
                f"Translation failed {model_name}:",
                e
            )

            last_error = e

    print(
        "Translation failed, using original analysis:",
        last_error
    )

    return text


# ============================================================
# PARSE TRADE
# ============================================================

def parse_trade(text):
    if not text:
        return None

    # Remove markdown emphasis
    clean = text.replace("*", "").strip()

    # Decision must be explicit
    lines = [
        line.strip()
        for line in clean.splitlines()
        if line.strip()
    ]

    decision = None

    for line in lines[:5]:
        upper = line.upper()

        if re.fullmatch(
            r"TRADE[:\s]*",
            upper
        ):
            decision = "TRADE"
            break

        if re.fullmatch(
            r"NO TRADE[:\s]*",
            upper
        ):
            decision = "NO TRADE"
            break

    if decision == "NO TRADE":
        return None

    if decision != "TRADE":
        # Fallback
        if re.search(
            r"\bNO\s+TRADE\b",
            clean,
            re.IGNORECASE
        ):
            return None

        if not re.search(
            r"\bTRADE\b",
            clean,
            re.IGNORECASE
        ):
            return None

    direction_match = re.search(
        r"DIRECTION\s*:\s*(BUY|SELL)",
        clean,
        re.IGNORECASE
    )

    entry_match = re.search(
        r"ENTRY\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        clean,
        re.IGNORECASE
    )

    sl_match = re.search(
        r"SL\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        clean,
        re.IGNORECASE
    )

    tp_match = re.search(
        r"TP1\s*:\s*([0-9]+(?:\.[0-9]+)?)",
        clean,
        re.IGNORECASE
    )

    if not all([
        direction_match,
        entry_match,
        sl_match,
        tp_match,
    ]):
        print(
            "Could not parse trade:",
            clean
        )
        return None

    return {
        "direction": direction_match.group(1).upper(),
        "entry": float(entry_match.group(1)),
        "sl": float(sl_match.group(1)),
        "tp1": float(tp_match.group(1)),
        "raw": clean,
    }


# ============================================================
# SEND LONG MESSAGE
# ============================================================

async def send_long_message(
    bot,
    chat_id,
    text,
    stop_button=False,
):
    """
    Sends text in chunks because Telegram has a message length limit.

    IMPORTANT:
    The STOP button is attached ONLY to the final chunk.
    """

    max_length = 3900

    chunks = []

    while len(text) > max_length:
        cut = text.rfind(
            "\n",
            0,
            max_length
        )

        if cut <= 0:
            cut = max_length

        chunks.append(
            text[:cut]
        )

        text = text[cut:].lstrip()

    if text:
        chunks.append(text)

    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1

        reply_markup = (
            stop_keyboard()
            if stop_button and is_last
            else None
        )

        await bot.send_message(
            chat_id=chat_id,
            text=chunk,
            reply_markup=reply_markup,
        )


# ============================================================
# WAIT FOR ENTRY
# ============================================================

async def wait_for_entry(
    bot,
    chat_id,
    trade,
):
    entry = trade["entry"]
    direction = trade["direction"]

    start_time = asyncio.get_running_loop().time()

    timeout = 5 * 60

    while True:

        if (
            asyncio.get_running_loop().time()
            - start_time
            > timeout
        ):
            return None

        price = await asyncio.to_thread(
            get_gold_price
        )

        distance = abs(
            price - entry
        )

        if distance <= 0.60:

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "🎯 تم الوصول إلى منطقة الدخول\n\n"
                    f"DIRECTION: {direction}\n"
                    f"ENTRY: {entry:.2f}\n"
                    f"Current Price: {price:.2f}"
                ),
            )

            return price

        await asyncio.sleep(5)


# ============================================================
# MONITOR TRADE
# ============================================================

async def monitor_trade(
    bot,
    chat_id,
    trade,
    entry_price,
):
    direction = trade["direction"]
    sl = trade["sl"]
    tp1 = trade["tp1"]

    start_time = asyncio.get_running_loop().time()

    timeout = 30 * 60

    while True:

        if (
            asyncio.get_running_loop().time()
            - start_time
            > timeout
        ):
            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⏱️ انتهت مدة مراقبة الصفقة "
                    "بدون وصول إلى SL أو TP1."
                ),
            )

            return

        price = await asyncio.to_thread(
            get_gold_price
        )

        hit = None

        if direction == "BUY":

            if price <= sl:
                hit = "SL"

            elif price >= tp1:
                hit = "TP1"

        elif direction == "SELL":

            if price >= sl:
                hit = "SL"

            elif price <= tp1:
                hit = "TP1"

        if hit:

            pnl = (
                price - entry_price
                if direction == "BUY"
                else entry_price - price
            )

            s = get_stats(chat_id)

            s["trades"] += 1
            s["pnl"] += pnl

            if hit == "TP1":
                s["wins"] += 1
            else:
                s["losses"] += 1

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    f"🏁 انتهت الصفقة: {hit}\n\n"
                    f"DIRECTION: {direction}\n"
                    f"Entry: {entry_price:.2f}\n"
                    f"Exit: {price:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}\n\n"
                    f"Simulated PnL: {pnl:+.2f}"
                ),
            )

            return

        await asyncio.sleep(5)


# ============================================================
# ANALYSIS LOOP
# ============================================================

async def analysis_loop(
    application,
    chat_id,
):
    bot = application.bot

    await bot.send_message(
        chat_id=chat_id,
        text=(
            "🚀 بدأ RustyGold التحليل...\n"
            "سيتم تحليل XAU/USD باستخدام 1H + 5M + 1M."
        ),
        reply_markup=stop_keyboard(),
    )

    while True:

        # ----------------------------------------------------
        # Fetch data
        # ----------------------------------------------------

        try:
            df_1h = await asyncio.to_thread(
                get_twelvedata,
                "1h",
                168,
            )

            df_5m = await asyncio.to_thread(
                get_twelvedata,
                "5min",
                288,
            )

            df_1m = await asyncio.to_thread(
                get_twelvedata,
                "1min",
                720,
            )

            current_price = await asyncio.to_thread(
                get_gold_price
            )

        except asyncio.CancelledError:
            raise

        except Exception as e:
            print("Data error:", e)

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ حدث خطأ أثناء جلب بيانات الذهب.\n"
                    f"{str(e)[:500]}"
                ),
                reply_markup=stop_keyboard(),
            )

            await asyncio.sleep(30)
            continue

        # ----------------------------------------------------
        # Prepare Gemini data
        # ----------------------------------------------------

        df_1m_for_gemini = df_1m.tail(180)
        df_1m_chart = df_1m.tail(240)

        ohlc_1h = format_ohlc(df_1h)

        ohlc_5m = format_ohlc(df_5m)

        ohlc_1m = format_ohlc(
            df_1m_for_gemini
        )

        chart_1h = await asyncio.to_thread(
            create_chart,
            df_1h,
            "XAU/USD 1H"
        )

        chart_5m = await asyncio.to_thread(
            create_chart,
            df_5m,
            "XAU/USD 5M"
        )

        chart_1m = await asyncio.to_thread(
            create_chart,
            df_1m_chart,
            "XAU/USD 1M"
        )

        # ----------------------------------------------------
        # Gemini analysis
        # ----------------------------------------------------

        try:
            raw_analysis = await asyncio.to_thread(
                call_gemini_analysis,
                chart_1h,
                chart_5m,
                chart_1m,
                ohlc_1h,
                ohlc_5m,
                ohlc_1m,
                current_price,
            )

        except asyncio.CancelledError:
            raise

        except Exception as e:
            print("Analysis error:", e)

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ حدث خطأ في تحليل Gemini.\n"
                    f"{str(e)[:500]}"
                ),
                reply_markup=stop_keyboard(),
            )

            await asyncio.sleep(30)
            continue

        print("\n========== GEMINI RAW ==========")
        print(raw_analysis)
        print("================================\n")

        # ----------------------------------------------------
        # Parse BEFORE translation
        # ----------------------------------------------------

        trade = parse_trade(
            raw_analysis
        )

        # ----------------------------------------------------
        # Translate explanation to Arabic
        # ----------------------------------------------------

        try:
            translated_analysis = await asyncio.to_thread(
                translate_analysis,
                raw_analysis
            )

        except asyncio.CancelledError:
            raise

        except Exception as e:
            print("Translation error:", e)

            translated_analysis = raw_analysis

        # ----------------------------------------------------
        # NO TRADE
        # ----------------------------------------------------

        if trade is None:

            message = (
                "🧠 RustyGold\n\n"
                f"{translated_analysis}\n\n"
                "⏳ لا توجد صفقة حاليًا.\n"
                "سيتم إعادة التحليل بعد 5 دقائق."
            )

            # IMPORTANT:
            # STOP BUTTON IS ON THE LAST ANALYSIS MESSAGE.
            await send_long_message(
                bot,
                chat_id,
                message,
                stop_button=True,
            )

            await asyncio.sleep(5 * 60)

            continue

        # ----------------------------------------------------
        # TRADE FOUND
        # ----------------------------------------------------

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
            "📖 التحليل:\n"
            f"{translated_analysis}\n\n"
            "🎯 في انتظار وصول السعر إلى ENTRY..."
        )

        # IMPORTANT:
        # STOP BUTTON IS ON THE FINAL CHUNK OF THIS MESSAGE.
        await send_long_message(
            bot,
            chat_id,
            trade_message,
            stop_button=True,
        )

        # ----------------------------------------------------
        # Wait for entry
        # ----------------------------------------------------

        entry_price = await wait_for_entry(
            bot,
            chat_id,
            trade,
        )

        if entry_price is None:

            await bot.send_message(
                chat_id=chat_id,
                text=(
                    "⏱️ لم يصل السعر إلى ENTRY خلال "
                    "المدة المحددة.\n"
                    "سيتم البحث عن فرصة جديدة."
                ),
                reply_markup=stop_keyboard(),
            )

            continue

        # ----------------------------------------------------
        # Monitor active simulated trade
        # ----------------------------------------------------

        await monitor_trade(
            bot,
            chat_id,
            trade,
            entry_price,
        )

        # ----------------------------------------------------
        # After trade, immediately reanalyze
        # ----------------------------------------------------

        await bot.send_message(
            chat_id=chat_id,
            text=(
                "🔄 انتهت الصفقة.\n"
                "سيتم إعادة تحليل السوق للبحث عن فرصة جديدة."
            ),
            reply_markup=stop_keyboard(),
        )


# ============================================================
# START COMMAND
# ============================================================

async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    chat_id = update.effective_chat.id

    text = (
        "🤖 RustyGold جاهز\n\n"
        "بوت تحليل XAU/USD باستخدام "
        "ICT + Price Action.\n\n"
        "اختر من الأزرار:"
    )

    await update.message.reply_text(
        text,
        reply_markup=main_keyboard(),
    )


# ============================================================
# STOP ANALYSIS
# ============================================================

async def stop_analysis(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):
    query = update.callback_query

    await query.answer(
        "🛑 تم إيقاف التحليل"
    )

    chat_id = query.message.chat_id

    task = analysis_tasks.get(
        chat_id
    )

    if task:

        task.cancel()

        # Remove immediately.
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
            "لا يوجد تحليل يعمل حاليًا.",
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

    chat_id = query.message.chat_id

    s = get_stats(chat_id)

    trades = s["trades"]
    wins = s["wins"]
    losses = s["losses"]
    pnl = s["pnl"]

    if trades > 0:
        win_rate = (
            wins / trades
        ) * 100
    else:
        win_rate = 0

    text = (
        "📊 ملخص RustyGold\n\n"
        f"Trades: {trades}\n"
        f"Wins: {wins}\n"
        f"Losses: {losses}\n"
        f"Win Rate: {win_rate:.2f}%\n"
        f"Simulated PnL: {pnl:+.2f}\n\n"
        "⚠️ هذه نتائج محاكاة وليست تداولًا حقيقيًا."
    )

    await query.message.reply_text(
        text,
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

    if query.data == "start_analysis":

        await query.answer(
            "🚀 بدأ التحليل"
        )

        chat_id = query.message.chat_id

        existing = analysis_tasks.get(
            chat_id
        )

        if existing and not existing.done():

            await query.message.reply_text(
                "⚠️ التحليل يعمل بالفعل.\n"
                "استخدم زر 🛑 إيقاف التحليل لإيقافه.",
                reply_markup=stop_keyboard(),
            )

            return

        task = asyncio.create_task(
            analysis_loop(
                context.application,
                chat_id,
            )
        )

        analysis_tasks[chat_id] = task

        def cleanup(done_task):
            current = analysis_tasks.get(
                chat_id
            )

            if current is done_task:
                analysis_tasks.pop(
                    chat_id,
                    None
                )

        task.add_done_callback(
            cleanup
        )

        return

    if query.data == "stop_analysis":

        await stop_analysis(
            update,
            context,
        )

        return

    if query.data == "summary":

        await summary(
            update,
            context,
        )

        return


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

    # Start Flask for Render
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
            start_command
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
        "RustyGold bot is starting..."
    )

    application.run_polling(
        allowed_updates=Update.ALL_TYPES
    )


if __name__ == "__main__":
    main()
