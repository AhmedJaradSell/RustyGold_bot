import os
import re
import io
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
import mplfinance as mpf

from flask import Flask
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
)

from google import genai


# =========================================================
# CONFIG
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

SYMBOL = "XAU/USD"

app = Flask(__name__)

# One background task per Telegram chat
tasks = {}

# Trading statistics
stats = {
    "wins": 0,
    "losses": 0,
    "trades": 0,
    "pnl": 0.0,
}

# Gemini client
gemini_client = genai.Client(api_key=GEMINI_KEY)

AVAILABLE_GEMINI_MODELS = []

# Used for blocking HTTP/API requests without freezing asyncio
executor = ThreadPoolExecutor(max_workers=4)


# =========================================================
# ICT PROMPT
# =========================================================

ICT_PROMPT = """
أنت محلل تداول متخصص في ICT وPrice Action.

أمامك صورتان لشارت XAU/USD:

1) فريم 1 دقيقة
2) فريم 5 دقائق

ولديك السعر الحقيقي الحالي.

مهمتك البحث عن SETUP حقيقي وليس انتظار وجود جميع عناصر ICT في نفس الوقت.

========================
طريقة التحليل
========================

أولًا:

حدد اتجاه وبنية السوق على فريم 5 دقائق:

- صاعد
- هابط
- أو متذبذب/غير واضح

ابحث عن:
- Higher High / Higher Low
- Lower High / Lower Low
- BOS
- CHoCH
- مناطق السيولة

ثم استخدم فريم 1 دقيقة للبحث عن دخول يتوافق مع اتجاه 5 دقائق قدر الإمكان.

========================
عناصر ICT
========================

ابحث عن:

- BOS
- CHoCH
- Liquidity Sweep
- FVG
- Order Block
- Displacement
- Previous High / Previous Low
- Equal Highs / Equal Lows
- مناطق السيولة القريبة

لكن انتبه:

لا يشترط وجود جميع هذه العناصر.

يمكن اعتبار الصفقة صالحة إذا وجدت مجموعة منطقية من الأدلة، مثل:

1) اتجاه واضح + BOS + FVG

أو:

2) اتجاه واضح + Liquidity Sweep + CHoCH

أو:

3) اتجاه واضح + Sweep + FVG

أو:

4) اتجاه واضح + Order Block + displacement

أو أي تركيبة مشابهة لها منطق واضح.

لا تبحث عن الكمال.

========================
الدخول
========================

إذا وجدت Setup واضحًا:

حدد:

ENTRY
SL
TP1

ENTRY يمكن أن يكون:

- السعر الحالي
- أو منطقة إعادة اختبار FVG
- أو منطقة Order Block
- أو مستوى كسر/إعادة اختبار واضح

SL يجب أن يكون خلف منطقة إبطال الفكرة:

في BUY:
أسفل القاع/منطقة الإبطال.

في SELL:
أعلى القمة/منطقة الإبطال.

TP1 يكون عند أقرب هدف منطقي للسيولة أو القمة/القاع المقابل.

لا تضع أرقامًا عشوائية.

========================
مهم جدًا
========================

لا تقل NO TRADE لمجرد أن أحد عناصر ICT غير موجود.

قل NO TRADE فقط عندما:

- الاتجاه غير واضح جدًا
- أو لا توجد بنية قابلة للتداول
- أو لا توجد منطقة دخول منطقية
- أو SL/TP لا يمكن تحديدهما بشكل منطقي

إذا كانت هناك فرصة معقولة وواضحة، أعطِ الصفقة.

لا تخترع صفقة فقط لتجنب NO TRADE.

========================
صيغة الإخراج
========================

إذا لم توجد فرصة:

NO TRADE

إذا وجدت فرصة:

TRADE
DIRECTION: BUY
ENTRY: 3850.25
SL: 3847.80
TP1: 3855.00

أو:

TRADE
DIRECTION: SELL
ENTRY: 3850.25
SL: 3853.00
TP1: 3845.50

بعد الأرقام اكتب شرحًا قصيرًا جدًا:

BIAS: BUY/SELL
STRUCTURE: ...
LIQUIDITY: ...
ENTRY REASON: ...
INVALIDATION: ...

لا تكتب تحليلاً طويلًا.

السعر الحقيقي الحالي:
{price}
"""


# =========================================================
# TELEGRAM UI
# =========================================================

def main_keyboard():
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "🚀 حلل يا جيمني",
                    callback_data="start_analysis",
                )
            ],
            [
                InlineKeyboardButton(
                    "🛑 إيقاف",
                    callback_data="stop",
                )
            ],
            [
                InlineKeyboardButton(
                    "📊 ملخص",
                    callback_data="summary",
                )
            ],
        ]
    )


# =========================================================
# TWELVEDATA
# =========================================================

def get_candles(interval, outputsize):
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

    data = response.json()

    if "values" not in data:
        raise RuntimeError(
            f"TwelveData error: {data}"
        )

    df = pd.DataFrame(data["values"])

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime")

    df.set_index("datetime", inplace=True)

    for column in ["open", "high", "low", "close"]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(
            df["volume"],
            errors="coerce",
        )

    df.dropna(
        subset=["open", "high", "low", "close"],
        inplace=True,
    )

    return df


# =========================================================
# LIVE GOLD PRICE
# =========================================================

def get_live_gold_price():
    url = "https://api.gold-api.com/price/XAU"

    response = requests.get(
        url,
        timeout=15,
    )

    data = response.json()

    price = data.get("price")

    if price is None:
        raise RuntimeError(
            f"Gold API error: {data}"
        )

    return float(price)


# =========================================================
# CHART RENDERING
# =========================================================

def make_chart(df, title):
    chart_df = df.copy()

    chart_df.index.name = "Date"

    buf = io.BytesIO()

    mpf.plot(
        chart_df,
        type="candle",
        style="charles",
        volume=False,
        figsize=(12, 6),
        title=title,
        tight_layout=True,
        savefig=dict(
            fname=buf,
            dpi=120,
            bbox_inches="tight",
        ),
    )

    buf.seek(0)

    return buf.getvalue()


# =========================================================
# GEMINI MODEL DISCOVERY
# =========================================================

def get_available_gemini_models():
    models = list(
        gemini_client.models.list()
    )

    available = []

    for model in models:
        name = getattr(
            model,
            "name",
            None,
        )

        actions = getattr(
            model,
            "supported_actions",
            [],
        ) or []

        if not name:
            continue

        if "generateContent" not in actions:
            continue

        if "gemini" not in name.lower():
            continue

        available.append(name)

    # Prefer flash models because they are generally more suitable
    # for repeated analysis loops.
    def model_priority(name):
        name_lower = name.lower()

        if "flash" in name_lower:
            return 0

        if "pro" in name_lower:
            return 1

        return 2

    available.sort(
        key=model_priority
    )

    return available


def load_gemini_models():
    global AVAILABLE_GEMINI_MODELS

    try:
        AVAILABLE_GEMINI_MODELS = (
            get_available_gemini_models()
        )

        print(
            "\n========== GEMINI MODELS =========="
        )

        if not AVAILABLE_GEMINI_MODELS:
            print("No compatible Gemini models found.")

        for model in AVAILABLE_GEMINI_MODELS:
            print(model)

        print(
            "===================================\n"
        )

        return AVAILABLE_GEMINI_MODELS

    except Exception as e:
        print(
            f"Gemini model discovery error: {e}"
        )

        AVAILABLE_GEMINI_MODELS = []

        return []


# =========================================================
# GEMINI ANALYSIS
# =========================================================

def analyze_with_gemini(
    price,
    image_1m,
    image_5m,
):
    global AVAILABLE_GEMINI_MODELS

    if not AVAILABLE_GEMINI_MODELS:
        load_gemini_models()

    if not AVAILABLE_GEMINI_MODELS:
        raise RuntimeError(
            "No Gemini models available."
        )

    prompt = ICT_PROMPT.format(
        price=f"{price:.2f}"
    )

    last_error = None

    for model_name in AVAILABLE_GEMINI_MODELS:

        try:

            print(
                f"Trying Gemini model: {model_name}"
            )

            response = (
                gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_1m,
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type": "image/png",
                                "data": image_5m,
                            }
                        },
                    ],
                )
            )

            text = getattr(
                response,
                "text",
                None,
            )

            if text and text.strip():

                print(
                    f"Gemini success: {model_name}"
                )

                return (
                    text.strip(),
                    model_name,
                )

        except Exception as e:

            print(
                f"Gemini failed on {model_name}: {e}"
            )

            last_error = e

            continue

    raise RuntimeError(
        f"All Gemini models failed: {last_error}"
    )


# =========================================================
# TRADE PARSER
# =========================================================

def extract_number(text):
    """
    Extract the first decimal number from text.
    Handles commas and optional USD-like text.
    """

    if not text:
        return None

    match = re.search(
        r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?",
        text,
    )

    if not match:
        return None

    value = match.group(0)

    value = value.replace(",", "")

    try:
        return float(value)
    except ValueError:
        return None


def extract_field(text, field_names):
    """
    Extract a numeric value after one of several possible labels.

    Examples accepted:

    ENTRY: 3850.25
    ENTRY = 3850.25
    ENTRY 3850.25
    Entry Price: 3850.25
    BUY @ 3850.25
    """

    for field in field_names:

        pattern = (
            rf"{re.escape(field)}"
            rf"\s*(?:PRICE)?"
            rf"\s*[:=\\-]?\s*"
            rf"([0-9][0-9,]*(?:\.[0-9]+)?)"
        )

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match:

            value = (
                match.group(1)
                .replace(",", "")
            )

            try:
                return float(value)
            except ValueError:
                pass

    return None


def parse_direction(text):
    upper = text.upper()

    buy_patterns = [
        r"\bBUY\b",
        r"\bLONG\b",
    ]

    sell_patterns = [
        r"\bSELL\b",
        r"\bSHORT\b",
    ]

    buy = any(
        re.search(
            pattern,
            upper,
        )
        for pattern in buy_patterns
    )

    sell = any(
        re.search(
            pattern,
            upper,
        )
        for pattern in sell_patterns
    )

    if buy and not sell:
        return "BUY"

    if sell and not buy:
        return "SELL"

    # If the model writes:
    # BUY @ 3850
    if re.search(
        r"\bBUY\s*@",
        upper,
    ):
        return "BUY"

    if re.search(
        r"\bSELL\s*@",
        upper,
    ):
        return "SELL"

    return None


def parse_trade(text):
    """
    Parse Gemini output into:

    {
        direction,
        entry,
        sl,
        tp1
    }

    Returns None for NO TRADE or invalid setup.
    """

    if not text:
        return None

    upper = text.upper()

    # Explicit no-trade response
    if re.search(
        r"\bNO\s*TRADE\b",
        upper,
    ):
        return None

    direction = parse_direction(
        text
    )

    # Standard labels
    entry = extract_field(
        text,
        [
            "ENTRY",
            "ENTRY PRICE",
            "ENTRY LEVEL",
        ],
    )

    sl = extract_field(
        text,
        [
            "SL",
            "STOP LOSS",
            "STOPLOSS",
        ],
    )

    tp1 = extract_field(
        text,
        [
            "TP1",
            "TP",
            "TAKE PROFIT",
            "TAKE PROFIT 1",
        ],
    )

    # Fallback for BUY @ price / SELL @ price
    if entry is None:

        match = re.search(
            r"\b(?:BUY|SELL|LONG|SHORT)"
            r"\s*@\s*"
            r"([0-9][0-9,]*(?:\.[0-9]+)?)",
            upper,
        )

        if match:
            entry = float(
                match.group(1).replace(",", "")
            )

    if (
        direction is None
        or entry is None
        or sl is None
        or tp1 is None
    ):
        return None

    # Basic numerical sanity checks
    if entry <= 0 or sl <= 0 or tp1 <= 0:
        return None

    if direction == "BUY":

        # For BUY, SL must be below entry
        # and TP1 must be above entry.
        if not (
            sl < entry < tp1
        ):
            print(
                "Invalid BUY setup:",
                entry,
                sl,
                tp1,
            )

            return None

    elif direction == "SELL":

        # For SELL, TP1 must be below entry
        # and SL must be above entry.
        if not (
            tp1 < entry < sl
        ):
            print(
                "Invalid SELL setup:",
                entry,
                sl,
                tp1,
            )

            return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
    }


# =========================================================
# FORMAT TRADE
# =========================================================

def format_trade(trade):
    return (
        f"📌 {trade['direction']}\n"
        f"ENTRY: {trade['entry']:.2f}\n"
        f"SL: {trade['sl']:.2f}\n"
        f"TP1: {trade['tp1']:.2f}"
    )


# =========================================================
# ANALYSIS LOOP
# =========================================================

async def analysis_loop(
    chat_id,
    context,
):

    current_task = asyncio.current_task()

    print(
        f"Analysis loop started for {chat_id}"
    )

    try:

        while True:

            print(
                f"Starting new analysis for {chat_id}"
            )

            # -------------------------------------------------
            # Fetch market data
            # -------------------------------------------------

            loop = asyncio.get_running_loop()

            try:

                df_1m, df_5m, price = (
                    await asyncio.gather(
                        loop.run_in_executor(
                            executor,
                            get_candles,
                            "1min",
                            60,
                        ),
                        loop.run_in_executor(
                            executor,
                            get_candles,
                            "5min",
                            96,
                        ),
                        loop.run_in_executor(
                            executor,
                            get_live_gold_price,
                        ),
                    )
                )

            except asyncio.CancelledError:
                raise

            except Exception as e:

                print(
                    f"Market data error: {e}"
                )

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚠️ تعذر جلب بيانات السوق.\n"
                        "سيتم إعادة المحاولة بعد دقيقة."
                    ),
                )

                await asyncio.sleep(60)

                continue

            # -------------------------------------------------
            # Make charts
            # -------------------------------------------------

            try:

                image_1m = await loop.run_in_executor(
                    executor,
                    make_chart,
                    df_1m,
                    "XAU/USD - 1 Minute",
                )

                image_5m = await loop.run_in_executor(
                    executor,
                    make_chart,
                    df_5m,
                    "XAU/USD - 5 Minute",
                )

            except asyncio.CancelledError:
                raise

            except Exception as e:

                print(
                    f"Chart error: {e}"
                )

                await asyncio.sleep(30)

                continue

            # -------------------------------------------------
            # Send current market snapshot
            # -------------------------------------------------

            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "🔎 جاري تحليل XAU/USD...\n"
                    f"السعر الحالي: {price:.2f}"
                ),
            )

            # -------------------------------------------------
            # Gemini
            # -------------------------------------------------

            try:

                analysis, model_name = (
                    await loop.run_in_executor(
                        executor,
                        analyze_with_gemini,
                        price,
                        image_1m,
                        image_5m,
                    )
                )

            except asyncio.CancelledError:
                raise

            except Exception as e:

                print(
                    f"Gemini analysis error: {e}"
                )

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚠️ تعذر تحليل الشارت بواسطة Gemini.\n"
                        "إعادة المحاولة بعد دقيقة."
                    ),
                )

                await asyncio.sleep(60)

                continue

            print(
                "\n========== GEMINI =========="
            )
            print(
                f"MODEL: {model_name}"
            )
            print(analysis)
            print(
                "============================\n"
            )

            trade = parse_trade(
                analysis
            )

            # -------------------------------------------------
            # NO TRADE
            # -------------------------------------------------

            if trade is None:

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚪ NO TRADE\n\n"
                        f"Gemini: {model_name}\n"
                        f"السعر: {price:.2f}\n\n"
                        "سيتم إعادة التحليل بعد 5 دقائق."
                    ),
                )

                await asyncio.sleep(
                    5 * 60
                )

                continue

            # -------------------------------------------------
            # TRADE FOUND
            # -------------------------------------------------

            stats["trades"] += 1

            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "🚨 SETUP FOUND\n\n"
                    + format_trade(trade)
                    + "\n\n"
                    f"Gemini: {model_name}\n"
                    "سأراقب الوصول إلى منطقة الدخول."
                ),
            )

            # -------------------------------------------------
            # WAIT FOR ENTRY
            # -------------------------------------------------

            entry = trade["entry"]
            sl = trade["sl"]
            tp1 = trade["tp1"]
            direction = trade["direction"]

            entry_reached = False

            # Maximum entry-wait time = 5 minutes
            for _ in range(60):

                await asyncio.sleep(5)

                try:

                    current_price = (
                        await loop.run_in_executor(
                            executor,
                            get_live_gold_price,
                        )
                    )

                except asyncio.CancelledError:
                    raise

                except Exception:
                    continue

                # -------------------------------------------------
                # Entry distance
                # -------------------------------------------------

                if abs(
                    current_price - entry
                ) <= 0.60:

                    entry_reached = True

                    await context.bot.send_message(
                        chat_id=chat_id,
                        text=(
                            "🟢 SIMULATED ENTRY\n\n"
                            + format_trade(trade)
                            + f"\n\nالسعر الحالي: {current_price:.2f}"
                        ),
                    )

                    break

            # -------------------------------------------------
            # Entry not reached
            # -------------------------------------------------

            if not entry_reached:

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚪ تم إلغاء الـ setup.\n"
                        "السعر لم يصل إلى منطقة الدخول خلال 5 دقائق.\n\n"
                        "سيتم البحث عن فرصة جديدة."
                    ),
                )

                continue

            # -------------------------------------------------
            # MONITOR TRADE
            # -------------------------------------------------

            result = None

            while True:

                await asyncio.sleep(5)

                try:

                    current_price = (
                        await loop.run_in_executor(
                            executor,
                            get_live_gold_price,
                        )
                    )

                except asyncio.CancelledError:
                    raise

                except Exception:
                    continue

                # BUY
                if direction == "BUY":

                    if current_price <= sl:

                        result = "LOSS"

                        break

                    if current_price >= tp1:

                        result = "WIN"

                        break

                # SELL
                elif direction == "SELL":

                    if current_price >= sl:

                        result = "LOSS"

                        break

                    if current_price <= tp1:

                        result = "WIN"

                        break

            # -------------------------------------------------
            # Result
            # -------------------------------------------------

            if result == "WIN":

                stats["wins"] += 1

                # Simulated PnL using 1R = entry-to-SL distance.
                risk = abs(
                    entry - sl
                )

                reward = abs(
                    tp1 - entry
                )

                pnl = reward

                stats["pnl"] += pnl

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "✅ TP1 HIT\n\n"
                        f"Direction: {direction}\n"
                        f"Entry: {entry:.2f}\n"
                        f"TP1: {tp1:.2f}\n\n"
                        f"Simulated PnL: +{pnl:.2f}"
                    ),
                )

            else:

                stats["losses"] += 1

                risk = abs(
                    entry - sl
                )

                pnl = -risk

                stats["pnl"] += pnl

                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "❌ SL HIT\n\n"
                        f"Direction: {direction}\n"
                        f"Entry: {entry:.2f}\n"
                        f"SL: {sl:.2f}\n\n"
                        f"Simulated PnL: {pnl:.2f}"
                    ),
                )

            # -------------------------------------------------
            # New analysis immediately
            # -------------------------------------------------

            await asyncio.sleep(2)

    except asyncio.CancelledError:

        print(
            f"Analysis task cancelled for {chat_id}"
        )

        raise

    except Exception as e:

        print(
            f"Analysis loop crashed: {e}"
        )

        try:

            await context.bot.send_message(
                chat_id=chat_id,
                text=(
                    "⚠️ توقفت عملية التحليل بسبب خطأ.\n"
                    "اضغط 🚀 لبدء تحليل جديد."
                ),
            )

        except Exception:
            pass

    finally:

        # Only remove the task if this is still
        # the currently registered task.
        if (
            tasks.get(chat_id)
            is current_task
        ):
            tasks.pop(
                chat_id,
                None,
            )

        print(
            f"Analysis loop finished for {chat_id}"
        )


# =========================================================
# /START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    await update.message.reply_text(
        "🤖 RustyGold جاهز\n\n"
        "اختر الأمر:",
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

    await query.answer()

    chat_id = query.message.chat_id

    # -------------------------------------------------
    # START ANALYSIS
    # -------------------------------------------------

    if query.data == "start_analysis":

        existing = tasks.get(chat_id)

        if existing and not existing.done():

            await query.message.reply_text(
                "⚠️ التحليل يعمل بالفعل."
            )

            return

        task = asyncio.create_task(
            analysis_loop(
                chat_id,
                context,
            )
        )

        tasks[chat_id] = task

        await query.message.reply_text(
            "🚀 بدأ RustyGold التحليل المستمر."
        )

    # -------------------------------------------------
    # STOP
    # -------------------------------------------------

    elif query.data == "stop":

        task = tasks.get(chat_id)

        if task and not task.done():

            # Cancel immediately.
            # We intentionally do NOT await the task.
            task.cancel()

            tasks.pop(
                chat_id,
                None,
            )

            await query.message.reply_text(
                "🛑 تم إيقاف المهمة فورًا."
            )

        else:

            await query.message.reply_text(
                "ℹ️ لا توجد مهمة تحليل تعمل حاليًا."
            )

    # -------------------------------------------------
    # SUMMARY
    # -------------------------------------------------

    elif query.data == "summary":

        trades = stats["trades"]
        wins = stats["wins"]
        losses = stats["losses"]
        pnl = stats["pnl"]

        if trades > 0:
            winrate = (
                wins / trades
            ) * 100
        else:
            winrate = 0

        await query.message.reply_text(
            "📊 RustyGold Summary\n\n"
            f"Trades: {trades}\n"
            f"Wins: {wins}\n"
            f"Losses: {losses}\n"
            f"Win rate: {winrate:.1f}%\n"
            f"Simulated PnL: {pnl:.2f}"
        )


# =========================================================
# FLASK
# =========================================================

@app.route("/")
def home():
    return "RustyGold is running."


def run_flask():
    app.run(
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "10000",
            )
        ),
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # Validate environment variables
    missing = []

    if not TELEGRAM_TOKEN:
        missing.append(
            "TELEGRAM_TOKEN"
        )

    if not TWELVEDATA_KEY:
        missing.append(
            "TWELVEDATA_KEY"
        )

    if not GEMINI_KEY:
        missing.append(
            "GEMINI_KEY"
        )

    if missing:

        raise RuntimeError(
            "Missing environment variables: "
            + ", ".join(missing)
        )

    # Load Gemini models at startup
    load_gemini_models()

    # Flask in background
    threading.Thread(
        target=run_flask,
        daemon=True,
    ).start()

    # Telegram
    application = (
        Application.builder()
        .token(TELEGRAM_TOKEN)
        .build()
    )

    application.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            button_handler,
        )
    )

    print(
        "RustyGold Telegram bot started."
    )

    application.run_polling()


if __name__ == "__main__":
    main()
