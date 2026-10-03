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
# ENVIRONMENT
# =========================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

SYMBOL = "XAU/USD"

if not TELEGRAM_TOKEN or not TWELVEDATA_KEY or not GEMINI_KEY:
    missing = []

    if not TELEGRAM_TOKEN:
        missing.append("TELEGRAM_TOKEN")

    if not TWELVEDATA_KEY:
        missing.append("TWELVEDATA_KEY")

    if not GEMINI_KEY:
        missing.append("GEMINI_KEY")

    raise RuntimeError(
        "Missing environment variables: " + ", ".join(missing)
    )


# =========================================================
# GEMINI
# =========================================================

gemini_client = genai.Client(api_key=GEMINI_KEY)

gemini_models = []


def load_gemini_models():
    """
    Find available Gemini models that support generateContent.
    """

    global gemini_models

    models = []

    try:
        available = gemini_client.models.list()

        for model in available:

            name = getattr(model, "name", "")

            if not name:
                continue

            actions = getattr(model, "supported_actions", [])

            if actions and "generateContent" not in actions:
                continue

            if "gemini" not in name.lower():
                continue

            models.append(name)

        # Prefer Flash models because they are generally faster.
        models.sort(
            key=lambda x: (
                "flash" not in x.lower(),
                x.lower(),
            )
        )

        gemini_models = models

        print("Available Gemini models:")

        for model in gemini_models:
            print(" -", model)

    except Exception as e:

        print("Failed to load Gemini models:", e)
        gemini_models = []


load_gemini_models()


# =========================================================
# THREAD POOL
# =========================================================

executor = ThreadPoolExecutor(max_workers=4)


# =========================================================
# BOT STATE
# =========================================================

tasks = {}

stats = {
    "wins": 0,
    "losses": 0,
    "trades": 0,
    "pnl": 0.0,
}


# =========================================================
# ICT PROMPT
# =========================================================

ICT_PROMPT = """
أنت محلل XAU/USD متخصص في ICT وPrice Action.

لديك أربعة مصادر للمعلومات:

1. بيانات OHLC دقيقة لفريم 5m.
2. بيانات OHLC دقيقة لفريم 1m.
3. صورة شارت 5m.
4. صورة شارت 1m.
5. السعر الحالي.

استخدم بيانات OHLC كأساس أساسي لتحديد المستويات والأسعار الدقيقة.

استخدم صور الشارت لفهم السياق البصري والبنية العامة.

لا تعتمد على الصورة وحدها إذا تعارضت مع أرقام OHLC.

========================
الخطوة 1 — 5M STRUCTURE
========================

حلل فريم 5m أولاً.

حدد:

- HH
- HL
- LH
- LL
- BOS
- CHoCH
- الاتجاه
- أو Range / تذبذب

حدد آخر بنية واضحة قبل السعر الحالي.

========================
الخطوة 2 — LIQUIDITY
========================

ابحث عن:

- Previous High
- Previous Low
- Equal Highs
- Equal Lows
- Buy-side liquidity
- Sell-side liquidity
- Liquidity Sweep

إذا حدث Sweep، حدد المستوى السعري الذي تم أخذه.

لا تعتبر مجرد لمس مستوى Sweep مؤكداً.

========================
الخطوة 3 — 1M CONFIRMATION
========================

بعد تحديد اتجاه 5m، انتقل إلى 1m.

ابحث عن:

- BOS
- CHoCH
- Displacement
- FVG
- Order Block
- Retest

لا تشترط وجود جميع عناصر ICT.

يكفي وجود مجموعة متوافقة ومنطقية مثل:

Structure + Liquidity + Confirmation

========================
الخطوة 4 — الاتجاه
========================

لا تدخل ضد اتجاه 5m إلا إذا ظهر على 1m تغير واضح في البنية بعد Liquidity Sweep.

إذا كان السوق متذبذباً وغير واضح:

NO TRADE

========================
الخطوة 5 — ENTRY
========================

حدد ENTRY من مستوى سعري حقيقي موجود في بيانات OHLC.

لا تخترع سعراً عشوائياً.

إذا كان الدخول من Retest:

حدد المستوى الذي يجب أن يعود إليه السعر.

========================
الخطوة 6 — STOP LOSS
========================

ضع SL خلف مستوى إبطال واضح.

BUY:
SL يجب أن يكون أسفل القاع أو منطقة الإبطال.

SELL:
SL يجب أن يكون أعلى القمة أو منطقة الإبطال.

لا تضع SL عشوائياً.

========================
الخطوة 7 — TAKE PROFIT
========================

حدد TP1 عند أقرب:

- Liquidity
- Previous High/Low
- Swing High/Low
- أو هدف سعري منطقي تدعمه البيانات.

========================
التحقق النهائي
========================

قبل إرسال الصفقة تحقق من ترتيب الأسعار.

BUY:

SL < ENTRY < TP1

SELL:

TP1 < ENTRY < SL

إذا لم يتحقق هذا الترتيب:

NO TRADE

========================
قاعدة مهمة
========================

لا تجبر نفسك على إعطاء صفقة.

إذا لم يكن هناك Setup واضح تدعمه البيانات:

NO TRADE

لا تخترع صفقة فقط لتجنب NO TRADE.

========================
الإخراج
========================

إذا وجدت صفقة:

TRADE
DIRECTION: BUY
ENTRY: 0000.00
SL: 0000.00
TP1: 0000.00

REASON:
5M BIAS: ...
LIQUIDITY: ...
1M STRUCTURE: ...
ENTRY: ...

أو:

TRADE
DIRECTION: SELL
ENTRY: 0000.00
SL: 0000.00
TP1: 0000.00

REASON:
5M BIAS: ...
LIQUIDITY: ...
1M STRUCTURE: ...
ENTRY: ...

إذا لا توجد صفقة:

NO TRADE
REASON: ...

========================
CURRENT PRICE
========================

CURRENT PRICE:
{price}

========================
5M OHLC DATA
========================

{ohlc_5m}

========================
1M OHLC DATA
========================

{ohlc_1m}
"""


# =========================================================
# TWELVEDATA
# =========================================================

def fetch_candles(interval, outputsize):

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

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    df = df.set_index("datetime")

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    df = df.sort_index()

    return df


# =========================================================
# GOLD PRICE
# =========================================================

def fetch_gold_price():

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

def dataframe_to_ohlc_text(df, max_rows):

    recent = df.tail(max_rows)

    lines = [
        "datetime,open,high,low,close"
    ]

    for idx, row in recent.iterrows():

        timestamp = idx.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        lines.append(
            f"{timestamp},"
            f"{row['open']:.2f},"
            f"{row['high']:.2f},"
            f"{row['low']:.2f},"
            f"{row['close']:.2f}"
        )

    return "\n".join(lines)


# =========================================================
# CHART
# =========================================================

def create_chart(df, title):

    chart_df = df.copy()

    chart_df.index.name = "Date"

    buf = io.BytesIO()

    mpf.plot(
        chart_df,
        type="candle",
        style="charles",
        title=title,
        ylabel="XAU/USD",
        volume=False,
        figsize=(12, 6),
        savefig=dict(
            fname=buf,
            dpi=120,
            bbox_inches="tight",
        ),
    )

    buf.seek(0)

    return buf.getvalue()


# =========================================================
# GEMINI ANALYSIS
# =========================================================

def generate_gemini_content(
    model_name,
    prompt,
    image_1m,
    image_5m,
):

    contents = [
        prompt,
        {
            "mime_type": "image/png",
            "data": image_5m,
        },
        {
            "mime_type": "image/png",
            "data": image_1m,
        },
    ]

    return gemini_client.models.generate_content(
        model=model_name,
        contents=contents,
    )


def analyze_with_gemini(
    price,
    image_1m,
    image_5m,
    ohlc_1m,
    ohlc_5m,
):

    if not gemini_models:
        load_gemini_models()

    if not gemini_models:
        return (
            "NO TRADE\n"
            "REASON: No available Gemini model."
        )

    prompt = ICT_PROMPT.format(
        price=f"{price:.2f}",
        ohlc_5m=ohlc_5m,
        ohlc_1m=ohlc_1m,
    )

    for model_name in gemini_models:

        try:

            print(
                f"Trying Gemini model: {model_name}"
            )

            response = generate_gemini_content(
                model_name,
                prompt,
                image_1m,
                image_5m,
            )

            text = getattr(
                response,
                "text",
                None,
            )

            if text:

                print(
                    f"Gemini success: {model_name}"
                )

                print(text)

                return text

        except Exception as e:

            print(
                f"Gemini model failed "
                f"{model_name}: {e}"
            )

    return (
        "NO TRADE\n"
        "REASON: All Gemini models failed."
    )


# =========================================================
# TRADE PARSER
# =========================================================

def extract_number(text, labels):

    for label in labels:

        pattern = (
            rf"{label}"
            rf"\s*[:=]\s*"
            rf"(\d+(?:\.\d+)?)"
        )

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:
            return float(match.group(1))

    return None


def parse_trade(text):

    if not text:
        return None

    upper = text.upper()

    if "NO TRADE" in upper:
        return None

    direction = None

    if re.search(
        r"DIRECTION\s*[:=]\s*BUY",
        upper,
    ):
        direction = "BUY"

    elif re.search(
        r"DIRECTION\s*[:=]\s*SELL",
        upper,
    ):
        direction = "SELL"

    else:

        if re.search(r"\bBUY\b", upper):
            direction = "BUY"

        elif re.search(r"\bSELL\b", upper):
            direction = "SELL"

    if direction is None:
        return None

    entry = extract_number(
        upper,
        [
            "ENTRY",
            "ENTRY PRICE",
        ],
    )

    sl = extract_number(
        upper,
        [
            "SL",
            "STOP LOSS",
            "STOPLOSS",
        ],
    )

    tp1 = extract_number(
        upper,
        [
            "TP1",
            "TP",
            "TAKE PROFIT",
            "TAKE PROFIT 1",
        ],
    )

    if (
        entry is None
        or sl is None
        or tp1 is None
    ):
        return None

    # Validate price ordering.
    if direction == "BUY":

        if not (
            sl < entry < tp1
        ):
            print(
                "Invalid BUY ordering:",
                entry,
                sl,
                tp1,
            )
            return None

    elif direction == "SELL":

        if not (
            tp1 < entry < sl
        ):
            print(
                "Invalid SELL ordering:",
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
# TELEGRAM MESSAGE
# =========================================================

async def send_message(
    context,
    chat_id,
    text,
):

    try:

        await context.bot.send_message(
            chat_id=chat_id,
            text=text,
        )

    except Exception as e:

        print(
            "Telegram send error:",
            e,
        )


# =========================================================
# WAIT FOR ENTRY
# =========================================================

async def wait_for_entry(
    context,
    chat_id,
    trade,
):

    entry = trade["entry"]

    # Entry zone requested for simulation.
    tolerance = 0.60

    # Monitor for 5 minutes.
    end_time = (
        asyncio.get_running_loop().time()
        + 300
    )

    while (
        asyncio.get_running_loop().time()
        < end_time
    ):

        try:

            price = await asyncio.to_thread(
                fetch_gold_price
            )

            distance = abs(
                price - entry
            )

            print(
                f"Waiting entry | "
                f"Price={price:.2f} "
                f"Entry={entry:.2f} "
                f"Distance={distance:.2f}"
            )

            if distance <= tolerance:

                return price

        except Exception as e:

            print(
                "Entry price error:",
                e,
            )

        await asyncio.sleep(5)

    return None


# =========================================================
# MONITOR TRADE
# =========================================================

async def monitor_trade(
    context,
    chat_id,
    trade,
):

    direction = trade["direction"]
    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]

    while True:

        try:

            price = await asyncio.to_thread(
                fetch_gold_price
            )

            print(
                f"Trade monitor | "
                f"{direction} | "
                f"Price={price:.2f} | "
                f"SL={sl:.2f} | "
                f"TP={tp1:.2f}"
            )

            if direction == "BUY":

                if price <= sl:

                    return "LOSS", price

                if price >= tp1:

                    return "WIN", price

            elif direction == "SELL":

                if price >= sl:

                    return "LOSS", price

                if price <= tp1:

                    return "WIN", price

        except Exception as e:

            print(
                "Trade monitor error:",
                e,
            )

        await asyncio.sleep(5)


# =========================================================
# ANALYSIS LOOP
# =========================================================

async def analysis_loop(
    context,
    chat_id,
):

    print(
        f"Started analysis loop for {chat_id}"
    )

    while True:

        try:

            # -----------------------------------------
            # FETCH DATA
            # -----------------------------------------

            df_1m = await asyncio.to_thread(
                fetch_candles,
                "1min",
                60,
            )

            df_5m = await asyncio.to_thread(
                fetch_candles,
                "5min",
                96,
            )

            price = await asyncio.to_thread(
                fetch_gold_price
            )

            # -----------------------------------------
            # CREATE CHARTS
            # -----------------------------------------

            image_1m = await asyncio.to_thread(
                create_chart,
                df_1m,
                "XAU/USD - 1 Minute",
            )

            image_5m = await asyncio.to_thread(
                create_chart,
                df_5m,
                "XAU/USD - 5 Minute",
            )

            # -----------------------------------------
            # CREATE OHLC TEXT
            # -----------------------------------------

            # Send the latest 60 candles on 1m.
            ohlc_1m = dataframe_to_ohlc_text(
                df_1m,
                60,
            )

            # Send the latest 48 candles on 5m.
            # 48 candles = approximately 4 hours.
            ohlc_5m = dataframe_to_ohlc_text(
                df_5m,
                48,
            )

            # -----------------------------------------
            # GEMINI
            # -----------------------------------------

            result = await asyncio.to_thread(
                analyze_with_gemini,
                price,
                image_1m,
                image_5m,
                ohlc_1m,
                ohlc_5m,
            )

            trade = parse_trade(result)

            # -----------------------------------------
            # NO TRADE
            # -----------------------------------------

            if trade is None:

                await send_message(
                    context,
                    chat_id,
                    (
                        "🔎 RustyGold\n\n"
                        f"السعر الحالي: {price:.2f}\n\n"
                        f"{result}"
                    ),
                )

                # Reanalyze after 5 minutes.
                await asyncio.sleep(300)

                continue

            # -----------------------------------------
            # TRADE FOUND
            # -----------------------------------------

            direction = trade["direction"]
            entry = trade["entry"]
            sl = trade["sl"]
            tp1 = trade["tp1"]

            await send_message(
                context,
                chat_id,
                (
                    "🚨 TRADE SETUP\n\n"
                    f"Direction: {direction}\n"
                    f"Entry: {entry:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}\n\n"
                    f"{result}"
                ),
            )

            # -----------------------------------------
            # WAIT FOR ENTRY
            # -----------------------------------------

            entry_price = await wait_for_entry(
                context,
                chat_id,
                trade,
            )

            if entry_price is None:

                await send_message(
                    context,
                    chat_id,
                    (
                        "⌛ Entry not reached "
                        "within 5 minutes.\n\n"
                        "Setup cancelled."
                    ),
                )

                continue

            # -----------------------------------------
            # SIMULATED TRADE START
            # -----------------------------------------

            stats["trades"] += 1

            await send_message(
                context,
                chat_id,
                (
                    "🟢 SIMULATED ENTRY\n\n"
                    f"Direction: {direction}\n"
                    f"Entry: {entry:.2f}\n"
                    f"Current: {entry_price:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}"
                ),
            )

            # -----------------------------------------
            # MONITOR
            # -----------------------------------------

            result_type, exit_price = (
                await monitor_trade(
                    context,
                    chat_id,
                    trade,
                )
            )

            # -----------------------------------------
            # WIN
            # -----------------------------------------

            if result_type == "WIN":

                stats["wins"] += 1

                if direction == "BUY":
                    pnl = exit_price - entry
                else:
                    pnl = entry - exit_price

                stats["pnl"] += pnl

                await send_message(
                    context,
                    chat_id,
                    (
                        "✅ TP1 HIT\n\n"
                        f"Exit: {exit_price:.2f}\n"
                        f"Simulated PnL: {pnl:+.2f}"
                    ),
                )

            # -----------------------------------------
            # LOSS
            # -----------------------------------------

            else:

                stats["losses"] += 1

                if direction == "BUY":
                    pnl = exit_price - entry
                else:
                    pnl = entry - exit_price

                stats["pnl"] += pnl

                await send_message(
                    context,
                    chat_id,
                    (
                        "❌ STOP LOSS HIT\n\n"
                        f"Exit: {exit_price:.2f}\n"
                        f"Simulated PnL: {pnl:+.2f}"
                    ),
                )

            # -----------------------------------------
            # REANALYZE
            # -----------------------------------------

            await asyncio.sleep(5)

        except asyncio.CancelledError:

            print(
                f"Analysis loop cancelled "
                f"for {chat_id}"
            )

            raise

        except Exception as e:

            print(
                "Analysis loop error:",
                e,
            )

            await send_message(
                context,
                chat_id,
                (
                    "⚠️ Error أثناء التحليل:\n"
                    f"{e}"
                ),
            )

            # Avoid rapid error loop.
            await asyncio.sleep(30)


# =========================================================
# START
# =========================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    keyboard = [
        [
            InlineKeyboardButton(
                "🚀 حلل يا جيمني",
                callback_data="start_analysis",
            )
        ],
        [
            InlineKeyboardButton(
                "🛑 إيقاف",
                callback_data="stop_analysis",
            )
        ],
        [
            InlineKeyboardButton(
                "📊 ملخص",
                callback_data="summary",
            )
        ],
    ]

    await update.message.reply_text(
        "RustyGold جاهز 🟡",
        reply_markup=InlineKeyboardMarkup(
            keyboard
        ),
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

    # -----------------------------------------
    # START ANALYSIS
    # -----------------------------------------

    if query.data == "start_analysis":

        existing = tasks.get(chat_id)

        if (
            existing
            and not existing.done()
        ):

            await query.message.reply_text(
                "⚠️ التحليل يعمل بالفعل."
            )

            return

        task = asyncio.create_task(
            analysis_loop(
                context,
                chat_id,
            )
        )

        tasks[chat_id] = task

        await query.message.reply_text(
            "🚀 بدأت المراقبة والتحليل."
        )

    # -----------------------------------------
    # STOP
    # -----------------------------------------

    elif query.data == "stop_analysis":

        task = tasks.get(chat_id)

        if task and not task.done():

            # Cancel immediately.
            task.cancel()

            # Do not await it here.
            # This makes STOP respond immediately.

            tasks.pop(chat_id, None)

            await query.message.reply_text(
                "🛑 تم إيقاف التحليل."
            )

        else:

            await query.message.reply_text(
                "لا يوجد تحليل يعمل حالياً."
            )

    # -----------------------------------------
    # SUMMARY
    # -----------------------------------------

    elif query.data == "summary":

        total = stats["trades"]
        wins = stats["wins"]
        losses = stats["losses"]
        pnl = stats["pnl"]

        if total > 0:
            winrate = (
                wins / total
            ) * 100
        else:
            winrate = 0

        await query.message.reply_text(
            (
                "📊 RustyGold Summary\n\n"
                f"Trades: {total}\n"
                f"Wins: {wins}\n"
                f"Losses: {losses}\n"
                f"Win rate: {winrate:.1f}%\n"
                f"Simulated PnL: {pnl:+.2f}"
            )
        )


# =========================================================
# FLASK
# =========================================================

flask_app = Flask(__name__)


@flask_app.route("/")
def home():

    return "RustyGold is running."


def run_flask():

    port = int(
        os.getenv(
            "PORT",
            "10000",
        )
    )

    flask_app.run(
        host="0.0.0.0",
        port=port,
    )


# =========================================================
# MAIN
# =========================================================

def main():

    # Flask keeps Render service alive.
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
            start,
        )
    )

    application.add_handler(
        CallbackQueryHandler(
            button_handler
        )
    )

    print(
        "RustyGold bot is starting..."
    )

    application.run_polling()


if __name__ == "__main__":
    main()



