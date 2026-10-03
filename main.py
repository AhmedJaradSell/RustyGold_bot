import os
import re
import io
import asyncio
import threading

import requests
import pandas as pd
import mplfinance as mpf

from PIL import Image
from flask import Flask

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)

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
        "Missing environment variables: "
        + ", ".join(missing)
    )


# =========================================================
# GEMINI
# =========================================================

gemini_client = genai.Client(
    api_key=GEMINI_KEY
)

gemini_models = []


def load_gemini_models():

    global gemini_models

    models = []

    try:

        available = gemini_client.models.list()

        for model in available:

            name = getattr(
                model,
                "name",
                "",
            )

            if not name:
                continue

            actions = getattr(
                model,
                "supported_actions",
                [],
            )

            if actions:
                if "generateContent" not in actions:
                    continue

            if "gemini" not in name.lower():
                continue

            models.append(name)

        # Prefer Flash models for speed.
        models.sort(
            key=lambda x: (
                "flash" not in x.lower(),
                x.lower(),
            )
        )

        gemini_models = models

        print("\nAvailable Gemini models:")

        for model in gemini_models:
            print(" -", model)

    except Exception as e:

        print(
            "Failed to load Gemini models:",
            e,
        )

        gemini_models = []


load_gemini_models()


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

أمامك:

1. صورة 1H لأسبوع كامل.
2. صورة 5M لآخر 24 ساعة.
3. صورة 1M لآخر 4 ساعات.
4. بيانات OHLC لفريم 1H لأسبوع.
5. بيانات OHLC لفريم 5M لآخر 24 ساعة.
6. بيانات OHLC لفريم 1M لآخر 180 شمعة.
7. السعر الحالي.

استخدم الأرقام كأساس لتحديد الأسعار والمستويات الدقيقة.
استخدم الصور لفهم الشكل البصري والبنية والسياق.

لا تعتمد على الصورة وحدها عندما تحتاج إلى تحديد سعر دقيق.

==================================================
PRIORITY
==================================================

حلل بالترتيب:

1H → 5M → 1M

ولا تبدأ من 1M ثم تحاول اختراع اتجاه من التفاصيل الصغيرة.

==================================================
1H — HIGHER TIMEFRAME
==================================================

حدد:

- الاتجاه العام
- HH
- HL
- LH
- LL
- مناطق القمم والقيعان المهمة
- أعلى وأدنى الأسبوع
- مناطق السيولة الرئيسية
- هل السوق Trending أم Ranging

لا تحتاج إلى ذكر كل شمعة.

حدد فقط البنية المهمة التي تؤثر على السعر الحالي.

==================================================
5M — INTRADAY STRUCTURE
==================================================

حلل آخر 24 ساعة.

ابحث عن:

- BOS
- CHoCH
- Swing High
- Swing Low
- Previous High
- Previous Low
- Equal Highs
- Equal Lows
- Buy-side Liquidity
- Sell-side Liquidity
- Liquidity Sweep

حدد آخر بنية واضحة قبل السعر الحالي.

==================================================
1M — ENTRY CONFIRMATION
==================================================

استخدم آخر 180 شمعة كبيانات دقيقة.

ابحث عن:

- Liquidity Sweep
- BOS
- CHoCH
- Displacement
- FVG
- Order Block
- Retest

لا تشترط وجود كل عناصر ICT.

يكفي وجود مجموعة متوافقة ومنطقية:

Structure + Liquidity + Confirmation

==================================================
IMPORTANT
==================================================

لا تدخل ضد اتجاه 1H/5M بدون سبب واضح.

إذا حدث Liquidity Sweep ثم ظهر تغير واضح في البنية على 1M، يمكن اعتبار ذلك سبباً لتغير السيناريو.

إذا كان السوق غير واضح أو متذبذباً:

NO TRADE

==================================================
ENTRY
==================================================

حدد ENTRY من سعر حقيقي موجود في بيانات OHLC أو من مستوى واضح في الشارت.

لا تخترع سعراً عشوائياً.

إذا كان الدخول يعتمد على Retest:

اذكر المستوى الذي يجب أن يعود إليه السعر.

==================================================
STOP LOSS
==================================================

ضع SL خلف مستوى إبطال واضح.

BUY:
SL أسفل القاع أو منطقة الإبطال.

SELL:
SL أعلى القمة أو منطقة الإبطال.

==================================================
TP1
==================================================

حدد TP1 عند أقرب هدف منطقي مثل:

- Liquidity
- Previous High
- Previous Low
- Swing High
- Swing Low
- منطقة سعرية واضحة تدعمها البيانات

==================================================
FINAL VALIDATION
==================================================

قبل إرسال الصفقة تحقق من ترتيب الأسعار:

BUY:
SL < ENTRY < TP1

SELL:
TP1 < ENTRY < SL

إذا لم يتحقق ذلك:

NO TRADE

==================================================
DO NOT FORCE A TRADE
==================================================

إذا لم يوجد Setup واضح:

NO TRADE

لا تخترع صفقة فقط لتجنب NO TRADE.

==================================================
OUTPUT
==================================================

إذا وجدت صفقة:

TRADE
DIRECTION: BUY
ENTRY: 0000.00
SL: 0000.00
TP1: 0000.00

REASON:
1H BIAS: ...
5M STRUCTURE: ...
LIQUIDITY: ...
1M CONFIRMATION: ...
ENTRY: ...

أو:

TRADE
DIRECTION: SELL
ENTRY: 0000.00
SL: 0000.00
TP1: 0000.00

REASON:
1H BIAS: ...
5M STRUCTURE: ...
LIQUIDITY: ...
1M CONFIRMATION: ...
ENTRY: ...

إذا لا توجد صفقة:

NO TRADE
REASON: ...

==================================================
CURRENT PRICE
==================================================

{price}

==================================================
1H OHLC — ONE WEEK
==================================================

{ohlc_1h}

==================================================
5M OHLC — LAST 24 HOURS
==================================================

{ohlc_5m}

==================================================
1M OHLC — LAST 180 CANDLES
==================================================

{ohlc_1m}
"""


# =========================================================
# TWELVEDATA
# =========================================================

def fetch_candles(
    interval,
    outputsize,
):

    url = (
        "https://api.twelvedata.com/"
        "time_series"
    )

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

    df["datetime"] = pd.to_datetime(
        df["datetime"]
    )

    df = df.set_index(
        "datetime"
    )

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

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
        ]
    )

    df = df.sort_index()

    return df


# =========================================================
# GOLD PRICE
# =========================================================

def fetch_gold_price():

    url = (
        "https://api.gold-api.com/"
        "price/XAU"
    )

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
# OHLC TEXT
# =========================================================

def dataframe_to_ohlc_text(
    df,
    max_rows=None,
):

    if max_rows is not None:
        data = df.tail(max_rows)
    else:
        data = df

    lines = [
        "datetime,open,high,low,close"
    ]

    for idx, row in data.iterrows():

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
# CHART CREATION
# =========================================================

def create_chart(
    df,
    title,
    max_rows=None,
):

    if max_rows is not None:
        chart_df = df.tail(max_rows).copy()
    else:
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
        figsize=(14, 7),
        tight_layout=True,
        savefig={
            "fname": buf,
            "dpi": 130,
            "bbox_inches": "tight",
        },
    )

    buf.seek(0)

    return buf.getvalue()


# =========================================================
# IMAGE CONVERSION
# =========================================================

def bytes_to_image(
    image_bytes,
):

    image = Image.open(
        io.BytesIO(image_bytes)
    )

    # Make sure the image is RGB.
    if image.mode != "RGB":
        image = image.convert("RGB")

    return image


# =========================================================
# GEMINI REQUEST
# =========================================================

def call_gemini(
    model_name,
    prompt,
    image_1h,
    image_5m,
    image_1m,
):

    img_1h = bytes_to_image(
        image_1h
    )

    img_5m = bytes_to_image(
        image_5m
    )

    img_1m = bytes_to_image(
        image_1m
    )

    response = (
        gemini_client
        .models
        .generate_content(
            model=model_name,
            contents=[
                prompt,
                "IMAGE 1 — 1H WEEK",
                img_1h,
                "IMAGE 2 — 5M LAST 24 HOURS",
                img_5m,
                "IMAGE 3 — 1M LAST 4 HOURS",
                img_1m,
            ],
        )
    )

    return response


# =========================================================
# GEMINI ANALYSIS
# =========================================================

def analyze_with_gemini(
    price,
    image_1h,
    image_5m,
    image_1m,
    ohlc_1h,
    ohlc_5m,
    ohlc_1m,
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
        ohlc_1h=ohlc_1h,
        ohlc_5m=ohlc_5m,
        ohlc_1m=ohlc_1m,
    )

    for model_name in gemini_models:

        try:

            print(
                f"Trying Gemini model: "
                f"{model_name}"
            )

            response = call_gemini(
                model_name,
                prompt,
                image_1h,
                image_5m,
                image_1m,
            )

            text = getattr(
                response,
                "text",
                None,
            )

            if text:

                print(
                    f"Gemini success: "
                    f"{model_name}"
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
# NUMBER EXTRACTION
# =========================================================

def extract_number(
    text,
    labels,
):

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

            return float(
                match.group(1)
            )

    return None


# =========================================================
# TRADE PARSER
# =========================================================

def parse_trade(text):

    if not text:
        return None

    upper = text.upper()

    if "NO TRADE" in upper:
        return None

    direction = None

    buy_match = re.search(
        r"DIRECTION\s*[:=]\s*BUY",
        upper,
    )

    sell_match = re.search(
        r"DIRECTION\s*[:=]\s*SELL",
        upper,
    )

    if buy_match:

        direction = "BUY"

    elif sell_match:

        direction = "SELL"

    else:

        if re.search(
            r"\bBUY\b",
            upper,
        ):

            direction = "BUY"

        elif re.search(
            r"\bSELL\b",
            upper,
        ):

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
            "TAKE PROFIT 1",
            "TAKE PROFIT",
            "TP",
        ],
    )

    if (
        entry is None
        or sl is None
        or tp1 is None
    ):

        return None

    # Validate BUY ordering.
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

    # Validate SELL ordering.
    if direction == "SELL":

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
# TELEGRAM SEND
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
    trade,
):

    entry = trade["entry"]

    # Simulation entry tolerance.
    tolerance = 0.60

    # Wait for maximum 5 minutes.
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
                "Waiting entry | "
                f"Price={price:.2f} | "
                f"Entry={entry:.2f} | "
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
                "Trade monitor | "
                f"{direction} | "
                f"Price={price:.2f} | "
                f"SL={sl:.2f} | "
                f"TP={tp1:.2f}"
            )

            if direction == "BUY":

                if price <= sl:

                    return (
                        "LOSS",
                        price,
                    )

                if price >= tp1:

                    return (
                        "WIN",
                        price,
                    )

            elif direction == "SELL":

                if price >= sl:

                    return (
                        "LOSS",
                        price,
                    )

                if price <= tp1:

                    return (
                        "WIN",
                        price,
                    )

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
        f"Started RustyGold "
        f"for chat {chat_id}"
    )

    while True:

        try:

            # =================================================
            # FETCH MARKET DATA
            # =================================================

            # 1H:
            # 168 candles ≈ 7 days.
            df_1h = await asyncio.to_thread(
                fetch_candles,
                "1h",
                168,
            )

            # 5M:
            # 288 candles ≈ 24 hours.
            df_5m = await asyncio.to_thread(
                fetch_candles,
                "5min",
                288,
            )

            # 1M:
            # 720 candles ≈ 12 hours.
            df_1m = await asyncio.to_thread(
                fetch_candles,
                "1min",
                720,
            )

            # Current live gold price.
            price = await asyncio.to_thread(
                fetch_gold_price
            )

            # =================================================
            # CREATE CHARTS
            # =================================================

            # 1H:
            # Full week.
            image_1h = await asyncio.to_thread(
                create_chart,
                df_1h,
                "XAU/USD - 1H - 1 Week",
                168,
            )

            # 5M:
            # Full 24 hours.
            image_5m = await asyncio.to_thread(
                create_chart,
                df_5m,
                "XAU/USD - 5M - 24 Hours",
                288,
            )

            # 1M:
            # Only latest 4 hours for visual clarity.
            image_1m = await asyncio.to_thread(
                create_chart,
                df_1m,
                "XAU/USD - 1M - Last 4 Hours",
                240,
            )

            # =================================================
            # PREPARE OHLC DATA
            # =================================================

            # 1H:
            # Full week.
            ohlc_1h = (
                dataframe_to_ohlc_text(
                    df_1h,
                    168,
                )
            )

            # 5M:
            # Full 24 hours.
            ohlc_5m = (
                dataframe_to_ohlc_text(
                    df_5m,
                    288,
                )
            )

            # 1M:
            # We collected 12 hours,
            # but send only the latest 180 candles
            # to keep the prompt focused.
            ohlc_1m = (
                dataframe_to_ohlc_text(
                    df_1m,
                    180,
                )
            )

            # =================================================
            # GEMINI
            # =================================================

            result = await asyncio.to_thread(
                analyze_with_gemini,
                price,
                image_1h,
                image_5m,
                image_1m,
                ohlc_1h,
                ohlc_5m,
                ohlc_1m,
            )

            # =================================================
            # PARSE
            # =================================================

            trade = parse_trade(
                result
            )

            # =================================================
            # NO TRADE
            # =================================================

            if trade is None:

                await send_message(
                    context,
                    chat_id,
                    (
                        "🔎 RustyGold\n\n"
                        f"Current price: "
                        f"{price:.2f}\n\n"
                        f"{result}"
                    ),
                )

                # Recheck after 5 minutes.
                await asyncio.sleep(300)

                continue

            # =================================================
            # TRADE FOUND
            # =================================================

            direction = trade[
                "direction"
            ]

            entry = trade[
                "entry"
            ]

            sl = trade[
                "sl"
            ]

            tp1 = trade[
                "tp1"
            ]

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

            # =================================================
            # WAIT FOR ENTRY
            # =================================================

            entry_price = (
                await wait_for_entry(
                    trade
                )
            )

            if entry_price is None:

                await send_message(
                    context,
                    chat_id,
                    (
                        "⌛ Entry was not reached "
                        "within 5 minutes.\n\n"
                        "Setup cancelled."
                    ),
                )

                continue

            # =================================================
            # SIMULATED ENTRY
            # =================================================

            stats["trades"] += 1

            await send_message(
                context,
                chat_id,
                (
                    "🟢 SIMULATED ENTRY\n\n"
                    f"Direction: {direction}\n"
                    f"Entry: {entry:.2f}\n"
                    f"Current: "
                    f"{entry_price:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}"
                ),
            )

            # =================================================
            # MONITOR
            # =================================================

            result_type, exit_price = (
                await monitor_trade(
                    trade
                )
            )

            # =================================================
            # WIN
            # =================================================

            if result_type == "WIN":

                stats["wins"] += 1

                if direction == "BUY":

                    pnl = (
                        exit_price
                        - entry
                    )

                else:

                    pnl = (
                        entry
                        - exit_price
                    )

                stats["pnl"] += pnl

                await send_message(
                    context,
                    chat_id,
                    (
                        "✅ TP1 HIT\n\n"
                        f"Exit: "
                        f"{exit_price:.2f}\n"
                        f"Simulated PnL: "
                        f"{pnl:+.2f}"
                    ),
                )

            # =================================================
            # LOSS
            # =================================================

            else:

                stats["losses"] += 1

                if direction == "BUY":

                    pnl = (
                        exit_price
                        - entry
                    )

                else:

                    pnl = (
                        entry
                        - exit_price
                    )

                stats["pnl"] += pnl

                await send_message(
                    context,
                    chat_id,
                    (
                        "❌ STOP LOSS HIT\n\n"
                        f"Exit: "
                        f"{exit_price:.2f}\n"
                        f"Simulated PnL: "
                        f"{pnl:+.2f}"
                    ),
                )

            # Small delay before next analysis.
            await asyncio.sleep(5)

        except asyncio.CancelledError:

            print(
                f"RustyGold task "
                f"cancelled: {chat_id}"
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

            # Prevent rapid repeated errors.
            await asyncio.sleep(30)


# =========================================================
# START COMMAND
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

    # =====================================================
    # START
    # =====================================================

    if query.data == "start_analysis":

        existing = tasks.get(
            chat_id
        )

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

    # =====================================================
    # STOP
    # =====================================================

    elif query.data == "stop_analysis":

        task = tasks.get(
            chat_id
        )

        if task and not task.done():

            task.cancel()

            # Remove immediately.
            tasks.pop(
                chat_id,
                None,
            )

            await query.message.reply_text(
                "🛑 تم إيقاف التحليل."
            )

        else:

            await query.message.reply_text(
                "لا يوجد تحليل يعمل حالياً."
            )

    # =====================================================
    # SUMMARY
    # =====================================================

    elif query.data == "summary":

        total = stats[
            "trades"
        ]

        wins = stats[
            "wins"
        ]

        losses = stats[
            "losses"
        ]

        pnl = stats[
            "pnl"
        ]

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
                f"Win rate: "
                f"{winrate:.1f}%\n"
                f"Simulated PnL: "
                f"{pnl:+.2f}"
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

    # Keep Render web service alive.
    threading.Thread(
        target=run_flask,
        daemon=True,
    ).start()

    application = (
        Application
        .builder()
        .token(
            TELEGRAM_TOKEN
        )
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




