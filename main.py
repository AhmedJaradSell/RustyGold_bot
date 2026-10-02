import os
import io
import re
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
import mplfinance as mpf
from flask import Flask
from google import genai

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


# ============================================================
# RENDER WEB SERVER
# ============================================================

web_app = Flask(__name__)


@web_app.route("/")
def home():
    return "OK - RustyGold Bot Live"


def run_web_server():
    port = int(os.environ.get("PORT", "10000"))

    web_app.run(
        host="0.0.0.0",
        port=port,
    )


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

BOT_TOKEN = os.getenv("TG_TOKEN", "").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "").strip()
TWELVE_KEY = os.getenv("TWELVE_API_KEY", "").strip()


if not BOT_TOKEN:
    raise RuntimeError("Missing TG_TOKEN")

if not GEMINI_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY")

if not TWELVE_KEY:
    raise RuntimeError("Missing TWELVE_API_KEY")


# ============================================================
# GEMINI CLIENT
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_KEY
)

AVAILABLE_GEMINI_MODELS = []


def get_available_gemini_models():
    """
    Get Gemini models available to this API key
    that support generateContent.
    """

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

        actions = (
            model.supported_actions
            or []
        )

        if not name:
            continue

        if "generateContent" not in actions:
            continue

        if "gemini" not in name.lower():
            continue

        available.append(name)

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

            print(
                "No Gemini models available."
            )

        else:

            for i, name in enumerate(
                AVAILABLE_GEMINI_MODELS,
                1,
            ):

                print(
                    f"{i}. {name}"
                )

        print(
            "===================================\n"
        )

        return AVAILABLE_GEMINI_MODELS

    except Exception as e:

        print(
            "Gemini model listing failed:",
            type(e).__name__,
            e,
        )

        AVAILABLE_GEMINI_MODELS = []

        return []


# ============================================================
# GLOBAL STATE
# ============================================================

tasks = {}

stats = {}

executor = ThreadPoolExecutor(
    max_workers=4
)


# ============================================================
# STATS
# ============================================================

def ensure_stats(chat_id):

    if chat_id not in stats:

        stats[chat_id] = {
            "wins": 0,
            "loss": 0,
            "pnl": 0.0,
            "trades": 0,
        }


def get_stats_text(chat_id):

    ensure_stats(chat_id)

    s = stats[chat_id]

    return (
        "📊 RustyGold — الملخص\n\n"
        f"✅ أرباح: {s['wins']}\n"
        f"❌ خسائر: {s['loss']}\n"
        f"📈 إجمالي الصفقات: {s['trades']}\n"
        f"💰 النتيجة: {s['pnl']:+.2f}"
    )


def is_running(chat_id):

    task = tasks.get(chat_id)

    return (
        task is not None
        and not task.done()
    )


# ============================================================
# TWELVE DATA
# ============================================================

def get_twelvedata(
    interval,
    outputsize,
):

    url = (
        "https://api.twelvedata.com/"
        "time_series"
    )

    params = {
        "symbol": "XAU/USD",
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVE_KEY,
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
            "TwelveData error: "
            + str(
                data.get(
                    "message",
                    data,
                )
            )
        )

    df = pd.DataFrame(
        data["values"]
    )

    required = [
        "datetime",
        "open",
        "high",
        "low",
        "close",
    ]

    for column in required:

        if column not in df.columns:

            raise RuntimeError(
                f"TwelveData missing column: "
                f"{column}"
            )

    df = df[required].copy()

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

    df["datetime"] = pd.to_datetime(
        df["datetime"],
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


# ============================================================
# CHART
# ============================================================

def make_chart(
    df,
    title,
):

    buffer = io.BytesIO()

    colors = mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        wick={
            "up": "#26a69a",
            "down": "#ef5350",
        },
        edge={
            "up": "#26a69a",
            "down": "#ef5350",
        },
    )

    style = mpf.make_mpf_style(
        marketcolors=colors,
        base_mpl_style="default",
        gridstyle="--",
        y_on_right=True,
    )

    last_price = float(
        df["close"].iloc[-1]
    )

    mpf.plot(
        df,
        type="candle",
        style=style,
        figratio=(16, 9),
        figscale=1.2,
        title=(
            f"{title} | "
            f"{last_price:.2f}"
        ),
        ylabel="Price",
        savefig=dict(
            fname=buffer,
            dpi=130,
            bbox_inches="tight",
        ),
    )

    buffer.seek(0)

    return buffer.getvalue()


# ============================================================
# GOLD PRICE
# ============================================================

def get_gold_price():

    response = requests.get(
        "https://api.gold-api.com/"
        "price/XAU/USD",
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()

    if "price" not in data:

        raise RuntimeError(
            f"Gold API error: {data}"
        )

    return float(
        data["price"]
    )


# ============================================================
# ICT PROMPT
# ============================================================

ICT_PROMPT = """
أنت محلل تداول متخصص في ICT وPrice Action.

لديك صورتان لشارت XAU/USD:

1) فريم 1 دقيقة
2) فريم 5 دقائق

ولديك السعر الحقيقي الحالي.

حلل الصور بصريًا.

ابحث تحديدًا عن:

- Market Structure
- BOS
- CHoCH
- Liquidity Sweep
- FVG
- Order Block
- Displacement
- اتجاه الحركة على الفريمين
- منطقة دخول منطقية

قواعد مهمة:

إذا لم توجد فرصة واضحة، أجب:

NO TRADE

إذا وجدت فرصة واضحة، ابدأ بهذا الشكل:

TRADE
DIRECTION: BUY أو SELL
ENTRY: رقم
SL: رقم
TP1: رقم

ثم اكتب شرحًا مختصرًا جدًا يوضح:

- BOS أو CHoCH
- Sweep
- FVG
- OB
- سبب الدخول

لا تخترع صفقة فقط لإعطاء نتيجة.

السعر الحقيقي الحالي:

{price}
"""


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_with_gemini(
    price,
    image_1m,
    image_5m,
):

    prompt = ICT_PROMPT.format(
        price=f"{price:.2f}"
    )

    global AVAILABLE_GEMINI_MODELS

    if not AVAILABLE_GEMINI_MODELS:

        load_gemini_models()

    if not AVAILABLE_GEMINI_MODELS:

        raise RuntimeError(
            "No Gemini models supporting "
            "generateContent are available."
        )

    last_error = None

    for model_name in (
        AVAILABLE_GEMINI_MODELS
    ):

        print(
            f"Trying Gemini model: "
            f"{model_name}"
        )

        try:

            response = (
                gemini_client.models
                .generate_content(
                    model=model_name,
                    contents=[
                        prompt,
                        {
                            "inline_data": {
                                "mime_type":
                                    "image/png",
                                "data":
                                    image_1m,
                            }
                        },
                        {
                            "inline_data": {
                                "mime_type":
                                    "image/png",
                                "data":
                                    image_5m,
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
                    f"SUCCESS: "
                    f"{model_name}"
                )

                return (
                    text.strip(),
                    model_name,
                )

        except Exception as e:

            last_error = e

            print(
                f"FAILED: "
                f"{model_name} -> "
                f"{type(e).__name__}: "
                f"{e}"
            )

            continue

    raise RuntimeError(
        "All Gemini models failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# PARSE GEMINI TRADE
# ============================================================

def extract_number(
    pattern,
    text,
):

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

    except ValueError:

        return None


def parse_trade(text):

    if not text:
        return None

    if "NO TRADE" in text.upper():
        return None

    entry = extract_number(
        r"ENTRY\s*[:=]\s*\$?"
        r"\s*([0-9]+(?:\.[0-9]+)?)",
        text,
    )

    sl = extract_number(
        r"SL\s*[:=]\s*\$?"
        r"\s*([0-9]+(?:\.[0-9]+)?)",
        text,
    )

    tp1 = extract_number(
        r"TP1\s*[:=]\s*\$?"
        r"\s*([0-9]+(?:\.[0-9]+)?)",
        text,
    )

    direction_match = re.search(
        r"DIRECTION\s*[:=]\s*"
        r"(BUY|SELL)",
        text,
        re.IGNORECASE,
    )

    if (
        entry is None
        or sl is None
        or tp1 is None
    ):

        return None

    if direction_match:

        direction = (
            direction_match
            .group(1)
            .upper()
        )

    else:

        direction = (
            "BUY"
            if tp1 > entry
            else "SELL"
        )

    if direction == "BUY":

        if not (
            sl < entry < tp1
        ):

            return None

    else:

        if not (
            tp1 < entry < sl
        ):

            return None

    return {
        "direction": direction,
        "entry": entry,
        "sl": sl,
        "tp1": tp1,
    }


# ============================================================
# MARKET DATA
# ============================================================

def collect_market_data():

    df_1m = get_twelvedata(
        "1min",
        60,
    )

    df_5m = get_twelvedata(
        "5min",
        96,
    )

    image_1m = make_chart(
        df_1m,
        "XAU/USD 1M",
    )

    image_5m = make_chart(
        df_5m,
        "XAU/USD 5M",
    )

    price = get_gold_price()

    return (
        image_1m,
        image_5m,
        price,
    )


# ============================================================
# BLOCKING -> ASYNC
# ============================================================

async def blocking(
    func,
    *args,
):

    loop = (
        asyncio
        .get_running_loop()
    )

    return await loop.run_in_executor(
        executor,
        lambda: func(*args),
    )


# ============================================================
# SLEEP WITH STOP CHECK
# ============================================================

async def sleep_or_stop(
    chat_id,
    seconds,
):

    for _ in range(seconds):

        if not is_running(chat_id):
            return False

        await asyncio.sleep(1)

    return is_running(chat_id)


# ============================================================
# WAIT FOR ENTRY
# ============================================================

async def wait_for_entry(
    chat_id,
    entry,
):

    # 5 minutes
    for _ in range(60):

        if not is_running(chat_id):
            return None

        try:

            current = await blocking(
                get_gold_price
            )

            if (
                abs(
                    current - entry
                )
                <= 0.60
            ):

                return current

        except Exception as e:

            print(
                "Entry price error:",
                e,
            )

        if not await sleep_or_stop(
            chat_id,
            5,
        ):

            return None

    return None


# ============================================================
# MONITOR TRADE
# ============================================================

async def monitor_trade(
    chat_id,
    trade,
):

    entry = trade["entry"]
    sl = trade["sl"]
    tp1 = trade["tp1"]
    direction = trade["direction"]

    while is_running(chat_id):

        try:

            current = await blocking(
                get_gold_price
            )

            if direction == "BUY":

                if current <= sl:
                    return (
                        "SL",
                        current,
                    )

                if current >= tp1:
                    return (
                        "TP",
                        current,
                    )

            else:

                if current >= sl:
                    return (
                        "SL",
                        current,
                    )

                if current <= tp1:
                    return (
                        "TP",
                        current,
                    )

        except Exception as e:

            print(
                "Trade monitor error:",
                e,
            )

        if not await sleep_or_stop(
            chat_id,
            5,
        ):

            return (
                None,
                None,
            )

    return (
        None,
        None,
    )


# ============================================================
# ANALYSIS LOOP
# ============================================================

async def analysis_loop(
    application,
    chat_id,
):

    ensure_stats(chat_id)

    try:

        while is_running(chat_id):

            try:

                # --------------------------------------------
                # MARKET DATA
                # --------------------------------------------

                await application.bot.send_message(
                    chat_id,
                    "⏳ بجمع شارت 1M و5M "
                    "والسعر الحقيقي..."
                )

                (
                    image_1m,
                    image_5m,
                    current_price,
                ) = await blocking(
                    collect_market_data
                )

                # --------------------------------------------
                # GEMINI
                # --------------------------------------------

                await application.bot.send_message(
                    chat_id,
                    "🤖 Gemini يحلل...\n"
                    f"السعر الحالي: "
                    f"{current_price:.2f}"
                )

                (
                    analysis,
                    used_model,
                ) = await blocking(
                    analyze_with_gemini,
                    current_price,
                    image_1m,
                    image_5m,
                )

                await application.bot.send_message(
                    chat_id,
                    "🤖 موديل Gemini المستخدم:\n"
                    f"{used_model}\n\n"
                    f"📋 التحليل:\n"
                    f"{analysis}"
                )

                # --------------------------------------------
                # PARSE
                # --------------------------------------------

                trade = parse_trade(
                    analysis
                )

                if not trade:

                    await application.bot.send_message(
                        chat_id,
                        "⏸️ لا توجد فرصة واضحة.\n"
                        "سأعيد التحليل بعد 5 دقائق."
                    )

                    if not await sleep_or_stop(
                        chat_id,
                        300,
                    ):

                        break

                    continue

                entry = trade["entry"]
                sl = trade["sl"]
                tp1 = trade["tp1"]
                direction = trade["direction"]

                await application.bot.send_message(
                    chat_id,
                    "🎯 فرصة مكتشفة\n\n"
                    f"الاتجاه: {direction}\n"
                    f"ENTRY: {entry:.2f}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}\n\n"
                    "👀 أراقب ENTRY "
                    "لمدة 5 دقائق..."
                )

                # --------------------------------------------
                # WAIT ENTRY
                # --------------------------------------------

                entry_price = (
                    await wait_for_entry(
                        chat_id,
                        entry,
                    )
                )

                if not is_running(chat_id):
                    break

                if entry_price is None:

                    await application.bot.send_message(
                        chat_id,
                        "⌛ لم يصل السعر إلى "
                        "ENTRY خلال 5 دقائق.\n"
                        "❌ ألغيت الفرصة."
                    )

                    continue

                # --------------------------------------------
                # ENTER
                # --------------------------------------------

                stats[chat_id]["trades"] += 1

                await application.bot.send_message(
                    chat_id,
                    "🟢 دخول افتراضي\n\n"
                    f"السعر: "
                    f"{entry_price:.2f}\n"
                    f"الاتجاه: {direction}\n"
                    f"SL: {sl:.2f}\n"
                    f"TP1: {tp1:.2f}\n\n"
                    "📡 أراقب الصفقة كل 5 ثواني..."
                )

                # --------------------------------------------
                # MONITOR
                # --------------------------------------------

                result, exit_price = (
                    await monitor_trade(
                        chat_id,
                        trade,
                    )
                )

                if not is_running(chat_id):
                    break

                if result == "SL":

                    stats[chat_id]["loss"] += 1

                    if direction == "BUY":

                        pnl = (
                            exit_price
                            - entry_price
                        )

                    else:

                        pnl = (
                            entry_price
                            - exit_price
                        )

                    stats[chat_id]["pnl"] += pnl

                    await application.bot.send_message(
                        chat_id,
                        "❌ ضرب SL\n\n"
                        f"الدخول: "
                        f"{entry_price:.2f}\n"
                        f"الخروج: "
                        f"{exit_price:.2f}\n"
                        f"النتيجة التقريبية: "
                        f"{pnl:+.2f}"
                    )

                elif result == "TP":

                    stats[chat_id]["wins"] += 1

                    if direction == "BUY":

                        pnl = (
                            exit_price
                            - entry_price
                        )

                    else:

                        pnl = (
                            entry_price
                            - exit_price
                        )

                    stats[chat_id]["pnl"] += pnl

                    await application.bot.send_message(
                        chat_id,
                        "✅ ضرب TP1\n\n"
                        f"الدخول: "
                        f"{entry_price:.2f}\n"
                        f"الخروج: "
                        f"{exit_price:.2f}\n"
                        f"النتيجة التقريبية: "
                        f"{pnl:+.2f}"
                    )

                # --------------------------------------------
                # NEXT ANALYSIS
                # --------------------------------------------

                if is_running(chat_id):

                    await application.bot.send_message(
                        chat_id,
                        "🔄 انتهت الصفقة.\n"
                        "سأعيد التحليل..."
                    )

                    if not await sleep_or_stop(
                        chat_id,
                        60,
                    ):

                        break

            except asyncio.CancelledError:

                raise

            except Exception as e:

                print(
                    f"[ERROR] "
                    f"chat={chat_id}: "
                    f"{type(e).__name__}: "
                    f"{e}"
                )

                try:

                    await application.bot.send_message(
                        chat_id,
                        "⚠️ حدث خطأ مؤقت:\n"
                        f"{type(e).__name__}: "
                        f"{e}\n\n"
                        "سأحاول مرة أخرى بعد "
                        "30 ثانية."
                    )

                except Exception:
                    pass

                if not await sleep_or_stop(
                    chat_id,
                    30,
                ):

                    break

    finally:

        current = (
            asyncio.current_task()
        )

        if tasks.get(chat_id) is current:

            tasks.pop(
                chat_id,
                None,
            )

        print(
            f"Analysis stopped "
            f"for chat {chat_id}"
        )


# ============================================================
# /START
# ============================================================

async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    chat_id = (
        update.effective_chat.id
    )

    ensure_stats(chat_id)

    keyboard = [

        [
            InlineKeyboardButton(
                "🚀 حلل يا جيمني",
                callback_data="analyze",
            )
        ],

        [
            InlineKeyboardButton(
                "🛑 إيقاف",
                callback_data="stop",
            ),

            InlineKeyboardButton(
                "📊 ملخص",
                callback_data="summary",
            ),
        ],
    ]

    await update.message.reply_text(
        "🥇 RustyGold جاهز\n\n"
        "اضغط «🚀 حلل يا جيمني» "
        "لبدء التحليل المستمر.",
        reply_markup=(
            InlineKeyboardMarkup(
                keyboard
            )
        ),
    )


# ============================================================
# BUTTON HANDLER
# ============================================================

async def handle_button(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
):

    query = update.callback_query

    await query.answer()

    chat_id = (
        query.message.chat_id
    )

    ensure_stats(chat_id)

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if query.data == "stop":

        task = tasks.get(chat_id)

        if task and not task.done():

            # Cancel immediately.
            # Do NOT await the task.
            task.cancel()

            # Remove immediately.
            tasks.pop(
                chat_id,
                None,
            )

            await query.message.reply_text(
                "🛑 تم إيقاف المهمة فورًا."
            )

        else:

            tasks.pop(
                chat_id,
                None,
            )

            await query.message.reply_text(
                "ℹ️ البوت غير شغال حاليًا."
            )

        return

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if query.data == "summary":

        await query.message.reply_text(
            get_stats_text(chat_id)
        )

        return

    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if query.data == "analyze":

        if is_running(chat_id):

            await query.message.reply_text(
                "🟢 التحليل شغال بالفعل."
            )

            return

        await query.message.reply_text(
            "🚀 بدأ RustyGold.\n"
            "Gemini رح يحلل الشارت باستمرار."
        )

        task = asyncio.create_task(
            analysis_loop(
                context.application,
                chat_id,
            )
        )

        tasks[chat_id] = task

        return


# ============================================================
# TELEGRAM ERROR HANDLER
# ============================================================

async def telegram_error(
    update,
    context,
):

    print(
        "[TELEGRAM ERROR]",
        repr(context.error),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "================================"
    )

    print(
        "RustyGold starting..."
    )

    print(
        f"Telegram token: "
        f"{'OK' if BOT_TOKEN else 'MISSING'}"
    )

    print(
        f"Gemini key: "
        f"{'OK' if GEMINI_KEY else 'MISSING'}"
    )

    print(
        f"TwelveData key: "
        f"{'OK' if TWELVE_KEY else 'MISSING'}"
    )

    # --------------------------------------------
    # Load available Gemini models
    # --------------------------------------------

    load_gemini_models()

    # --------------------------------------------
    # Render web server
    # --------------------------------------------

    threading.Thread(
        target=run_web_server,
        daemon=True,
    ).start()

    # --------------------------------------------
    # Telegram
    # --------------------------------------------

    telegram_app = (
        Application
        .builder()
        .token(BOT_TOKEN)
        .build()
    )

    telegram_app.add_handler(
        CommandHandler(
            "start",
            start,
        )
    )

    telegram_app.add_handler(
        CallbackQueryHandler(
            handle_button
        )
    )

    telegram_app.add_error_handler(
        telegram_error
    )

    print(
        "Telegram polling started."
    )

    telegram_app.run_polling(
        drop_pending_updates=True
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()
