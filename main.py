import os
from flask import Flask
app = Flask(__name__)
@app.route('/')
def home(): return "OK - Bot Live"

# باقي الكود
import requests, asyncio, io, re
import pandas as pd
import mplfinance as mpf
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

BOT_TOKEN = os.getenv("TG_TOKEN","").strip()
GEMINI_KEY = os.getenv("GEMINI_API_KEY","").strip()
TWELVE_KEY = os.getenv("TWELVE_API_KEY","").strip()

print(f"Token exists: {bool(BOT_TOKEN)} - len {len(BOT_TOKEN)}")

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
active = {}
stats = {}

def get_twelvedata(interval, outputsize):
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={outputsize}&apikey={TWELVE_KEY}"
    r = requests.get(url, timeout=15).json()
    if 'values' not in r: raise Exception(str(r))
    df = pd.DataFrame(r['values'])
    df = df.rename(columns={'datetime':'time','open':'open','high':'high','low':'low','close':'close'})
    df[['open','high','low','close']] = df[['open','high','low','close']].astype(float)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    df.set_index('time', inplace=True)
    return df

def make_chart(df, title):
    buf = io.BytesIO()
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', wick={'up':'#26a69a','down':'#ef5350'})
    s = mpf.make_mpf_style(marketcolors=mc, base_mpl_style='yahoo', gridstyle='--', y_on_right=True)
    last = df['close'].iloc[-1]
    mpf.plot(df, type='candle', style=s, figratio=(16,9), figscale=1.3, title=f"{title} {last:.2f}", ylabel='Price', savefig=dict(fname=buf, dpi=150, bbox_inches='tight'))
    buf.seek(0)
    return buf

def get_gold_price():
    return float(requests.get("https://api.gold-api.com/price/XAU/USD", timeout=10).json()['price'])

def parse_numbers(text):
    try:
        entry = float(re.search(r'ENTRY[:\s]*([0-9.]+)', text, re.I).group(1))
        sl = float(re.search(r'SL[:\s]*([0-9.]+)', text, re.I).group(1))
        tp1 = float(re.search(r'TP1?[:\s]*([0-9.]+)', text, re.I).group(1))
        return entry, sl, tp1
    except: return None

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in stats: stats[chat_id] = {'wins':0,'loss':0,'pnl':0.0}
    kb = [[InlineKeyboardButton("🚀 حلل يا جيمني", callback_data='analyze')], [InlineKeyboardButton("🛑 إيقاف", callback_data='stop')]]
    await update.message.reply_text("RustyGold جاهز 🥇\nبشوف الشارت كصورة", reply_markup=InlineKeyboardMarkup(kb))

async def handle_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    if chat_id not in stats: stats[chat_id] = {'wins':0,'loss':0,'pnl':0.0}
    if query.data == 'stop':
        active[chat_id] = False
        await query.message.reply_text("✅ تم الإيقاف")
        return
    if query.data == 'analyze':
        if active.get(chat_id):
            await query.message.reply_text("شغال...")
            return
        active[chat_id] = True
        await query.message.reply_text("⏳ بحلل الصور...")
        while active.get(chat_id):
            try:
                df_1m = get_twelvedata("1min", 60)
                df_5m = get_twelvedata("5min", 96)
                img_1m = make_chart(df_1m, "XAUUSD 1M")
                img_5m = make_chart(df_5m, "XAUUSD 5M")
                real = get_gold_price()
                prompt = f"خبير ICT. السعر الحقيقي={real}. صورة1=1دقيقة صورة2=5دقائق. هل يوجد BOS/CHoCH+sweep+FVG؟ اذا نعم اكتب ENTRY: {real} SL: (0.3%) TP1: (1:1.5) وشرح سطر. اذا لا قل لا تتداول"
                res = model.generate_content([prompt, {'mime_type':'image/png','data': img_1m.getvalue()}, {'mime_type':'image/png','data': img_5m.getvalue()}])
                txt = res.text
                await context.bot.send_message(chat_id, f"{txt}\nسعر: {real}")
                if "لا تتداول" in txt:
                    await asyncio.sleep(300)
                    continue
                parsed = parse_numbers(txt)
                if not parsed:
                    await asyncio.sleep(300)
                    continue
                entry, sl, tp1 = parsed
                is_buy = entry > sl
                await context.bot.send_message(chat_id, f"👀 براقب {entry}")
                entered=False
                for _ in range(60):
                    if not active.get(chat_id): break
                    cur = get_gold_price()
                    if abs(cur-entry)<0.6:
                        entered=True
                        await context.bot.send_message(chat_id, f"✅ دخول {cur}")
                        break
                    await asyncio.sleep(5)
                if not entered:
                    await asyncio.sleep(5)
                    continue
                while active.get(chat_id):
                    cur = get_gold_price()
                    hit_sl = (cur <= sl) if is_buy else (cur >= sl)
                    hit_tp = (cur >= tp1) if is_buy else (cur <= tp1)
                    if hit_sl or hit_tp:
                        await context.bot.send_message(chat_id, f"{'❌ SL' if hit_sl else '✅ TP'} {cur}")
                        break
                    await asyncio.sleep(5)
                await asyncio.sleep(60)
            except Exception as e:
                await context.bot.send_message(chat_id, f"⚠️ {e}")
                await asyncio.sleep(30)

def run_bot():
    print("Starting bot polling...")
    app_bot = Application.builder().token(BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CallbackQueryHandler(handle_btn))
    app_bot.run_polling()

if __name__ == "__main__":
    import threading
    port = int(os.environ.get("PORT", 10000))
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=port), daemon=True).start()
    run_bot()
