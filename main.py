from flask import Flask
app = Flask(__name__)
@app.route('/')
def home(): return "RustyGold Bot Running - OK"

import os, requests, asyncio, io, re
import pandas as pd
import mplfinance as mpf
import google.generativeai as genai
from datetime import datetime
import pytz
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

BOT_TOKEN = os.getenv("TG_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
TWELVE_KEY = os.getenv("TWELVE_API_KEY")

genai.configure(api_key=GEMINI_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

active = {}
stats = {}

def get_twelvedata(interval, outputsize):
    url = f"https://api.twelvedata.com/time_series?symbol=XAU/USD&interval={interval}&outputsize={outputsize}&apikey={TWELVE_KEY}"
    r = requests.get(url, timeout=15).json()
    if 'values' not in r: raise Exception(r.get('message', str(r)))
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
    mpf.plot(df, type='candle', style=s, figratio=(16,9), figscale=1.3, title=f"{title} - {last:.2f}", ylabel='Price', savefig=dict(fname=buf, dpi=150, bbox_inches='tight'))
    buf.seek(0)
    return buf

def get_gold_price():
    r = requests.get("https://api.gold-api.com/price/XAU/USD", timeout=10).json()
    return float(r['price'])

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
    kb = [[InlineKeyboardButton("🚀 حلل يا جيمني", callback_data='analyze')], [InlineKeyboardButton("🛑 إيقاف", callback_data='stop')], [InlineKeyboardButton("📊 ملخص", callback_data='summary')]]
    await update.message.reply_text("RustyGold جاهز 🥇", reply_markup=InlineKeyboardMarkup(kb))

async def handle_btn(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat_id
    if chat_id not in stats: stats[chat_id] = {'wins':0,'loss':0,'pnl':0.0}
    if query.data == 'stop':
        active[chat_id] = False
        await query.message.reply_text("✅ تم الإيقاف.")
        return
    if query.data == 'summary':
        s = stats[chat_id]
        await query.message.reply_text(f"📊 ملخص:\nربح: {s['wins']} | خسارة: {s['loss']}\nPnL: {s['pnl']:.2f}%")
        return
    if query.data == 'analyze':
        if active.get(chat_id):
            await query.message.reply_text("شغال أصلاً...")
            return
        active[chat_id] = True
        await query.message.reply_text("⏳ بدأ التحليل التلقائي...")
        while active.get(chat_id):
            try:
                df_1m = get_twelvedata("1min", 60)
                df_5m = get_twelvedata("5min", 96)
                img_1m = make_chart(df_1m, "XAUUSD 1M")
                img_5m = make_chart(df_5m, "XAUUSD 5M")
                real = get_gold_price()
                prompt = f"أنت خبير ICT سكالبينغ، انظر للصور فقط. السعر الحقيقي الآن = {real} الصورة 1 = 1د آخر ساعة، الصورة 2 = 5د آخر 8 ساعات. هل يوجد BOS/CHoCH على 1m؟ هل يوجد Sweep لآخر قمة/قاع + FVG+OB واضح؟ إذا نعم اكتب بالضبط: ENTRY: {real} SL: (0.3% تحت/فوق) TP1: (1:1.5) TP2: سيولة وتحليل سطرين فقط. إذا لا يوجد قل فقط: لا تتداول"
                res = model.generate_content([prompt, {'mime_type':'image/png','data': img_1m.getvalue()}, {'mime_type':'image/png','data': img_5m.getvalue()}])
                txt = res.text
                await context.bot.send_message(chat_id, f"📊 Gemini:\n{txt}\nسعر حقيقي: {real}")
                if "لا تتداول" in txt:
                    await context.bot.send_message(chat_id, "⏱️ مافي فرصة، بعد 5د...")
                    img_1m.close(); img_5m.close()
                    await asyncio.sleep(300)
                    continue
                parsed = parse_numbers(txt)
                if not parsed:
                    await context.bot.send_message(chat_id, "⚠️ ما قدرت أقرأ الأرقام، بعد 5د...")
                    await asyncio.sleep(300)
                    continue
                entry, sl, tp1 = parsed
                is_buy = entry > sl
                await context.bot.send_message(chat_id, f"👀 براقب الدخول {entry} لمدة 5 دقائق...")
                entered = False
                for _ in range(60):
                    if not active.get(chat_id): break
                    cur = get_gold_price()
                    if abs(cur - entry) < 0.6:
                        entered = True
                        await context.bot.send_message(chat_id, f"✅ تم الدخول! {cur}")
                        break
                    await asyncio.sleep(5)
                if not entered:
                    await context.bot.send_message(chat_id, "⏱️ لم يصل، تحليل جديد...")
                    img_1m.close(); img_5m.close()
                    await asyncio.sleep(5)
                    continue
                await context.bot.send_message(chat_id, f"🔄 مراقبة... SL {sl} TP {tp1}")
                while active.get(chat_id):
                    cur = get_gold_price()
                    pnl_pct = (cur - entry)/entry*100 if is_buy else (entry - cur)/entry*100
                    hit_sl = (cur <= sl) if is_buy else (cur >= sl)
                    hit_tp = (cur >= tp1) if is_buy else (cur <= tp1)
                    if hit_sl:
                        stats[chat_id]['loss']+=1
                        stats[chat_id]['pnl']+=pnl_pct
                        await context.bot.send_message(chat_id, f"❌ ضرب SL {cur} {pnl_pct:.2f}%")
                        break
                    if hit_tp:
                        stats[chat_id]['wins']+=1
                        stats[chat_id]['pnl']+=pnl_pct
                        await context.bot.send_message(chat_id, f"✅ ضرب TP {cur} {pnl_pct:.2f}%")
                        break
                    await asyncio.sleep(5)
                img_1m.close(); img_5m.close()
                await asyncio.sleep(60)
            except Exception as e:
                await context.bot.send_message(chat_id, f"⚠️ خطأ: {e}")
                await asyncio.sleep(30)

def run_bot():
    app_bot = Application.builder().token(BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(CallbackQueryHandler(handle_btn))
    app_bot.run_polling()

if __name__ == "__main__":
    import threading
    threading.Thread(target=lambda: app.run(host='0.0.0.0', port=10000)).start()
    run_bot()
