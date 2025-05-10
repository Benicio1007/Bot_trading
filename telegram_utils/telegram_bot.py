from telegram import Update, ReplyKeyboardMarkup, Bot, KeyboardButton
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
from datetime import datetime
import asyncio
import threading
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes
from .pnl_telegram import *

TELEGRAM_TOKEN = "7884812992:AAGo2tfp9HVM6rFuot4NU7wPj9ikMXD3pXA"
AUTHORIZED_USER_ID = 6928389763

bot_pausado = False
bot_referencia = None  # Se establece desde realtime_bot.py
metrics = None
operations = None
capital_actual_func = None  # función para obtener el capital actual




bot = Bot(token=TELEGRAM_TOKEN)
bot_state = {
    "balance": 0,
    "pnl_total": 0,
    "operaciones_count": 0,
    "ultimo_trade": {}
}

def autorizado(user_id):
    return user_id == AUTHORIZED_USER_ID

# ✅ NUEVO COMANDO /start CON BOTONES COMPLETOS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update.effective_user.id):
        await update.message.reply_text("⛔ No estás autorizado para usar este bot.")
        return
    keyboard = [
        ["/start", "/stop"],
        ["/status", "/lasttrade"],
        ["/capital", "/stats"],
        ["/logs", "/send_sheet"],
        ["/modo_seguro", "/modo_agresivo"],
        ["📋 Comandos"]
    ]
    await update.message.reply_text(
        "🤖 *Bot iniciado.* Usá el menú para navegar los comandos disponibles.",
        parse_mode="Markdown",
        reply_markup=ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    )

# ✅ COMANDO QUE MUESTRA TODOS LOS COMANDOS DETALLADOS
async def mostrar_comandos(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update.effective_user.id): return
    msg = """
📋 *Lista de comandos disponibles*:

/start → Iniciar o reanudar el bot.
/stop → Pausar el bot de inmediato.
/stats → Muestra PNL actual, winrate, cantidad de operaciones, drawdown.
/capital → Muestra tu capital actualizado.
/logs → Muestra últimas 3 operaciones.
/modo_seguro → Activa reglas más conservadoras.
/modo_agresivo → Aumenta riesgo y frecuencia.
/reset → Reinicia el bot completamente.
/send_sheet → Enlace al Google Sheet con el log.
/status → Muestra estado general del bot.
/lasttrade → Muestra detalles de la última operación.
/help → Ayuda rápida.
"""
    await update.message.reply_markdown(msg)

# Comandos existentes
async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update.effective_user.id): return
    msg = f"""
📈 *Estado actual del bot*:
💰 Balance: ${bot_state['balance']:.2f}
📊 PnL Total: {bot_state['pnl_total']:.2f}
📄 Operaciones: {bot_state['operaciones_count']}
"""
    await update.message.reply_markdown(msg)

async def last_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update.effective_user.id): return
    if not bot_state["ultimo_trade"]:
        await update.message.reply_text("No hay operaciones registradas todavía.")
        return
    t = bot_state["ultimo_trade"]
    msg = f"""
📄 *Última operación cerrada*:
📈 Activo: {t.get('symbol')}
📉 Tipo: {t.get('side')}
📍 Entrada: {t.get('entry_price')}
🎯 TP: {t.get('tp')}
🛑 SL: {t.get('sl')}
💰 PnL: {t.get('pnl')}
⏱️ Entrada: {t.get('entry_time')}
⏱️ Salida: {t.get('exit_time')}
"""
    await update.message.reply_markdown(msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not autorizado(update.effective_user.id): return
    await mostrar_comandos(update, context)


# Notificaciones
async def notificar_html(texto_html):
    try:
        await bot.send_message(chat_id=AUTHORIZED_USER_ID, text=texto_html, parse_mode="HTML")
    except Exception as e:
        print(f"❌ Error al enviar mensaje HTML: {e}")

async def notificar_operacion_abierta(symbol, side, qty, price):
    msg = f"""

🚀 <b>OPERACIÓN ABIERTA</b>\n
🔹 <b>Activo:</b> {symbol}
📈 <b>Tipo:</b> {side.upper()}
📦 <b>Cantidad:</b> {qty}
💵 <b>Entrada:</b> {price}
⏱ <b>Fecha:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    await notificar_html(msg)

async def notificar_operacion_cerrada(operations, current, pnl, com_entrada, com_salida, res, ts_exit):
    color = 'green' if pnl > 0 else 'red'
    msg = f"""
📉 <b>OPERACIÓN CERRADA</b>\n
🔹 <b>Activo:</b> {operations['symbol']}
📈 <b>Tipo:</b> {operations['side'].upper()}
💵 <b>Entrada:</b> {operations['entry_price']}
🏁 <b>Salida:</b> {current}
📊 <b>PnL Neto:</b> <b>{pnl:.2f} USD</b>
💸 <b>Comisión total:</b> {(com_entrada + com_salida):.2f} USD
🎯 <b>Resultado:</b> {res}
⏱ <b>Duración:</b> {operations['ts_entry']} → {ts_exit}
"""
    await notificar_html(msg)


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_running
    if not autorizado(update.effective_user.id): return

    if bot_running:
        bot_running = False
        await update.message.reply_text("⛔ Bot detenido.")
    
    elif drawdown_actual >= 0.25:
        bot_running = False
        await update.message.reply_text("⛔ Bot detenido debido a que el drawdown supero el 25 porciento revisa que mierda pasa.")
    
    else:
        await update.message.reply_text("⚠️ El bot no estaba en ejecución.")


async def enviar_mensaje_inicio(chat_id):
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(
        chat_id=chat_id,
        text="🤖 ¡Bot de trading activo!\nUsa los botones para controlar:",
        reply_markup=get_keyboard()
    )

def get_keyboard():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("⏸️ Pausar", callback_data='pausar')],
        [InlineKeyboardButton("▶️ Reanudar", callback_data='reanudar')],
        [InlineKeyboardButton("📊 Ver estado", callback_data='estado')],
        [InlineKeyboardButton("📈 Trade actual", callback_data='operacion')],
        [InlineKeyboardButton("💰 Balance", callback_data='balance')],
    ])


async def handle_button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_pausado

    query = update.callback_query
    await query.answer()

    if query.data == 'pausar':
        bot_pausado = True
        if pnl >= pnl_bruto - (com_entrada + com_salida):
            await query.edit_message_text("⏸️ Bot pausado. Esperando reanudación...", reply_markup=get_keyboard())
        else:
            await query.edit_message_text("El Bot no fue pausado debido a una operacion actual negativa, espere...", reply_markup=get_keyboard())
            


    elif query.data == 'reanudar':
        bot_pausado = False
        await query.edit_message_text("✅ Bot reanudado. Operando nuevamente.", reply_markup=get_keyboard())
        await enviar_estado_en_vivo(query)

    elif query.data == 'estado':
        await enviar_estado_en_vivo(query)

    elif query.data == 'operacion':
        await enviar_operacion_actual(query)

    elif query.data == 'balance':
        balance = capital_actual_func() if capital_actual_func else 0
        await query.edit_message_text(f"💰 Balance actual: ${balance:.2f}", reply_markup=get_keyboard())

async def enviar_estado_en_vivo(query):
    if metrics:
        win_rate = (metrics['wins'] / metrics['total_trades']) * 100 if metrics['total_trades'] else 0
        pnl = metrics['pnl_sum']
        msg = (
            f"📊 *Estado actual del bot:*\n"
            f"• Trades: {metrics['total_trades']}\n"
            f"• Wins: {metrics['wins']}\n"
            f"• Losses: {metrics['losses']}\n"
            f"• Win rate: {win_rate:.2f}%\n"
            f"• PNL acumulado: ${pnl:.2f}"
        )
        await query.edit_message_text(msg, parse_mode='Markdown', reply_markup=get_keyboard())

async def enviar_operacion_actual(query):
    if operations and operations.get('open'):
        msg = (
            f"📈 *Operación actual:*\n"
            f"• Activo: {operations['symbol']}\n"
            f"• Entrada: {operations['entry_price']}\n"
            f"• Dirección: {operations['side']}\n"
            f"• Cantidad: {operations['qty']}\n"
            f"• Timestamp: {operations['ts_entry']}"
        )
    else:
        msg = "🔍 No hay operación activa en este momento."
    await query.edit_message_text(msg, parse_mode='Markdown', reply_markup=get_keyboard())

async def run_telegram_bot():

    application = Application.builder().token(TELEGRAM_TOKEN).build()
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(handle_button))

    def start_bot():
        asyncio.set_event_loop(asyncio.new_event_loop())  
        application.run_polling()
       

    thread = threading.Thread(target=start_bot, name="TelegramBot")
    thread.daemon = True
    thread.start()
    await enviar_mensaje_inicio(AUTHORIZED_USER_ID)
