import time
import pandas as pd
from datetime import datetime, timezone
from broker import PaperBroker
from strategy import MLStrategy
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv
import os
import requests
import socket
import telegram_utils.pnl_telegram as state
from telegram_utils import telegram_bot
from telegram_utils import pnl_telegram
import threading
import asyncio
import os
from informe.generar_informe import generar_y_enviar_informe
from informe.enviar_mail import enviar_email
import calendar
from sheets.sheets import upload_to_sheets
from timing.timing import Timer



# 📌 Configuración
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT']  # Lista de activos
INTERVAL = '5m'
CAPITAL_TOTAL = 1000
LEVERAGE = 30
POSITION_RATIO = 0.20
TP_PORCENTAJE = 0.004  # 0.4%
SL_PORCENTAJE = 0.002  # 0.2%
COMISION_PORCENTAJE = 0.0004
MIN_NOTIONAL = 100
MAX_CICLOS = 20
RETRAIN_INTERVAL = 10  # velas para reentrenar modelo dinámico
VOLATILITY_CICLOS = 10  # velas para reselección de activo
EXTREME_VOLATILITY_THRESHOLD = 0.05  # filter threshold
SAFE_MODE_VALIDATIONS = {'max_spread': 0.0015, 'min_volume': 10}
bot = PaperBroker()
capital_actual = bot.get_balance()
riesgo_actual = POSITION_RATIO  
riesgo_minimo = 0.10
riesgo_maximo = 0.30
drawdown_actual = (CAPITAL_TOTAL - capital_actual) / CAPITAL_TOTAL
now_utc = datetime.now(timezone.utc)
close_position = False

# 🔧 Inicialización
bot = PaperBroker()
strategy = MLStrategy(threshold=0.80)

# 📜 Crear archivo CSV si no existe
csv_file = 'sheets.operaciones.csv'
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Tipo', 'Activo', 'Entrada', 'Salida', 'Resultado', 'PNL Neto',
            'Comisión', 'Qty', 'Timestamp Entrada', 'Timestamp Salida'
        ])

# Métricas en vivo
metrics = {'total_trades': 0, 'wins': 0, 'losses': 0, 'pnl_sum': 0}

def esperar_internet(reintentos=99999, espera=10):
    for _ in range(reintentos):
        try:
            # Intenta conectar con Google DNS
            socket.create_connection(("8.8.8.8", 53), timeout=5)
            print("✅ Conexión a internet detectada.")
            return True
        except OSError:
            print("❌ Sin conexión. Reintentando en", espera, "segundos...")
            time.sleep(espera)
    print("❌ No se pudo establecer conexión tras múltiples intentos.")
    return False

def guardar_operacion(tipo, activo, entrada, salida, resultado, pnl_neto, comision, qty, ts_entrada, ts_salida):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            tipo, activo, entrada, salida, resultado, round(pnl_neto, 2),
            round(comision, 2), qty,
            ts_entrada.strftime('%Y-%m-%d %H:%M:%S'),
            ts_salida.strftime('%Y-%m-%d %H:%M:%S')
        ])


def enviar_informe_mensual():
    ruta_informe = os.path.join("informe", "informe_mensual.html")

    if not os.path.exists(ruta_informe):
        print("❌ No se encontró el informe mensual HTML.")
        return

    with open(ruta_informe, "r", encoding="utf-8") as f:
        cuerpo_html = f.read()

    fecha_actual = datetime.now().strftime("%B %Y").capitalize()
    asunto = f"📊 Informe mensual de rendimiento - {fecha_actual}"


    enviar_email(asunto, cuerpo_html)


def calcular_cantidad(symbol, usd_value):
    price = bot.fetch_ticker(symbol)['last']
    if not price or price <= 0:
        raise ValueError("Precio inválido")
    cantidad = round(usd_value / price, 3)
    return cantidad, price


def convertir_a_df(ohlcv):
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def hay_evento_economico_cercano_local(now, minutos=15):
    try:
        with open('eventos_economicos_usa_2025.csv', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                evento_dt = datetime.strptime(row['Fecha (UTC)'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                if abs((now - evento_dt).total_seconds()) <= minutos * 60:
                    print(f"🚨 Evento cercano detectado: {row['Evento']} ({evento_dt})")
                    return True
    except Exception as e:
        print(f"❌ Error al leer eventos económicos: {e}")
    return False

def enviar_informe_si_es_fin_de_mes():
    hoy = datetime.now()
    ultimo_dia = calendar.monthrange(hoy.year, hoy.month)[1]
    if hoy.day == ultimo_dia:
        print("📤 Enviando informe mensual...")
        generar_y_enviar_informe()


def initial_train():
    print("📅 Cargando datos para entrenamiento inicial...")
    combined = pd.DataFrame()
    for sym in SYMBOLS:
        raw = bot.fetch_ohlcv(sym, INTERVAL, limit=500)
        df = convertir_a_df(raw)
        df['symbol'] = sym
        combined = pd.concat([combined, df])
    strategy.train(combined)
    print("📚 Entrenamiento inicial completado.")


def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            asyncio.create_task(coro)
        else:
            loop.run_until_complete(coro)
    except RuntimeError:
        asyncio.run(coro)

initial_train()
print("🚀 Iniciando bot en paper trading con mejoras...")
operations = {'open': False, 'symbol': None, 'entry_price': None,
              'side': None, 'qty': None, 'ts_entry': None, 'candles': 0}


process_candles = 0
last_bar_time = None
volatility_candle_count = 0


esperar_internet()
timer = Timer()


telegram_bot.operations = operations
telegram_bot.metrics = metrics
telegram_bot.capital_actual_func = lambda: bot.get_balance()
asyncio.run(telegram_bot.run_telegram_bot())


while True:

    if telegram_bot.bot_pausado:
        print("⏸️ Bot pausado por Telegram...")
        while telegram_bot.bot_pausado:
            if operations['open']:
                pnl_neto = pnl_bruto - (com_entrada + com_salida)  # Esta función deberías tenerla
                
                if pnl_neto >= 0:
                    print("✅ Operación positiva detectada durante pausa. Cerrando...")
                    close_position = True
                    operacion_abierta = False
                    operacion_actual = None
                
                else:
                    print(f"❌ PNL neto negativo ({pnl_neto:.2f}). Esperando para cerrar...")
            
            time.sleep(5)
        
        print("✅ Bot reanudado.")
        continue

    
    try:
        process_candles += 1

        # 1) Selección de activo más volátil cada VOLATILITY_CICLOS velas
        if not operations['open'] and volatility_candle_count % VOLATILITY_CICLOS == 0:
            vols = {}
            for sym in SYMBOLS:
                data = convertir_a_df(bot.fetch_ohlcv(sym, INTERVAL, limit=VOLATILITY_CICLOS))
                vols[sym] = (data['close'].pct_change().abs().mean())
            # filtro volatilidad extrema
            filtered = {s: v for s, v in vols.items() if v < EXTREME_VOLATILITY_THRESHOLD}
            if not filtered:
                filtered = vols
            symbol = max(filtered, key=filtered.get)
            print(f"🔄 Activo seleccionado: {symbol} con volatilidad {vols[symbol]:.4f}")

        # 2) Obtener datos y señales
        timer.start("Generación de señal")
        ohlcv = bot.fetch_ohlcv(symbol, INTERVAL, limit=100)
        df = convertir_a_df(ohlcv)
        df = strategy.generate_signals(df)
        last = df.iloc[-1]
        signal = last['signal']
        ts = last['timestamp']
        timer.stop("Generación de señal")

        print(f"\n🕒 {symbol} - Último candle: {ts} | Señal: {signal}")

        # Validaciones modo seguro: spread y volumen
        ticker = bot.fetch_ticker(symbol)
        ask = ticker.get('ask')
        bid = ticker.get('bid')
        volume = df['volume'].iloc[-1]
        if ask is None or bid is None or bid == 0:
            print("⚠️ Datos de ticker incompletos, omitiendo validación de spread.")
            safe = False
        else:
            spread = abs(ask - bid) / bid
            safe = (spread < SAFE_MODE_VALIDATIONS['max_spread'] and volume > SAFE_MODE_VALIDATIONS['min_volume'])
        print(f"🔍 Spread={'N/A' if ask is None or bid is None else f'{spread:.4f}'}, Volumen={volume:.2f}, SafeMode={'OK' if safe else '🚨'}")
        if spread > SAFE_MODE_VALIDATIONS['max_spread']:
            print(f"⚠️ Spread alto: {spread} > {SAFE_MODE_VALIDATIONS['max_spread']}")
        if volume < SAFE_MODE_VALIDATIONS['min_volume']:
            print(f"⚠️ Volumen bajo: {volume} < {SAFE_MODE_VALIDATIONS['min_volume']}")

        # Reentrenamiento dinámico
        if process_candles % RETRAIN_INTERVAL == 0:
            print("🔄 Reentrenando modelo dinámicamente...")
            strategy.train(df)

        # Gestión de operación abierta
        ticker = bot.fetch_ticker(symbol)  # ❗ Se actualiza cada ciclo
        current = ticker.get('last')
        candle_ts = last['timestamp']  # Debe venir de df = strategy.generate_signals(df)
        candle_dt = candle_ts.to_pydatetime()
        if operations['open']:
            if last_bar_time is None:
                last_bar_time = candle_dt
            
            elif candle_dt > last_bar_time:
                operations['candles'] += 1
                last_bar_time = candle_dt
                print(f"🕐 Nueva vela cerrada | Velas acumuladas: {operations['candles']}")
                
            if current:
                pnl_bruto = (current - operations['entry_price']) * operations['qty'] if operations['side'] == 'buy' else (operations['entry_price'] - current) * operations['qty']
                com_entrada = operations['entry_price'] * operations['qty'] * COMISION_PORCENTAJE
                com_salida = current * operations['qty'] * COMISION_PORCENTAJE
                pnl = pnl_bruto - (com_entrada + com_salida)
                
                bot.balance['USDT'] += pnl
                print(f"⏳ Operación abierta | PNL neto={pnl:.2f} USD | Velas={operations['candles']}")
                
                close_position = False
                ts_exit = datetime.now()
                
                if operations['side'] == 'long':
                    tp_price = operations['entry_price'] * (1 + TP_PORCENTAJE)
                    sl_price = operations['entry_price'] * (1 - SL_PORCENTAJE)
                    print(f"LONG ▶ Precio actual: {current:.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}")
                    
                    if current >= tp_price:
                        close_position = True
                        reason = 'Take Profit alcanzado 🎯'
                        res = 'GANANCIA'
                        
                    elif current <= sl_price:
                        close_position = True
                        reason = 'Stop Loss alcanzado 🛑'
                        res = 'PÉRDIDA'
                        
                elif operations['side'] == 'short':
                    tp_price = operations['entry_price'] * (1 - TP_PORCENTAJE)
                    sl_price = operations['entry_price'] * (1 + SL_PORCENTAJE)
                    print(f"SHORT ▶ Precio actual: {current:.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}")
                    
                    if current <= tp_price:
                        close_position = True
                        reason = 'Take Profit alcanzado 🎯'
                        res = 'GANANCIA'
                        
                    elif current >= sl_price:
                        close_position = True
                        reason = 'Stop Loss alcanzado 🛑'
                        res = 'PÉRDIDA'
                        
                if close_position:
                    metrics['total_trades'] += 1
                    metrics['wins'] += int(pnl > 0)
                    metrics['losses'] += int(pnl <= 0)
                    metrics['pnl_sum'] += pnl
                    print(f"📝 Log decisión cierre: {reason}")

                    if pnl > 0:
                        riesgo_actual = min(riesgo_actual + 0.05, riesgo_maximo)
                        print(f"📈 Aumentando riesgo a {riesgo_actual*100:.0f}% por ganancia.")
                    
                    else:
                        riesgo_actual = max(riesgo_actual - 0.05, riesgo_minimo)
                        print(f"📉 Reduciendo riesgo a {riesgo_actual*100:.0f}% por pérdida.")
                    
                    
                   
                    run_async(telegram_bot.notificar_operacion_cerrada(operations, current, pnl, com_entrada, com_salida, res, ts_exit))
                    guardar_operacion(
                    operations['side'], operations['symbol'], operations['entry_price'], current,
                    res, pnl, (com_entrada + com_salida), operations['qty'], operations['ts_entry'], ts_exit)
                    operations.update({'open': False, 'candles': 0})
                    state.drawdown_actual = drawdown_actual
                    state.pnl = pnl
                    state.pnl_bruto = pnl_bruto
                    state.com_entrada = com_entrada
                    state.com_salida = com_salida
                    
                    try:
                        threading.Thread(target=upload_to_sheets).start()
                        print("✅ Datos subidos a Google Sheets.")
                    
                    except Exception as e:
                        print(f"❌ Error al subir a Sheets: {e}")
        
            

        # Abrir nueva operación
        if not operations['open'] and signal in [1,-1] and safe and not hay_evento_economico_cercano_local(now_utc,15):
            side = 'buy' if signal==1 else 'sell'
            riesgo_actual = POSITION_RATIO * (1 - min(bot.get_drawdown(), 0.5))
            capital = CAPITAL_TOTAL * riesgo_actual
            qty = round((capital * LEVERAGE) / df['close'].iloc[-1], 3)
            price = df['close'].iloc[-1]
            if qty*price >= MIN_NOTIONAL:
                timer.start("Ejecución de orden")
                bot.place_order(symbol, side, qty)
                timer.stop("Ejecución de orden")
                
                timer.start("Notificación Telegram")
                run_async(telegram_bot.notificar_operacion_abierta(symbol, side, qty, price))
                timer.stop("Notificación Telegram")

                if metrics['total_trades'] > 5:
                    asyncio.run(telegram_bot.enviar_mensaje_inicio(telegram_bot.AUTHORIZED_USER_ID))

                operations.update({'open':True,'symbol':symbol,'entry_price':price,
                                  'side':side,'qty':qty,'ts_entry':datetime.now(),'candles':0, 'last_candle_ts': ts})
                print(f"🚨 Entrada {side} en {symbol} @ {price} Qty={qty}")
            else:
                print(f"⚠️ Notional insuficiente para abrir: {qty*price:.2f}")

        # Métricas en vivo
        if metrics['total_trades']>0:
            win_rate = metrics['wins']/metrics['total_trades']*100
            avg_pnl = metrics['pnl_sum']/metrics['total_trades']
            print(f"📊 Trades:{metrics['total_trades']} | Winrate:{win_rate:.2f}% | AvgPnL:{avg_pnl:.2f}")

         # Esperar próxima iteración (revisión cada 10 segundos)
        time.sleep(10)


        if drawdown_actual >= 0.25:
            print(f"🚩 Drawdown crítico alcanzado: {drawdown_actual*100:.2f}% (Balance: {capital_actual:.2f} USD)")
            enviar_email("⛔ Bot detenido por drawdown",
                         f"Drawdown alcanzado: {drawdown_actual*100:.2f}%<br>Balance actual: {capital_actual:.2f} USD")
            break  # Detiene el bot


    except KeyboardInterrupt:
        print("\n🚩 Bot detenido manualmente.")
        break
    except Exception as e:
        print(f"❌ Error general: {e}")
        time.sleep(10)


    
