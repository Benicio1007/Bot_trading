import time
import pandas as pd
from datetime import datetime, timezone
from bot.broker import PaperBroker
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import csv
import os
import socket
import sheets.telegram_utils.pnl_telegram as state
from sheets.telegram_utils import telegram_bot
from sheets.telegram_utils import pnl_telegram
import threading
import asyncio
import os
from informe.generar_informe import generar_y_enviar_informe
from informe.enviar_mail import enviar_email
import calendar
from sheets.sheets import upload_to_sheets
from timing.timing import Timer
import torch
from data.modelos.feature_extractor import CNNFeatureExtractor
from data.modelos.sequence_model import SequenceModel
from data.modelos.attention_layer import AttentionBlock
from data.modelos.drl_agent import PPOAgent
from data.training.utils import load_config, normalize
import numpy as np
import torch.nn.functional as F
from data.prepare_data import compute_indicators
import ta

# 📌 Configuración
SYMBOLS = ['BTC/USDT', 'ETH/USDT']  # Lista de activos
INTERVAL = '1m'
CAPITAL_TOTAL = 1000
LEVERAGE = 30
POSITION_RATIO = 0.20
TP_PORCENTAJE = 0.0045  # 0.45%
SL_PORCENTAJE = 0.0025  # 0.25%
COMISION_PORCENTAJE = 0.0004
MIN_NOTIONAL = 100
MAX_CICLOS = 20
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
operations = {'open': False, 'symbol': None, 'entry_price': None,
              'side': None, 'qty': None, 'ts_entry': None, 'candles': 0}
last_selected_symbol = None  # 🔥 Variable global para rotación

# 🔧 Inicialización
bot = PaperBroker()
# justo después de cargar el config
config = load_config(os.path.join(os.path.dirname(__file__), "..", "data", "config", "config.yaml"))

# 📜 Crear archivo CSV si no existe
csv_file = os.path.join(os.path.dirname(__file__), "..", "sheets.operaciones.csv")
if not os.path.isfile(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Tipo', 'Activo', 'Entrada', 'Salida', 'Resultado', 'PNL Neto',
            'Comisión', 'Qty', 'Timestamp Entrada', 'Timestamp Salida'
        ])

# Métricas en vivo
metrics = {'total_trades': 0, 'wins': 0, 'losses': 0, 'pnl_sum': 0}

def esta_en_consolidacion(df, umbral_pct=0.0015, vol_factor=0.3):
    """
    🔥 Filtro MUY LIGHT de consolidación para mejorar win rate
    - Detecta rangos muy pequeños (consolidación)
    - Verifica volumen bajo (falta de interés)
    - Evita trades en mercados laterales
    """
    # Rango de precio en las últimas 5 velas
    rango_pct = (df['high'].iloc[-5:].max() - df['low'].iloc[-5:].min()) / df['close'].iloc[-1]
    
    # Volumen promedio vs volumen actual
    vol_avg = df['volume'].iloc[-20:].mean()  # Promedio de 20 velas
    vol_now = df['volume'].iloc[-1]
    
    # Volatilidad reciente (últimas 10 velas)
    volatilidad = df['close'].iloc[-10:].pct_change().std()
    
    # Criterios de consolidación MUY LIGHT
    rango_pequeno = rango_pct < umbral_pct
    volumen_bajo = vol_now < vol_avg * vol_factor
    volatilidad_baja = volatilidad < 0.0005  # 0.05% de volatilidad (más permisivo)
    
    # Solo evita si hay TODAS las señales de consolidación (más restrictivo)
    señales_consolidacion = sum([rango_pequeno, volumen_bajo, volatilidad_baja])
    
    if señales_consolidacion >= 3:  # TODAS las señales deben estar presentes
        print(f"🔍 Consolidación detectada: Rango={rango_pct:.4f}, Vol={vol_now/vol_avg:.2f}x, Volat={volatilidad:.4f}")
        return True
    
    return False

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
    ruta_informe = os.path.join(os.path.dirname(__file__), "..", "informe", "informe_mensual.html")

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
    df = pd.DataFrame(ohlcv, columns=pd.Index(['timestamp', 'open', 'high', 'low', 'close', 'volume']))
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def hay_evento_economico_cercano_local(now, minutos_antes=15, minutos_despues=30):
    try:
        with open(os.path.join(os.path.dirname(__file__), "..", "eventos_economicos_usa_2025.csv"), newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                evento_dt = datetime.strptime(row['Fecha (UTC)'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
                delta = (now - evento_dt).total_seconds() / 60
                if -minutos_antes <= delta <= minutos_despues:
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

def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            asyncio.create_task(coro)
        else:
            loop.run_until_complete(coro)
    except RuntimeError:
        asyncio.run(coro)

def classify_candle(row):
    body  = row['close'] - row['open']
    upper = row['high']  - max(row['close'], row['open'])
    lower = min(row['close'], row['open']) - row['low']

    if abs(body) < 0.001 * row['open']:
        return 0              
    elif body > 0 and upper < body*0.1 and lower < body*0.1:
        return 1              
    elif body < 0 and upper < -body*0.1 and lower < -body*0.1:
        return -1            
    else:
        return 2              

def compute_indicators_realtime(df):
    # Usar la función original de prepare_data.py
    df = compute_indicators(df)
    
    # 🔥 Agregar candle_type EXACTAMENTE como en entrenamiento
    df['candle_type'] = df.apply(classify_candle, axis=1).astype(np.int8)
    
    return df

def generar_senal_ia(df, symbol="BTC/USDT", interval="1m"):
    print(f"🔍 DEBUG - Iniciando generar_senal_ia para {symbol}")
    df = compute_indicators_realtime(df)
    df = df.copy()
    print(f"🔍 DEBUG - DataFrame después de compute_indicators: {df.shape}")

    if isinstance(symbol, list):
        symbol = symbol[0]
    if isinstance(interval, list):
        interval = interval[0]

    symbol_map = {"BTC/USDT": 0, "ETH/USDT": 1}
    timeframe_map = {"1m": 0, "5m": 1}

    df["symbol_code"] = symbol_map.get(symbol, 0)
    df["timeframe_code"] = timeframe_map.get(interval, 0)
    print(f"🔍 DEBUG - Symbol code: {df['symbol_code'].iloc[-1]}, Timeframe code: {df['timeframe_code'].iloc[-1]}")

    cols = config["data"]["features"]
    seq_len = config["data"]["sequence_length"]
    print(f"🔍 DEBUG - Features requeridos: {len(cols)}, Sequence length: {seq_len}")

    # 🔥 Validación de features faltantes y orden correcto
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    if missing:
        print(f"🚨 ERROR: Faltan features requeridos: {missing}")
        return 0, 0.0
    if extra:
        print(f"⚠️ Features extra detectados: {extra}")
    
    # 🔥 Re-ordenar features exactamente como en entrenamiento
    df = df[cols]
    print(f"🔍 DEBUG - Features ordenados correctamente: {list(df.columns)}")

    if len(df) < seq_len + 1:
        print(f"🔍 DEBUG - Error: len(df)={len(df)}, necesitas al menos {seq_len + 1}")
        return 0, 0.0

    print(f"🔍 DEBUG - DataFrame tiene {len(df)} filas, suficientes features")
    df_feat = normalize(df).fillna(0)
    x_seq = df_feat.iloc[-seq_len:].values
    x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 🔍 DEBUG: Mostrar muestra de datos de entrada
    print(f"🔍 DEBUG - Datos de entrada:")
    print(f"   Shape: {x_tensor.shape}")
    print(f"   Primeros 5 valores del primer feature: {x_tensor[0, 0, :5].cpu().numpy()}")
    print(f"   Últimos 5 valores del último feature: {x_tensor[0, -1, -5:].cpu().numpy()}")
    print(f"   Rango de valores: [{x_tensor.min().item():.4f}, {x_tensor.max().item():.4f}]")
    print(f"   Features disponibles: {len(cols)} - {cols[:5]}...")

    with torch.no_grad():
        features = feature_extractor(x_tensor)
        sequence = sequence_model(features)
        context = attention(sequence)
        logits = agent(context)
        prob = torch.sigmoid(logits.squeeze()).item()
        print(f"🔍 DEBUG - Logits: {logits.squeeze().item():.6f}, Prob: {prob:.6f}")

    fixed_threshold = config['training'].get('fixed_threshold', 0.32)
    if prob >= fixed_threshold:
        return 1, prob
    elif prob <= (1 - fixed_threshold):
        return -1, prob
    else:
        return 0, prob

device = torch.device(config['training']['device'])

feature_extractor = CNNFeatureExtractor(config).to(device)
sequence_model = SequenceModel(config).to(device)
attention = AttentionBlock(config).to(device)
agent = PPOAgent(config).to(device)

checkpoint = torch.load(os.path.join(os.path.dirname(__file__), "..", "checkpoints", "model_last.pt"), map_location=device, weights_only=False)
feature_extractor.load_state_dict(checkpoint['feature_extractor'])
sequence_model.load_state_dict(checkpoint['sequence_model'])
attention.load_state_dict(checkpoint['attention'])
agent.load_state_dict(checkpoint['agent'])

feature_extractor.eval()
sequence_model.eval()
attention.eval()
agent.eval()

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
                # Calcular pnl_bruto, com_entrada y com_salida igual que en el bloque principal
                current = bot.fetch_ticker(operations['symbol']).get('last')
                if current:
                    pnl_bruto = (current - operations['entry_price']) * operations['qty'] if operations['side'] == 'buy' else (operations['entry_price'] - current) * operations['qty']
                    com_entrada = operations['entry_price'] * operations['qty'] * COMISION_PORCENTAJE
                    com_salida = current * operations['qty'] * COMISION_PORCENTAJE
                    pnl_neto = pnl_bruto - (com_entrada + com_salida)
                    
                    if pnl_neto >= 0:
                        print("✅ Operación positiva detectada durante pausa. Cerrando...")
                        close_position = True
                        operacion_abierta = False
                        operacion_actual = None
                    else:
                        print(f"❌ PNL neto negativo ({pnl_neto:.2f}). Esperando para cerrar...")
                else:
                    print("⚠️ No se pudo obtener el precio actual para calcular PNL durante pausa.")
                
            time.sleep(5)
        
        print("✅ Bot reanudado.")
        continue

    try:
        process_candles += 1

        # 🔄 Selección de activo con rotación inteligente
        if not operations['open'] and volatility_candle_count % VOLATILITY_CICLOS == 0:
            ranking = []
            for sym in SYMBOLS:
                data = convertir_a_df(bot.fetch_ohlcv(sym, INTERVAL, limit=VOLATILITY_CICLOS))
                vola = data['close'].pct_change().abs().mean()
                vol = data['volume'].iloc[-1]
                score = vola * np.log(vol + 1)
                ranking.append((score, sym, vola, vol))
            ranking.sort(reverse=True)
            
            # 🔥 ROTACIÓN INTELIGENTE: Alternar entre BTC y ETH
            if last_selected_symbol:
                # Si el último fue ETH, priorizar BTC y viceversa
                if last_selected_symbol == 'ETH/USDT':
                    # Priorizar BTC
                    btc_score = next((r[0] for r in ranking if r[1] == 'BTC/USDT'), 0)
                    eth_score = next((r[0] for r in ranking if r[1] == 'ETH/USDT'), 0)
                    if btc_score > eth_score * 0.7:  # Si BTC tiene al menos 70% del score de ETH
                        symbol = 'BTC/USDT'
                        score, vola, vol = next((r[0], r[2], r[3]) for r in ranking if r[1] == 'BTC/USDT')
                    else:
                        score, symbol, vola, vol = ranking[0][0], ranking[0][1], ranking[0][2], ranking[0][3]
                else:
                    # Priorizar ETH
                    btc_score = next((r[0] for r in ranking if r[1] == 'BTC/USDT'), 0)
                    eth_score = next((r[0] for r in ranking if r[1] == 'ETH/USDT'), 0)
                    if eth_score > btc_score * 0.7:  # Si ETH tiene al menos 70% del score de BTC
                        symbol = 'ETH/USDT'
                        score, vola, vol = next((r[0], r[2], r[3]) for r in ranking if r[1] == 'ETH/USDT')
                    else:
                        score, symbol, vola, vol = ranking[0][0], ranking[0][1], ranking[0][2], ranking[0][3]
            else:
                # Primera vez: usar el mejor score
                score, symbol, vola, vol = ranking[0][0], ranking[0][1], ranking[0][2], ranking[0][3]
            
            last_selected_symbol = symbol
            print(f"🔄 Activo seleccionado: {symbol} | Score={score:.6f} | Volatilidad={vola:.4f} | Volumen={vol:.2f}")
            print(f"📊 Ranking completo: {[(r[1], f'{r[0]:.4f}') for r in ranking]}")

        # 🔥 Obtener datos del símbolo actual
        ohlcv = bot.fetch_ohlcv(symbol, INTERVAL, limit=100)
        df = convertir_a_df(ohlcv)
        
        # 🔥 Filtro de consolidación LIGHT para mejorar win rate
        # 🔥 SOLO aplicar si NO hay operación abierta
        consolidation_detected = False
        if not operations['open']:
            consolidation_detected = esta_en_consolidacion(df, umbral_pct=0.0015, vol_factor=0.3)
            if consolidation_detected:
                print("⏸️ Zona de consolidación detectada. Trade evitado.")

        # 🔥 SIEMPRE revisar operaciones abiertas, independientemente de consolidación
        ticker = bot.fetch_ticker(symbol)  
        current = ticker.get('last')
        candle_ts = df.iloc[-1]['timestamp']
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
                
                if operations['side'] == 'buy':  # 🔥 Corregido: era 'long', debe ser 'buy'
                    volatilidad = df['close'].pct_change().std()  # desviación estándar reciente
                    TP_dinamico = min(0.01, max(0.003, volatilidad * 3))
                    SL_dinamico = min(0.007, max(0.002, volatilidad * 2))
                    
                    tp_price = operations['entry_price'] * (1 + TP_dinamico)
                    sl_price = operations['entry_price'] * (1 - SL_dinamico)
                    print(f"LONG ▶ Precio actual: {current:.4f} | TP: {tp_price:.4f} | SL: {sl_price:.4f}")
                    
                    if current >= tp_price:
                        close_position = True
                        reason = 'Take Profit alcanzado 🎯'
                        res = 'GANANCIA'
                        
                    elif current <= sl_price:
                        close_position = True
                        reason = 'Stop Loss alcanzado 🛑'
                        res = 'PÉRDIDA'
                        
                elif operations['side'] == 'sell':  # 🔥 Corregido: era 'short', debe ser 'sell'
                    volatilidad = df['close'].pct_change().std()  # desviación estándar reciente
                    TP_dinamico = min(0.01, max(0.003, volatilidad * 3))
                    SL_dinamico = min(0.007, max(0.002, volatilidad * 2))
                    tp_price = operations['entry_price'] * (1 - TP_dinamico)  # 🔥 Corregido: para sell, TP es menor
                    sl_price = operations['entry_price'] * (1 + SL_dinamico)   # 🔥 Corregido: para sell, SL es mayor
                    
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
        
        # 🔥 Solo generar señales si NO hay consolidación y NO hay operación abierta
        if not consolidation_detected and not operations['open']:
            timer.start("Generación de señal")
            signal, prob = generar_senal_ia(df, symbol=symbol, interval=INTERVAL)
            last = df.iloc[-1]
            ts = last['timestamp']
            fixed_threshold = config['training'].get('fixed_threshold', 0.35)
            print(f"🧠 Señal IA generada: {signal} (1=buy, -1=sell, 0=hold)")
            print(f"📊 Confianza: {prob:.4f} | Umbral: {fixed_threshold:.4f} | Diferencia: {abs(prob - 0.5):.4f}")
            timer.stop("Generación de señal")

            print(f"\n🕒 {symbol} - Último candle: {ts} | Señal: {signal}")

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

            # 🔥 Solo abrir nuevas operaciones si no hay consolidación
            if signal in [1,-1] and safe and not hay_evento_economico_cercano_local(now_utc,15,30):
                side = 'buy' if signal==1 else 'sell'
                riesgo_actual = POSITION_RATIO * (1 - min(bot.get_drawdown(), 0.5))
                capital = CAPITAL_TOTAL * riesgo_actual
                qty = round((capital * LEVERAGE) / df['close'].iloc[-1], 3)
                price = df['close'].iloc[-1]
                if qty*price >= MIN_NOTIONAL:
                    timer.start("Ejecución de orden")
                    spread_pct = (ask - bid) / bid * 100 if bid else None
                    order = None
                    order_price = None
                    if spread_pct is not None and spread_pct > 0.02:
                        # Orden limit/post-only
                        order_price = ask if side == 'buy' else bid
                        order = bot.place_order(symbol, side, qty, price=order_price, order_type='limit')
                        print(f"📝 Orden LIMIT enviada: {side} {qty} {symbol} @ {order_price:.4f} (spread {spread_pct:.4f}%)")
                    else:
                        # Orden de mercado
                        order = bot.place_order(symbol, side, qty, order_type='market')
                        print(f"📝 Orden MARKET enviada: {side} {qty} {symbol} (spread {spread_pct:.4f}%)")

                    # Loguear después de la ejecución
                    if order:
                        executed_price = order.get('price', order_price if order_price is not None else price)
                        fee = executed_price * qty * COMISION_PORCENTAJE if executed_price else None
                        funding = 0  # Simulado o real si tienes acceso
                        print(f"✅ Ejecutado: {side} {qty} {symbol} @ {executed_price} | Fee: {fee:.4f} | Funding: {funding:.4f}")

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
        
        # 🔥 Actualizar contador de volatilidad
        volatility_candle_count += 1
        
        if metrics['total_trades']>0:
            win_rate = metrics['wins']/metrics['total_trades']*100
            avg_pnl = metrics['pnl_sum']/metrics['total_trades']
            print(f"📊 Trades:{metrics['total_trades']} | Winrate:{win_rate:.2f}% | AvgPnL:{avg_pnl:.2f}")

        time.sleep(10)

        if drawdown_actual >= 0.25:
            print(f"🚩 Drawdown crítico alcanzado: {drawdown_actual*100:.2f}% (Balance: {capital_actual:.2f} USD)")
            enviar_email("⛔ Bot detenido por drawdown",
                         f"Drawdown alcanzado: {drawdown_actual*100:.2f}%<br>Balance actual: {capital_actual:.2f} USD")
            break  

    except KeyboardInterrupt:
        print("\n🚩 Bot detenido manualmente.")
        break
    except Exception as e:
        print(f"❌ Error general: {e}")
        time.sleep(10) 