import os 
import pandas as pd
import time
from datetime import datetime
from broker import PaperBroker
from strategy import MLStrategy
from features import build_features

# Configuración
SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'SOL/USDT']
INTERVAL = '5m'
CAPITAL_TOTAL = 1000
LEVERAGE = 30
POSITION_RATIO = 0.20
TP_PORCENTAJE = 0.004
SL_PORCENTAJE = 0.002
COMISION_PORCENTAJE = 0.0004
MIN_NOTIONAL = 100
RETRAIN_INTERVAL = 10
VOLATILITY_CICLOS = 10
EXTREME_VOLATILITY_THRESHOLD = 0.05

bot = PaperBroker()
strategy = MLStrategy(threshold=0.80)

def convertir_a_df(data):
    return pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])

def initial_train():
    combined = pd.DataFrame()
    for sym in SYMBOLS:
        raw = bot.fetch_ohlcv(sym, INTERVAL, limit=500)
        df = convertir_a_df(raw)
        df['symbol'] = sym
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        combined = pd.concat([combined, df])
    strategy.train(combined)

try:
    _ = strategy.model.predict([[0]*10])
except:
    initial_train()

capital_actual = bot.get_balance()
capital_maximo = capital_actual
riesgo_actual = POSITION_RATIO
riesgo_minimo = 0.10
riesgo_maximo = 0.30
drawdown_actual = 0
drawdown_maximo = 0

csv_file = 'operaciones.csv'
if os.path.exists(csv_file):
    os.remove(csv_file)

metrics = {'total_trades': 0, 'wins': 0, 'losses': 0, 'pnl_sum': 0}
operaciones_log = []

symbol_data = {}
for symbol in SYMBOLS:
    file_name = f"data/{symbol.replace('/', '')}_5m.csv"
    df = pd.read_csv(file_name)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = build_features(df)
    symbol_data[symbol] = df

symbol = SYMBOLS[0]
df = symbol_data[symbol]
operaciones = {'open': False}
volatility_candle_count = 0

for i in range(100, len(df)):
    volatility_candle_count += 1

    if not operaciones['open'] and volatility_candle_count % VOLATILITY_CICLOS == 0:
        vols = {}
        for sym in SYMBOLS:
            data = symbol_data[sym].iloc[i-VOLATILITY_CICLOS:i]
            vols[sym] = data['close'].pct_change().abs().mean()
        filtered = {s: v for s, v in vols.items() if v < EXTREME_VOLATILITY_THRESHOLD}
        if not filtered:
            filtered = vols
        symbol = max(filtered, key=filtered.get)
        df = symbol_data[symbol]

    row = df.iloc[i]
    price = row['close']
    ts = row['timestamp']

    if operaciones['open']:
        operaciones['candles'] += 1
        current = price
        pnl_bruto = (current - operaciones['entry_price']) * operaciones['qty'] if operaciones['side'] == 'buy' else (operaciones['entry_price'] - current) * operaciones['qty']
        com_entrada = operaciones['entry_price'] * operaciones['qty'] * COMISION_PORCENTAJE
        com_salida = current * operaciones['qty'] * COMISION_PORCENTAJE
        pnl = pnl_bruto - (com_entrada + com_salida)

        if operaciones['side'] == 'buy':
            tp = operaciones['entry_price'] * (1 + TP_PORCENTAJE)
            sl = operaciones['entry_price'] * (1 - SL_PORCENTAJE)
            close_condition = current >= tp or current <= sl
        else:
            tp = operaciones['entry_price'] * (1 - TP_PORCENTAJE)
            sl = operaciones['entry_price'] * (1 + SL_PORCENTAJE)
            close_condition = current <= tp or current >= sl

        if close_condition:
            res = 'GANANCIA' if pnl > 0 else 'PÉRDIDA'
            ts_exit = ts
            metrics['total_trades'] += 1
            metrics['wins'] += int(pnl > 0)
            metrics['losses'] += int(pnl <= 0)
            metrics['pnl_sum'] += pnl
            capital_actual += pnl
            capital_maximo = max(capital_maximo, capital_actual)
            drawdown_actual = (capital_maximo - capital_actual) / capital_maximo
            drawdown_maximo = max(drawdown_maximo, drawdown_actual)
            riesgo_actual = min(riesgo_actual + 0.05, riesgo_maximo) if pnl > 0 else max(riesgo_actual - 0.05, riesgo_minimo)

            operaciones_log.append([ 
                operaciones['side'], symbol, operaciones['entry_price'], current,
                res, round(pnl, 2), round(com_entrada + com_salida, 2), operaciones['qty'],
                operaciones['ts_entry'], ts_exit, round(capital_actual, 2)
            ])

            operaciones = {'open': False}

    else:
        sub_df = df.iloc[i-100:i]
        sub_df = strategy.generate_signals(sub_df)
        last = sub_df.iloc[-1]
        signal = last['signal']

        if signal in [1, -1]:
            side = 'buy' if signal == 1 else 'sell'
            capital = CAPITAL_TOTAL * riesgo_actual
            qty = round((capital * LEVERAGE) / price, 3)
            if qty * price >= MIN_NOTIONAL:
                operaciones = {
                    'open': True,
                    'symbol': symbol,
                    'entry_price': price,
                    'side': side,
                    'qty': qty,
                    'ts_entry': ts,
                    'candles': 0
                }

# Guardar CSV y métricas
if operaciones_log:
    df_ops = pd.DataFrame(operaciones_log, columns=[ 
        'Tipo', 'Activo', 'Entrada', 'Salida', 'Resultado', 'PNL Neto',
        'Comisión', 'Qty', 'Timestamp Entrada', 'Timestamp Salida', 'Capital'
    ])
    df_ops.to_csv(csv_file, index=False)

winrate = metrics['wins'] / metrics['total_trades'] * 100 if metrics['total_trades'] > 0 else 0
avg_pnl = metrics['pnl_sum'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
pnl_acumulado = metrics['pnl_sum']

# Mostrar los resultados finales
print("\n================= RESULTADOS FINALES =================")
print(f"📈 Total de operaciones: {metrics['total_trades']}")
print(f"✅ Winrate: {winrate:.2f}%")
print(f"💵 PnL promedio por trade: {avg_pnl:.2f} USD")
print(f"💰 PnL acumulado total: {pnl_acumulado:.2f} USD")
print(f"📉 Drawdown máximo: {drawdown_maximo*100:.2f}%")
print(f"💰 Balance final: {capital_actual:.2f} USD")
print("=====================================================\n")
