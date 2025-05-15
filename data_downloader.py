import ccxt
import pandas as pd
import os
import time
from datetime import datetime

# Configuración
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
interval = '5m'
limit = 1000  # Binance solo deja 1000 velas por request
total_candles = 8_000
exchange = ccxt.binance()
save_dir = "data"
os.makedirs(save_dir, exist_ok=True)

def fetch_ohlcv(symbol, interval, since):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=interval, since=since)
    except Exception as e:
        print(f"❌ Error al descargar {symbol}: {e}")
        return []

for symbol in symbols:
    print(f"📥 Descargando {symbol}...")
    all_candles = []
    since = exchange.parse8601('2025-01-01T00:00:00Z')
    
    while len(all_candles) < total_candles:
        candles = fetch_ohlcv(symbol, interval, since)
        if not candles:
            break
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        print(f"✅ {len(all_candles)} velas descargadas de {symbol}")
        time.sleep(1)

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    filename = os.path.join(save_dir, f"{symbol.replace('/', '')}_{interval}.csv")
    df.to_csv(filename, index=False)
    print(f"💾 Guardado en {filename}")
