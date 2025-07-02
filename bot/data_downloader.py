import ccxt
import pandas as pd
import os
import time
import random
from datetime import datetime, timedelta

# Configuración
symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT']
timeframes = {
    '5m': {'limit': 1000, 'total_candles': 10000},
    '1m': {'limit': 1000, 'total_candles': 50000}
}
# Marzo 2025
exchange = ccxt.binance()
save_dir = "data/dataset/marzo_2025"
os.makedirs(save_dir, exist_ok=True)

def get_marzo_start_date():
    """Devuelve el 1 de marzo de 2025"""
    return datetime(2025, 3, 1)

def get_marzo_end_date():
    """Devuelve el 31 de marzo de 2025"""
    return datetime(2025, 3, 31, 23, 59, 59)

def fetch_ohlcv(symbol, interval, since, limit):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=limit)
    except Exception as e:
        print(f"❌ Error al descargar {symbol} {interval}: {e}")
        return []

def download_data_for_marzo(symbol, timeframe, total_candles, limit):
    """Descarga datos para un símbolo y timeframe específico de marzo 2025"""
    print(f"📥 Descargando {symbol} {timeframe} para marzo 2025...")
    
    # Obtener fecha de inicio de marzo
    start_date = get_marzo_start_date()
    end_date = get_marzo_end_date()
    since = int(start_date.timestamp() * 1000)
    end_timestamp = int(end_date.timestamp() * 1000)
    
    all_candles = []
    attempts = 0
    max_attempts = 100  # Máximo 100 intentos para evitar loops infinitos
    
    while len(all_candles) < total_candles and attempts < max_attempts:
        candles = fetch_ohlcv(symbol, timeframe, since, limit)
        if not candles:
            break
        
        # Filtrar solo velas de marzo 2025
        marzo_candles = [candle for candle in candles if candle[0] <= end_timestamp]
        all_candles.extend(marzo_candles)
        
        # Si la última vela está fuera de marzo, parar
        if candles[-1][0] > end_timestamp:
            break
            
        since = candles[-1][0] + 1
        attempts += 1
        
        print(f"✅ {len(all_candles)} velas descargadas de {symbol} {timeframe} marzo 2025")
        time.sleep(0.5)  # Rate limiting
    
    if len(all_candles) >= total_candles:
        all_candles = all_candles[:total_candles]  # Limitar al número exacto
    
    return all_candles

def main():
    print("🚀 Iniciando descarga de datos de marzo 2025...")
    print(f"📅 Período: {get_marzo_start_date().strftime('%d/%m/%Y')} - {get_marzo_end_date().strftime('%d/%m/%Y')}")
    print(f"📁 Guardando en: {save_dir}")
    
    for symbol in symbols:
        symbol_clean = symbol.replace('/', '')
        
        for timeframe, config in timeframes.items():
            print(f"\n📊 Descargando {symbol} {timeframe} para marzo 2025")
            
            # Descargar datos
            candles = download_data_for_marzo(
                symbol, timeframe, 
                config['total_candles'], config['limit']
            )
            
            if candles:
                # Crear DataFrame
                df = pd.DataFrame(candles)
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Guardar archivo
                filename = os.path.join(save_dir, f"{symbol_clean}_{timeframe}.csv")
                df.to_csv(filename, index=False)
                
                print(f"💾 Guardado: {filename} ({len(df)} registros)")
                print(f"📅 Rango: {df['timestamp'].min()} a {df['timestamp'].max()}")
            else:
                print(f"❌ No se pudieron descargar datos para {symbol} {timeframe} marzo 2025")
            
            time.sleep(1)  # Pausa entre descargas
    
    print("\n✅ Descarga de marzo 2025 completada!")

if __name__ == "__main__":
    main()
