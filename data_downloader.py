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
years = [2021, 2022, 2023, 2024, 2025]
exchange = ccxt.binance()
save_dir = "data/dataset"
os.makedirs(save_dir, exist_ok=True)

def get_random_start_date(year):
    """Genera una fecha aleatoria de inicio dentro del año"""
    start_of_year = datetime(year, 1, 1)
    end_of_year = datetime(year, 12, 31)
    
    # Generar fecha aleatoria
    random_date = start_of_year + timedelta(
        days=random.randint(0, (end_of_year - start_of_year).days)
    )
    
    # Asegurar que hay suficientes datos después de esta fecha
    days_remaining = (end_of_year - random_date).days
    if days_remaining < 30:  # Si quedan menos de 30 días, ajustar
        random_date = end_of_year - timedelta(days=30)
    
    return random_date

def fetch_ohlcv(symbol, interval, since, limit):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe=interval, since=since, limit=limit)
    except Exception as e:
        print(f"❌ Error al descargar {symbol} {interval}: {e}")
        return []

def download_data_for_year(symbol, year, timeframe, total_candles, limit):
    """Descarga datos para un símbolo, año y timeframe específico"""
    print(f"📥 Descargando {symbol} {timeframe} para {year}...")
    
    # Obtener fecha de inicio aleatoria
    start_date = get_random_start_date(year)
    since = int(start_date.timestamp() * 1000)
    
    all_candles = []
    attempts = 0
    max_attempts = 50  # Máximo 50 intentos para evitar loops infinitos
    
    while len(all_candles) < total_candles and attempts < max_attempts:
        candles = fetch_ohlcv(symbol, timeframe, since, limit)
        if not candles:
            break
        
        all_candles.extend(candles)
        since = candles[-1][0] + 1
        attempts += 1
        
        print(f"✅ {len(all_candles)} velas descargadas de {symbol} {timeframe} {year}")
        time.sleep(0.5)  # Rate limiting
    
    if len(all_candles) >= total_candles:
        all_candles = all_candles[:total_candles]  # Limitar al número exacto
    
    return all_candles

def main():
    print("🚀 Iniciando descarga de datos por años...")
    
    for year in years:
        print(f"\n📅 Procesando año {year}")
        year_dir = os.path.join(save_dir, str(year))
        os.makedirs(year_dir, exist_ok=True)
        
        for symbol in symbols:
            symbol_clean = symbol.replace('/', '')
            
            for timeframe, config in timeframes.items():
                print(f"\n📊 Descargando {symbol} {timeframe} para {year}")
                
                # Descargar datos
                candles = download_data_for_year(
                    symbol, year, timeframe, 
                    config['total_candles'], config['limit']
                )
                
                if candles:
                    # Crear DataFrame
                    df = pd.DataFrame(candles)
                    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Guardar archivo
                    filename = os.path.join(year_dir, f"{symbol_clean}_{timeframe}.csv")
                    df.to_csv(filename, index=False)
                    
                    print(f"💾 Guardado: {filename} ({len(df)} registros)")
                    print(f"📅 Rango: {df['timestamp'].min()} a {df['timestamp'].max()}")
                else:
                    print(f"❌ No se pudieron descargar datos para {symbol} {timeframe} {year}")
                
                time.sleep(1)  # Pausa entre descargas
    
    print("\n✅ Descarga completada!")

if __name__ == "__main__":
    main()
