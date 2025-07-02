import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ta.trend
import ta.momentum
import ta.volatility
import ta.volume
from tqdm import tqdm
from collections import Counter

class TradingDataset(Dataset):
    def __init__(self, data, sequence_length=60):
        self.sequence_length = sequence_length
        self.data = data
        self.X, self.y = self.create_sequences()

    def create_sequences(self):
        X, y = [], []
        for i in range(len(self.data) - self.sequence_length - 1):
            seq = self.data[i:i+self.sequence_length]
            close_now = self.data[i+self.sequence_length - 1][3]  # close actual
            future_prices = [self.data[i+self.sequence_length + j][3] for j in range(3)]  # próximos 3
            direction = 1 if max(future_prices) > close_now * 1.002 else 0 if min(future_prices) < close_now * 0.998 else None
            if direction is None:
                continue  
            X.append(seq)
            y.append(direction)
        return np.array(X), np.array(y)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

def compute_indicators(df):
    df = df.copy()

    # === Features clásicos ===
    # EMA
    df['ema_9'] = ta.trend.ema_indicator(df['close'], window=9, fillna=True)
    df['ema_21'] = ta.trend.ema_indicator(df['close'], window=21, fillna=True)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50, fillna=True)
    df['ema_slope'] = df['ema_9'].diff()

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14, fillna=True)

    # MACD
    df['macd_hist'] = ta.trend.macd_diff(df['close'], fillna=True)

    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14, fillna=True)

    # Stochastic Oscillator
    df['stochastic_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], fillna=True)
    df['stochastic_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], fillna=True)

    # Bollinger Bands Percent
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2, fillna=True)
    df['bb_percent'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())

    # Bollinger Bands Width
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2, fillna=True)
    df['bollinger_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']

    # Candle features
    df['candle_body'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['is_doji'] = (df['candle_body'] < (df['high'] - df['low']) * 0.1).astype(int)
    df['is_marubozu'] = ((df['upper_wick'] < df['candle_body'] * 0.1) & (df['lower_wick'] < df['candle_body'] * 0.1)).astype(int)
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df['close_near_highs'] = ((df['close'] > df['high'] * 0.95)).astype(int)

    # Hora, minuto, día de la semana
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['minute'] = pd.to_datetime(df['timestamp']).dt.minute
    df['weekday'] = pd.to_datetime(df['timestamp']).dt.weekday

    # === Features de volumen ===
    df['delta_volume'] = df['volume'].diff().fillna(0)
    df['volume_spike'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(int)
    df['institutional_volume_bar'] = (df['volume'] > df['volume'].rolling(50).mean() * 2).astype(int)
    
    # 🔥 imbalance_ratio
    if 'bid_volume' in df.columns and 'ask_volume' in df.columns:
        df['imbalance_ratio'] = (df['bid_volume'] - df['ask_volume']) / (df['bid_volume'] + df['ask_volume'] + 1e-9)
    else:
        df['imbalance_ratio'] = 0.0

    # ⚠️ sweep_detection
    df['sweep_detection'] = ((df['volume'] > df['volume'].rolling(10).mean() * 3) & (df['close'] > df['open'])).astype(int)

    # 💸 funding_rate (solo para BTC y ETH, simulado si no existe)
    if 'funding_rate' in df.columns:
        df['funding_rate'] = df['funding_rate']
    else:
        df['funding_rate'] = 0.0

    # 🧊 price_spread_pct
    if 'ask_price' in df.columns and 'bid_price' in df.columns:
        df['price_spread_pct'] = (df['ask_price'] - df['bid_price']) / df['close']
    else:
        df['price_spread_pct'] = 0.0

    # 💥 aggressive_volume_ratio
    if 'market_buy_volume' in df.columns:
        df['aggressive_volume_ratio'] = df['market_buy_volume'] / (df['volume'] + 1e-9)
    else:
        df['aggressive_volume_ratio'] = 0.0

    # 🛑 stop_run_detector
    df['stop_run_detector'] = ((df['high'] > df['high'].shift(1)) & (df['close'] < df['open'])).astype(int)

    # Momentum 5
    df['momentum_5'] = df['close'] / df['close'].shift(5) - 1

    # RSI Cross
    df['rsi_cross'] = ((df['rsi'] > 50) & (df['rsi'].shift(1) <= 50)).astype(int)

    # 1. Señal anterior: predicción binaria del modelo anterior (o 0 al principio)
    if 'prev_signal' not in df.columns:
        df['prev_signal'] = 0  # por si corres desde cero
    # 2. Desplazá una fila hacia abajo para que el valor del minuto anterior
    df['previous_signal'] = df['prev_signal'].shift(1).fillna(0).astype(np.int8)
    # 3. (opcional) eliminá 'prev_signal' si ya no la necesitás
    df.drop(columns=['prev_signal'], inplace=True)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

def encode_symbol(symbol):
    symbol_map = {'BTCUSDT': 0, 'ETHUSDT': 1, 'SOLUSDT': 2, 'XRPUSDT': 3}
    return symbol_map.get(symbol, -1)


def create_dataset(config):
    inputs = config['data']['inputs']
    sequence_length = config['data']['sequence_length']
    features = config['data']['features']

    combined_data = []

    # Agrupar archivos por símbolo
    symbol_groups = {}
    for entry in inputs:
        symbol = entry['symbol']
        timeframe = entry['timeframe']
        if symbol not in symbol_groups:
            symbol_groups[symbol] = {}
        symbol_groups[symbol][timeframe] = entry['path']

    for symbol, tf_paths in tqdm(symbol_groups.items()):
        if 1 not in tf_paths or 5 not in tf_paths:
            continue  # Aseguramos que existan ambos timeframes

        df_1m = pd.read_csv(tf_paths[1])
        df_5m = pd.read_csv(tf_paths[5])

        # Indicadores
        df_1m = compute_indicators(df_1m)
        df_5m = compute_indicators(df_5m)

        # Convertimos timestamps (recomendado en UTC)
        df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'])
        df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])

        # Merge alineado por tiempo (el más reciente 5m antes o igual al 1m)
        df_5m.set_index('timestamp', inplace=True)
        df_1m['timestamp_5m'] = df_1m['timestamp'].apply(
            lambda ts: df_5m.index[df_5m.index <= ts].max()
        )
        df_merged = df_1m.merge(df_5m, left_on='timestamp_5m', right_index=True, suffixes=('_1m', '_5m'))

        # Features a usar
        selected_features = []
        for f in features:
            if f not in ['symbol_code', 'timeframe_code']:
                selected_features.append(f + "_1m")
                selected_features.append(f + "_5m")
        df_merged['symbol_code'] = encode_symbol(symbol) / 3  # Normalizado
        selected_features += ['symbol_code']

        # Asegurar que todos los features existan en el DataFrame
        for feat in selected_features:
            if feat not in df_merged.columns:
                df_merged[feat] = 0
        
        df_final = df_merged[selected_features].copy()
        numeric_cols = [col for col in df_final.columns if col != 'symbol_code']
        # Normalización rolling sobre DataFrame
        if len(numeric_cols) > 0:
            df_numeric = df_final[numeric_cols]
            if not isinstance(df_numeric, pd.DataFrame):
                df_numeric = pd.DataFrame(df_numeric, columns=pd.Index(numeric_cols))
            rolling_mean = df_numeric.rolling(100).mean()
            rolling_std = df_numeric.rolling(100).std()
            df_final[numeric_cols] = (df_numeric - rolling_mean) / rolling_std
        df_final = df_final.dropna()
        combined_data.append(df_final.values)

    all_data = np.concatenate(combined_data, axis=0)
    dataset = TradingDataset(all_data, sequence_length)
    num_features = all_data.shape[1]
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
    labels = dataset.y
    counts = Counter(labels)
    min_count = min(counts.values())
    indices_0 = [i for i, y in enumerate(labels) if y == 0]
    indices_1 = [i for i, y in enumerate(labels) if y == 1]
    balanced_indices = indices_0[:min_count] + indices_1[:min_count]
    np.random.shuffle(balanced_indices)
    dataset.X = dataset.X[balanced_indices]
    dataset.y = dataset.y[balanced_indices]
    unique, counts = np.unique(labels, return_counts=True)
    total = sum(counts)
    weights = [total / c for c in counts]
    class_weights = torch.tensor(weights, dtype=torch.float32)

    print(f"Clase 0: {counts[0]}, Clase 1: {counts[1]}")
    print(f"Pesos de clase: {class_weights}")

    return dataset, num_features, class_weights

   