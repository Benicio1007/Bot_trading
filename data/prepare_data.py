import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import ta
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

    # EMA
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10, fillna=True)
    df['ema_50'] = ta.trend.ema_indicator(df['close'], window=50, fillna=True)

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14, fillna=True)

    # MACD
    macd = ta.trend.macd(df['close'], fillna=True)
    df['macd'] = macd
    df['macd_signal'] = ta.trend.macd_signal(df['close'], fillna=True)
    df['macd_hist'] = ta.trend.macd_diff(df['close'], fillna=True)

    # ATR
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14, fillna=True)

    # OBV
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'], fillna=True)

    # Stochastic Oscillator
    stoch = ta.momentum.stoch(df['high'], df['low'], df['close'], fillna=True)
    df['stochastic_k'] = stoch
    df['stochastic_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], fillna=True)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2, fillna=True)
    df['bollinger_upper'] = bb.bollinger_hband()
    df['bollinger_lower'] = bb.bollinger_lband()

    # ADX
    df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=14, fillna=True)

    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour / 23.0


    body = df['close'] - df['open']
    wick = df['high'] - df['low']
    body_ratio = abs(body) / wick.replace(0, 1e-9)

    df['candle_type'] = 0  # default doji
    df.loc[body > 0, 'candle_type'] = 1  # bullish
    df.loc[body < 0, 'candle_type'] = -1  # bearish
    
    df['close_change_1'] = df['close'].pct_change(1)
    df['close_change_3'] = df['close'].pct_change(3)
    df['close_change_5'] = df['close'].pct_change(5)

    df['volume_change'] = df['volume'].pct_change().fillna(0)
    df['liquidity_gap'] = (df['high'] - df['low']) / df['close']


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

        df_final = df_merged[selected_features].dropna()

        # Normalizar (excepto symbol_code)
        numeric_cols = [col for col in df_final.columns if col != 'symbol_code']
        df_final[numeric_cols] = (df_final[numeric_cols] - df_final[numeric_cols].rolling(100).mean()) / df_final[numeric_cols].rolling(100).std()
        df_final = df_final.dropna()  # asegurate de limpiar NaNs resultantes

        combined_data.append(df_final.values)

    # Juntamos todo
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

   