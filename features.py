import pandas as pd
import numpy as np

def detect_order_blocks(df):
    df = df.copy()
    df['order_block'] = 0
    for i in range(2, len(df)):
        if df['high'].iloc[i] < df['high'].iloc[i - 1] and df['low'].iloc[i] > df['low'].iloc[i - 1]:
            df.at[df.index[i], 'order_block'] = 1
    return df

def detect_liquidity_zones(df):
    df = df.copy()
    df['liquidity_zone'] = 0
    window = 20
    for i in range(window, len(df)):
        recent_lows = df['low'].iloc[i - window:i]
        recent_highs = df['high'].iloc[i - window:i]
        if df['low'].iloc[i] < recent_lows.min() * 1.001:
            df.at[df.index[i], 'liquidity_zone'] = 1
        elif df['high'].iloc[i] > recent_highs.max() * 0.999:
            df.at[df.index[i], 'liquidity_zone'] = -1
    return df

def detect_anomalous_volume(df):
    df = df.copy()
    df['anomalous_volume'] = 0
    vol_mean = df['volume'].rolling(window=20).mean()
    vol_std = df['volume'].rolling(window=20).std()
    z_score = (df['volume'] - vol_mean) / vol_std
    df['anomalous_volume'] = (z_score > 2).astype(int)
    return df

def build_features(df):
    df = df.copy()
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=10).std()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['macd'] = df['ema_9'] - df['ema_21']
    df['rsi'] = compute_rsi(df['close'], 14)
    df = detect_order_blocks(df)
    df = detect_liquidity_zones(df)
    df = detect_anomalous_volume(df)
    df = df.dropna()
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi