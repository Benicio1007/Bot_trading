import pandas as pd
import numpy as np

def detect_order_blocks(df):
    df = df.copy()
    df['order_block'] = 0
    for i in range(2, len(df)):
        if df['high'].iloc[i] < df['high'].iloc[i-1] and df['low'].iloc[i] > df['low'].iloc[i-1]:
            df.at[df.index[i], 'order_block'] = 1
    return df

def detect_liquidity_zones(df):
    df = df.copy()
    df['liquidity_zone'] = 0
    window=20
    for i in range(window, len(df)):
        lows = df['low'].iloc[i-window:i]
        highs = df['high'].iloc[i-window:i]
        if df['low'].iloc[i] < lows.min()*1.001:
            df.at[df.index[i], 'liquidity_zone']=1
        elif df['high'].iloc[i] > highs.max()*0.999:
            df.at[df.index[i], 'liquidity_zone']=-1
    return df

def detect_anomalous_volume(df):
    df = df.copy()
    df['anomalous_volume'] = 0
    mean = df['volume'].rolling(20).mean()
    std = df['volume'].rolling(20).std()
    z = (df['volume'] - mean)/std
    df['anomalous_volume'] = (z>2).astype(int)
    return df

def compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain/avg_loss
    return 100 - (100/(1+rs))

def build_features(df):
    df = df.copy()
    # core indicators
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(10).std()
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21']=df['close'].ewm(span=21, adjust=False).mean()
    df['macd']=df['ema_9']-df['ema_21']
    df['rsi']=compute_rsi(df['close'],14)
    df['hour']=df['timestamp'].dt.hour
    # additional features
    df['atr']=df['high'].rolling(14).max()-df['low'].rolling(14).min()
    low_rsi = df['rsi'].rolling(14).min()
    high_rsi = df['rsi'].rolling(14).max()
    df['stoch_rsi'] = (df['rsi']-low_rsi)/(high_rsi-low_rsi)
    df['vol_rel']=df['volume']/df['volume'].rolling(20).mean()
    df['ema_50']=df['close'].ewm(span=50, adjust=False).mean()
    df['ema_cross'] = (df['ema_9']>df['ema_21']).astype(int)
    df['trend_strength']=df['ema_9']/df['ema_50']
    df['ema_100']=df['close'].ewm(span=100,adjust=False).mean()
    df['market_trend']=(df['close']>df['ema_100']).astype(int)
    # original detections
    df = detect_order_blocks(df)
    df = detect_liquidity_zones(df)
    df = detect_anomalous_volume(df)
    return df.dropna()
