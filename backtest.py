import os
import pandas as pd
import torch
from datetime import datetime
from data.modelos.feature_extractor import CNNFeatureExtractor
from data.modelos.sequence_model import SequenceModel
from data.modelos.attention_layer import AttentionBlock
from data.modelos.drl_agent import PPOAgent
from data.prepare_data import compute_indicators
from data.training.utils import normalize, load_config
import numpy as np

# === CONFIGURACIÓN ===
config = load_config("data/config/config.yaml")
config['data']['features'] = ['open', 'high', 'low', 'close', 'volume']
features = config['data']['features']
device = torch.device(config['training']['device'])
sequence_length = config['data']['sequence_length']

# === MODELO ===
feature_extractor = CNNFeatureExtractor(config).to(device)
sequence_model = SequenceModel(config).to(device)
attention = AttentionBlock(config).to(device)
agent = PPOAgent(config).to(device)

checkpoint = torch.load("checkpoints/model_last.pt", map_location=device)
feature_extractor.load_state_dict(checkpoint['feature_extractor'])
sequence_model.load_state_dict(checkpoint['sequence_model'])
attention.load_state_dict(checkpoint['attention'])
agent.load_state_dict(checkpoint['agent'])

feature_extractor.eval()
sequence_model.eval()
attention.eval()
agent.eval()

# === FUNCIONES ===
def cargar_dataset(path):
    df = pd.read_csv(path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def generar_probabilidad(df, symbol_code, timeframe_code):
    df = compute_indicators(df)
    df['symbol_code'] = symbol_code
    df['timeframe_code'] = timeframe_code
    if len(df) < sequence_length + 1 or not all(f in df.columns for f in features):
        print("❌ No hay suficientes datos o faltan features.")
        return None
    x = normalize(df[features]).fillna(0)
    x_seq = x.iloc[-sequence_length:].values
    x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        f = feature_extractor(x_tensor)
        s = sequence_model(f)
        c = attention(s)
        logits = agent(c)
        prob = torch.sigmoid(logits[:, 1]).item()
        print(f"🧠 Probabilidad generada: {prob:.4f}")
        return prob


def ejecutar_backtest(file_path, symbol_code):
    df = cargar_dataset(file_path)
    capital = 1000
    balance = capital
    trades = []
    open_trade = None
    equity_curve = []

    for i in range(sequence_length, len(df)):
        ventana = df.iloc[i-sequence_length:i+1].copy()
        prob = generar_probabilidad(ventana, symbol_code, 1)  # timeframe 5m = 1
        if prob is None:
            continue

        close = ventana.iloc[-1]['close']
        timestamp = ventana.iloc[-1]['timestamp']
        volatility = ventana['close'].pct_change().std()
        TP = min(0.01, max(0.003, volatility * 3))
        SL = min(0.007, max(0.002, volatility * 2))

        if open_trade:
            change = (close - open_trade['entry']) / open_trade['entry'] if open_trade['side'] == 'buy' else (open_trade['entry'] - close) / open_trade['entry']
            if change >= TP:
                pnl = open_trade['qty'] * (close - open_trade['entry']) if open_trade['side'] == 'buy' else open_trade['qty'] * (open_trade['entry'] - close)
                trades.append((timestamp, 'TP', pnl))
                balance += pnl
                open_trade = None
            elif change <= -SL:
                pnl = open_trade['qty'] * (close - open_trade['entry']) if open_trade['side'] == 'buy' else open_trade['qty'] * (open_trade['entry'] - close)
                trades.append((timestamp, 'SL', pnl))
                balance += pnl
                open_trade = None
        else:
            if prob >= 0.52:
                qty = (balance * 0.2 * 30) / close
                open_trade = {'entry': close, 'side': 'buy', 'qty': qty}
            elif prob <= 0.48:
                qty = (balance * 0.2 * 30) / close
                open_trade = {'entry': close, 'side': 'sell', 'qty': qty}

        equity_curve.append(balance)

    # === MÉTRICAS ===
    pnl_total = balance - capital
    win_trades = [t for t in trades if t[2] > 0]
    loss_trades = [t for t in trades if t[2] <= 0]
    winrate = len(win_trades) / len(trades) * 100 if trades else 0
    avg_pnl = np.mean([t[2] for t in trades]) if trades else 0
    max_dd = max([max(equity_curve[:i]) - e for i, e in enumerate(equity_curve) if i > 0], default=0)

    return balance, trades, pnl_total, winrate, avg_pnl, max_dd

# === EJECUCIÓN ===
symbols = {
    'BTCUSDT_5m.csv': 0,
    'ETHUSDT_5m.csv': 1,
    'SOLUSDT_5m.csv': 2,
    'XRPUSDT_5m.csv': 3
}

for archivo, code in symbols.items():
    path = os.path.join("data", archivo)
    final_balance, operaciones, pnl, winrate, avg_pnl, max_dd = ejecutar_backtest(path, code)
    print(f"\n🧪 Backtest para {archivo}")
    print(f"Balance final: {final_balance:.2f} USD")
    print(f"Total PnL: {pnl:.2f} USD")
    print(f"Winrate: {winrate:.2f}%")
    print(f"Promedio PnL por trade: {avg_pnl:.2f} USD")
    print(f"Max Drawdown: {max_dd:.2f} USD")
    print(f"Cantidad de trades: {len(operaciones)}")
    for t in operaciones:
        print(f"{t[0]} | {t[1]} | PnL: {t[2]:.2f} USD")
