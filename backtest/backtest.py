import pandas as pd
import numpy as np
from strategy import MLStrategy


PARAMS = {
    'stop_loss_pct': 0.01,
    'take_profit_pct': 0.02,
    'fee': 0.0004,
    'initial_capital': 1000,
    'leverage': 10,
    'risk_per_trade_pct': 0.1,
    'data_file': 'data/BTCUSDT_5m.csv',
    'prob_threshold': 0.86,
}

def run_backtest(params):
    df = pd.read_csv(params['data_file'])
    'prob_threshold': 0.86,
    strat.train(df)
    df_signals = strat.generate_signals(df)

    long_signals = len(df_signals[df_signals['signal'] == 1])
    short_signals = len(df_signals[df_signals['signal'] == -1])
    print(f"📊 Señales generadas - LONG: {long_signals}, SHORT: {short_signals}")

    capital = params['initial_capital']
    peak = capital
    drawdown = 0
    trades = []
    total_comm = 0
    active_trade = None

    for i, row in df_signals.iterrows():
        if active_trade:
            price = row['close']
            t = active_trade
            hit_tp = price >= t['tp'] if t['type'] == 'LONG' else price <= t['tp']
            hit_sl = price <= t['sl'] if t['type'] == 'LONG' else price >= t['sl']
            if hit_tp or hit_sl:
                exit_price = t['tp'] if hit_tp else t['sl']
                raw_pnl = (exit_price - t['entry']) * t['size'] * (1 if t['type'] == 'LONG' else -1)
                comm_exit = abs(exit_price * t['size']) * params['fee']
                pnl = raw_pnl - comm_exit
                capital += pnl
                total_comm += comm_exit
                trades.append({**t, 'exit': exit_price, 'pnl': pnl, 'comm': t['comm'] + comm_exit})
                peak = max(peak, capital)
                drawdown = max(drawdown, peak - capital)
                active_trade = None

        if not active_trade and row['signal'] != 0:
            risk_amount = capital * params['risk_per_trade_pct']
            entry = row['close']
            sl_price = entry * (1 - params['stop_loss_pct'] if row['signal'] == 1 else 1 + params['stop_loss_pct'])
            stop_loss_distance = abs(entry - sl_price)
            size = (risk_amount * params['leverage']) / stop_loss_distance if stop_loss_distance > 0 else 0

            max_position_value = capital * params['leverage']
            position_value = size * entry
            if position_value > max_position_value:
                size = max_position_value / entry
                position_value = size * entry

            if size <= 0:
                continue

            comm_entry = position_value * params['fee']
            capital -= comm_entry
            total_comm += comm_entry
            tp_price = entry * (1 + params['take_profit_pct'] if row['signal'] == 1 else 1 - params['take_profit_pct'])
            active_trade = {
                'type': 'LONG' if row['signal'] == 1 else 'SHORT',
                'entry': entry,
                'size': size,
                'sl': sl_price,
                'tp': tp_price,
                'exit': None,
                'pnl': 0,
                'comm': comm_entry
            }

    df_tr = pd.DataFrame(trades)
    df_tr.to_csv('operaciones.csv', index=False)

    if df_tr.empty:
        print("\n⚠️ No se ejecutaron trades. Revisa el umbral, señales o condiciones de entrada.")
        return

    wins = df_tr[df_tr['pnl'] > 0]
    metrics = {
        'trades': len(df_tr),
        'win_rate': len(wins) / len(df_tr) * 100 if len(df_tr) > 0 else 0,
        'final_cap': capital,
        'net': capital - params['initial_capital'],
        'max_dd': drawdown,
        'total_comm': total_comm
    }

    emoji = '🔥' if metrics['win_rate'] > 70 else ('⚠️' if metrics['win_rate'] > 50 else '💩')

    print("\n📈💥 BACKTEST RESULT 💥📉")
    print(f"Total Trades: {metrics['trades']}")
    print(f"Win Rate: {metrics['win_rate']:.2f}% {emoji}")
    print(f"Final Capital: ${metrics['final_cap']:.2f}")
    print(f"Net Profit: ${metrics['net']:.2f}")
    print(f"Max Drawdown: ${metrics['max_dd']:.2f}")
    print(f"Total Commission Paid: ${metrics['total_comm']:.2f}")

if __name__ == '__main__':
    run_backtest(PARAMS)
