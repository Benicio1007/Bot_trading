import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from features import build_features

class MLStrategy:
    def __init__(self, threshold=0.92, scale_pos_weight=1.5):
        self.threshold = threshold
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=10
        )
        self.trained = False

    def label_data(self, df):
        df = df.copy()
        df['future_return'] = df['close'].shift(-12) / df['close'] - 1
        df['label'] = 0
        df.loc[df['future_return'] > 0.003, 'label'] = 1
        df.loc[df['future_return'] < -0.003, 'label'] = -1
        return df.dropna()

    def train(self, df):
        df = build_features(df)
        labeled = self.label_data(df)
        # keep only binary labels
        labeled = labeled[labeled['label'] != 0]
        # features and target
        X = labeled[[
            'return','volatility','rsi','ema_9','ema_21','macd',
            'order_block','liquidity_zone','anomalous_volume','hour',
            'atr','stoch_rsi','vol_rel','ema_cross','trend_strength','market_trend'
        ]]
        y = labeled['label'].replace(-1, 0)

        if y.nunique() < 2:
            print("⚠️ No hay suficientes clases para entrenar.")
            return

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        print(f"✅ Accuracy: {acc*100:.2f}% | F1: {f1:.2f} | MCC: {mcc:.2f}")
        self.trained = True

    def generate_signals(self, df):
        df = build_features(df)
        if not self.trained:
            print("⚠️ Modelo no entrenado, no se generan señales.")
            df['signal'] = 0
            return df
        X = df[[
            'return','volatility','rsi','ema_9','ema_21','macd',
            'order_block','liquidity_zone','anomalous_volume','hour',
            'atr','stoch_rsi','vol_rel','ema_cross','trend_strength','market_trend'
        ]]
        proba = self.model.predict_proba(X)[:,1]
        df['proba_long'] = proba
        df['signal'] = 0
        df.loc[proba > self.threshold, 'signal'] = 1
        df.loc[proba < (1 - self.threshold), 'signal'] = -1
        return df