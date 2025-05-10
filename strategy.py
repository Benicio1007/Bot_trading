import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from features import build_features

class MLStrategy:
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    def label_data(self, df):
        df = df.copy()
        df['future_return'] = df['close'].shift(-12) / df['close'] - 1
        df['label'] = 0
        df.loc[df['future_return'] > 0.001, 'label'] = 1   # 🔧 Más sensible
        df.loc[df['future_return'] < -0.001, 'label'] = -1
        df = df.dropna()
        return df
    
    def train(self, df):
        df = build_features(df)
        df = self.label_data(df)
        df = df[df['label'] != 0]  # ✅ Filtrar etiquetas neutras
        X = df[['return', 'volatility', 'rsi', 'ema_9', 'ema_21', 'macd',
            'order_block', 'liquidity_zone', 'anomalous_volume']]
        y = df['label']
        # ✅ Asegurarse que X e y tengan el mismo largo ANTES de reemplazar
        X = X.iloc[-len(y):]
        y = y.iloc[-len(X):].replace(-1, 0)  # ✅ Ahora sí reemplazás -1 por 0
        if y.nunique() < 2:
            print("⚠️ No hay suficientes clases (0 y 1) para entrenar el modelo.")
            return
        print(f"📊 Distribución de etiquetas: {y.value_counts().to_dict()}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"📊 XGBoost Test Accuracy: {acc * 100:.2f}%")



    def generate_signals(self, df):
        df = build_features(df)
        X = df[['return', 'volatility', 'rsi', 'ema_9', 'ema_21', 'macd',
                'order_block', 'liquidity_zone', 'anomalous_volume']]
        try:
            preds_proba = self.model.predict_proba(X)
        except:
            print("⚠️ El modelo no fue entrenado. No se pueden generar señales.")
            df['signal'] = 0
            return df

        df['proba_long'] = preds_proba[:, 1]
        df['signal'] = 0
        df.loc[df['proba_long'] > self.threshold, 'signal'] = 1
        df.loc[df['proba_long'] < (1 - self.threshold), 'signal'] = -1
        print(f"📊 Señales generadas - LONG: {len(df[df['signal'] == 1])}, SHORT: {len(df[df['signal'] == -1])}")
        return df
