from datetime import datetime

class LiveData:
    def __init__(self):
        self.pnl_total = 0
        self.equity = 1000  # Capital inicial
        self.operaciones = []

    def agregar_operacion(self, tipo, entrada, salida, resultado):
        self.operaciones.append({
            "Tipo": tipo,
            "Entrada": entrada,
            "Salida": salida,
            "Resultado": resultado,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        self.pnl_total += resultado
        self.equity += resultado

    def obtener_dataframe(self):
        import pandas as pd
        if not self.operaciones:
            return pd.DataFrame(columns=["Tipo", "Entrada", "Salida", "Resultado", "Timestamp", "Equity"])
        df = pd.DataFrame(self.operaciones)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values("Timestamp")
        df["Equity"] = df["Resultado"].cumsum() + 1000
        return df
