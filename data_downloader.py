import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 Backtest Visual - BTCUSDT 5m")

@st.cache_data
def cargar_datos():
    df = pd.read_csv("operaciones.csv")
    df["Ganancia"] = df["pnl"]
    return df

df = cargar_datos()

if df.empty:
    st.warning("El archivo 'operaciones.csv' está vacío.")
    st.stop()

# Métricas
st.header("📊 Métricas Generales")
ganadoras = df[df["Ganancia"] > 0]
perdedoras = df[df["Ganancia"] <= 0]

col1, col2, col3 = st.columns(3)
col1.metric("Total Operaciones", len(df))
col2.metric("Ganadoras", len(ganadoras))
col3.metric("Perdedoras", len(perdedoras))

win_rate = len(ganadoras) / len(df) * 100
st.metric("Win Rate", f"{win_rate:.2f}%")

# Evolución del capital
st.header("💰 Evolución del Capital")
capital = 1000
capitales = []
for g in df["Ganancia"]:
    capital += g
    capitales.append(capital)
df["Capital"] = capitales
df["Operación"] = df.index + 1

fig = px.line(df, x="Operación", y="Capital", title="Evolución del Capital")
st.plotly_chart(fig, use_container_width=True)

# Barras de ganancias
st.header("📊 Ganancia por operación")
fig2 = px.bar(df, x="Operación", y="Ganancia", color="type", title="Resultado por operación")
st.plotly_chart(fig2, use_container_width=True)

# Mostrar tabla
with st.expander("📋 Tabla de operaciones"):
    st.dataframe(df)
