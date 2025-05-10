import pandas as pd
from datetime import datetime
from .enviar_mail import enviar_email

def generar_y_enviar_informe():
    try:
        df = pd.read_csv("operaciones.csv")

        ahora = datetime.now()
        periodo = ahora.strftime("%B %Y")

        df["Timestamp Entrada"] = pd.to_datetime(df["Timestamp Entrada"])
        df_mes = df[df["Timestamp Entrada"].dt.month == ahora.month]

        if df_mes.empty:
            print("📭 No hay operaciones este mes.")
            return

        total_ops = len(df_mes)
        ganadas = df_mes[df_mes["Resultado"] == "GANANCIA"].shape[0]
        perdidas = df_mes[df_mes["Resultado"] == "PERDIDA"].shape[0]
        win_rate = 100 * ganadas / total_ops if total_ops > 0 else 0
        pnl_total = df_mes["PNL Neto"].sum()
        comisiones = df_mes["Comisión"].sum()

        filas_html = ""
        for _, row in df_mes.iterrows():
            filas_html += f"""
            <tr>
                <td>{row['Tipo']}</td>
                <td>{row['Activo']}</td>
                <td>{row['Entrada']}</td>
                <td>{row['Salida']}</td>
                <td>{row['Resultado']}</td>
                <td>{row['PNL Neto']}</td>
                <td>{row['Comisión']}</td>
                <td>{row['Qty']}</td>
                <td>{row['Timestamp Entrada']}</td>
                <td>{row['Timestamp Salida']}</td>
            </tr>
            """

        with open("informe/informe_mensual.html", "r", encoding="utf-8") as f:
            html_template = f.read()

        cuerpo = html_template.format(
            periodo=periodo,
            total_ops=total_ops,
            ganadas=ganadas,
            perdidas=perdidas,
            win_rate=win_rate,
            pnl_total=pnl_total,
            comisiones=comisiones,
            filas=filas_html
        )

        enviar_email(f"📊 Informe mensual de trading - {periodo}", cuerpo)
    except Exception as e:
        print(f"❌ Error generando informe mensual: {e}")