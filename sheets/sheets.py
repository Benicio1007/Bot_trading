import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# ========== CONFIG ========== #
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]
CREDENTIALS_FILE = "sheets/credenciales.json"
SPREADSHEET_NAME = "Log de Trades_bot"
# ============================ #

def upload_to_sheets():
    # Autenticación
    creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=SCOPE)
    client = gspread.authorize(creds)

    # Leer el CSV
    df = pd.read_csv("sheets/operaciones.csv")

    # Abrir la hoja de cálculo
    spreadsheet = client.open(SPREADSHEET_NAME)

    # Eliminar hoja si existe y crear nueva
    try:
        spreadsheet.del_worksheet(spreadsheet.worksheet("Trades"))
    except:
        pass
    sheet = spreadsheet.add_worksheet(title="Trades", rows="1000", cols="20")

    # Subir encabezado y datos
    sheet.update([df.columns.values.tolist()] + df.values.tolist())

    # Crear hoja de resumen
    try:
        spreadsheet.del_worksheet(spreadsheet.worksheet("Resumen"))
    except:
        pass
    resumen = spreadsheet.add_worksheet(title="Resumen", rows="30", cols="10")

    resumen.update("A1", [["Métrica", "Valor"]])
    resumen.update("A2", [
        ["Total Trades", f"=COUNTA(Trades!A2:A)"],
        ["Ganancia Total", f"=SUM(Trades!F2:F)"],
        ["Ganancia Promedio", f"=AVERAGE(Trades!F2:F)"],
        ["Winrate %", f"=COUNTIF(Trades!F2:F,\">0\") / COUNTA(Trades!F2:F)"],
        ["Total Ganadoras", f"=COUNTIF(Trades!F2:F,\">0\")"],
        ["Total Perdedoras", f"=COUNTIF(Trades!F2:F,\"<0\")"]
    ])

    # Crear gráficos con Google Sheets API (a través de gspread)
    from gspread_formatting import batch_updater, charts

    with batch_updater(spreadsheet) as batch:
        # 🎯 Gráfico de Ganancia Acumulada
        chart = charts.add_chart(
            sheet,
            "Ganancia Acumulada",
            "LINE",
            start_anchor_cell="H2",
            end_anchor_cell="N20",
            domain=("A2", f"A{len(df)+1}"),
            series=[("G2", f"G{len(df)+1}")],  # Asume columna G = Ganancia Acumulada
            headers=True,
        )

        # 🟢 Pie Chart de ganadoras vs perdedoras
        charts.add_chart(
            resumen,
            "Ganadoras vs Perdedoras",
            "PIE",
            start_anchor_cell="D2",
            end_anchor_cell="H16",
            domain=("A5", "A6"),  # Nombres
            series=[("B5", "B6")],  # Valores
            headers=False,
        )

        # 📊 Column chart Ganancia por operación
        charts.add_chart(
            sheet,
            "Ganancia por Operación",
            "COLUMN",
            start_anchor_cell="H25",
            end_anchor_cell="N40",
            domain=("A2", f"A{len(df)+1}"),
            series=[("F2", f"F{len(df)+1}")],
            headers=True,
        )

    print("✅ Datos y gráficos subidos a Google Sheets")
