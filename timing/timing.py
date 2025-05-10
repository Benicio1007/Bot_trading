import time
import csv
from datetime import datetime
import os

class Timer:
    def __init__(self, save_to_csv=True, csv_path="timing/timings.csv"):
        self.timestamps = {}
        self.save_to_csv = save_to_csv
        self.csv_path = csv_path

        # Crear carpeta si no existe
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        # Encabezado del CSV si no existe
        if self.save_to_csv and not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "etiqueta", "tiempo_segundos"])

    def start(self, label: str):
        self.timestamps[label] = time.perf_counter()

    def stop(self, label: str):
        if label in self.timestamps:
            elapsed = time.perf_counter() - self.timestamps[label]
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"⏱️ Tiempo en '{label}': {elapsed:.6f} segundos")

            if self.save_to_csv:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                     writer = csv.writer(f)
                     writer.writerow([timestamp, label, f"{elapsed:.6f}"])
            del self.timestamps[label]
        else:
            print(f"⚠️ Timer '{label}' no fue iniciado.")
