import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import sys
from pathlib import Path

# Agregar el directorio padre al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent))

from prepare_data import compute_indicators

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def normalize(df):
    return (df - df.mean()) / (df.std() + 1e-8)

class CustomDataset(Dataset):
    def __init__(self, config):
        self.sequence_length = config['data']['sequence_length']
        self.samples = []  # Cada elemento será (serie, label)
        self.length = 0
        self.config = config
        self.file_infos = []  # [(df, symbol, timeframe)]
        
        for input_cfg in config['data']['inputs']:
            file_path = input_cfg['path']
            if not os.path.exists(file_path):
                continue
            
            df = pd.read_csv(file_path)
            df = compute_indicators(df)

            # Agregar candle_type
            df['candle_type'] = 0
            df.loc[df['close'] > df['open'], 'candle_type'] = 1
            df.loc[df['close'] < df['open'], 'candle_type'] = -1

            # Agregar symbol_code y timeframe_code
            symbol_map = {'BTCUSDT': 0, 'ETHUSDT': 1,}
            symbol = input_cfg['symbol']
            df['symbol_code'] = symbol_map.get(symbol, -1)
            df['timeframe_code'] = input_cfg['timeframe']

            # Esta función cubre features clásicos y nuevos (como bollinger_width, momentum_5, rsi_cross, previous_signal)
            def feature_present(col):
                return (
                    col in df.columns or
                    f"{col}_1m" in df.columns or
                    f"{col}_5m" in df.columns
                )
            missing = [col for col in config['data']['features'] if not feature_present(col)]
            print(f"Archivo: {file_path}")
            print(f"Features presentes: {list(df.columns)}")
            if missing:
                raise ValueError(f"🚨 Faltan features requeridos: {missing} en archivo {file_path}")
            else:
                print("Todos los features requeridos están presentes.\n")
            if not all(feature_present(col) for col in config['data']['features']):
                continue

            if df.empty:
                raise ValueError(f"🚨 El archivo {file_path} quedó vacío después de eliminar NaNs. Revisa la calidad de tus datos o ajusta tus indicadores.")

            
            df = normalize(df[config['data']['features']])
            self.file_infos.append((df, symbol, input_cfg['timeframe']))
            self.length += max(0, len(df) - self.sequence_length - 1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Encuentra a qué archivo pertenece el idx
        offset = 0
        for df, symbol, timeframe in self.file_infos:
            n = max(0, len(df) - self.sequence_length - 1)
            if idx < offset + n:
                i = idx - offset
                seq = df.iloc[i:i+self.sequence_length].values
                if np.isnan(seq).any() or np.isinf(seq).any():
                    raise ValueError(f"Secuencia inválida: contiene NaN o inf en idx={idx}")
                current_price = df.iloc[i + self.sequence_length - 1]['close']  # 'close' actual
                future_price = df.iloc[i + self.sequence_length]['close']       # 'close' siguiente
                change = (future_price - current_price) / (current_price + 1e-9)
                label = 1 if change > 0.003 else 0
                return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            offset += n
        raise IndexError('Index out of range in CustomDataset')

def split_dataset_temporally(config, validation_split=0.2):
    """
    Divide el dataset temporalmente: los datos más recientes van a validación
    """
    dataset = CustomDataset(config)
    
    # Ordenar por timestamp (asumiendo que los datos están ordenados cronológicamente)
    total_samples = len(dataset)
    validation_size = int(total_samples * validation_split)
    train_size = total_samples - validation_size
    
    # Crear índices para train y validation
    train_indices = list(range(train_size))
    validation_indices = list(range(train_size, total_samples))
    
    print(f"📊 División temporal del dataset:")
    print(f"   Training: {train_size} muestras (primeros {100-validation_split*100:.0f}%)")
    print(f"   Validation: {validation_size} muestras (últimos {validation_split*100:.0f}%)")
    
    return train_indices, validation_indices