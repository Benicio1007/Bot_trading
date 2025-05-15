import os
import yaml
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from data.prepare_data import compute_indicators

def load_config(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)

def normalize(df):
    return (df - df.mean()) / (df.std() + 1e-8)

class CustomDataset(Dataset):
    def __init__(self, config):
        self.sequence_length = config['data']['sequence_length']
        self.data = []
        
        for input_cfg in config['data']['inputs']:
            file_path = input_cfg['path']
            if not os.path.exists(file_path):
                continue
            
            df = pd.read_csv(file_path)

            df = compute_indicators(df)

            
            if not all(col in df.columns for col in config['data']['features'][:5]):
                continue
            
            df = normalize(df[config['data']['features'][:5]])
            self.data.append(df.values)

        self.X, self.y = self._build_sequences()

    def _build_sequences(self):
        sequences = []
        targets = []
        for series in self.data:
            for i in range(len(series) - self.sequence_length - 1):  # aseguramos acceso a i + sequence_length
                seq = series[i:i+self.sequence_length]
                
                current_price = series[i + self.sequence_length - 1][3]  # 'close' actual
                future_price = series[i + self.sequence_length][3]       # 'close' siguiente
                change = (future_price - current_price) / (current_price + 1e-9)
                label = 1 if change > 0.003 else 0  # 0.3% de subida como umbral positivo
                
                sequences.append(seq)
                targets.append(label)
                
        sequences = np.array(sequences)
        targets = np.array(targets)
        
        return torch.tensor(sequences, dtype=torch.float32), torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]