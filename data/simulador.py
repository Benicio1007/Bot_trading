import torch
import yaml
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from training.utils import CustomDataset, load_config
from modelos.feature_extractor import CNNFeatureExtractor
from modelos.sequence_model import SequenceModel
from modelos.attention_layer import AttentionBlock
from modelos.drl_agent import PPOAgent
from torch.utils.data import DataLoader, Subset

def test_model():
    # Cargar configuración
    config_path = "data/config/config.yaml"
    config = load_config(config_path)
    device = torch.device(config['training']['device'])

    # Dataset de test: usar solo los últimos datos para evitar overlap con training
    # Nota: El año se configura en config.yaml (actualmente usando 2021)
    dataset = CustomDataset(config)
    
    # Usar solo el 10% más reciente para test (datos del año configurado)
    total_samples = len(dataset)
    test_size = int(total_samples * 0.1)
    test_indices = list(range(total_samples - test_size, total_samples))
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    print(f"🚀 Dataset de test cargado con {len(test_dataset)} ejemplos (últimos {test_size} del dataset).")
    print(f"📅 Usando datos del año configurado en config.yaml")

    # Modelos
    feature_extractor = CNNFeatureExtractor(config).to(device)
    sequence_model = SequenceModel(config).to(device)
    attention = AttentionBlock(config).to(device)
    agent = PPOAgent(config).to(device)

    # Cargar pesos entrenados
    checkpoint_path = "checkpoints/model_last.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_extractor.load_state_dict(checkpoint['feature_extractor'])
    sequence_model.load_state_dict(checkpoint['sequence_model'])
    attention.load_state_dict(checkpoint['attention'])
    agent.load_state_dict(checkpoint['agent'])

    feature_extractor.eval()
    sequence_model.eval()
    attention.eval()
    agent.eval()

    # Variables para métricas y simulación
    all_probs = []
    all_labels = []

    balance = 200.0  # capital inicial en USD
    leverage = 25
    risk_reward_ratio = 2  # 2:1

    # Variables para simulación
    total_wins = 0
    total_losses = 0
    trades = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            features = feature_extractor(X)
            sequence = sequence_model(features)
            context = attention(sequence)
            action_logits = agent(context)  # logits [batch, 3]

            # Usar la probabilidad de clase "Buy" (index 1)
            probs = torch.softmax(action_logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            print("Primeras 20 probabilidades de clase 1:", probs[:20])

            all_probs.extend(preds.tolist())
            all_labels.extend(y.cpu().numpy().tolist())

            # Simulación de operaciones
            for pred, actual in zip(preds, y.cpu().numpy()):
                if pred == 1:  # si el modelo decide entrar en largo
                    trades += 1
                    if actual == 1:
                        profit = balance * 0.02 * leverage * risk_reward_ratio
                        balance += profit
                        total_wins += 1
                    else:
                        loss = balance * 0.02 * leverage
                        balance -= loss
                        total_losses += 1

    # Métricas finales
    labels_np = np.array(all_labels)
    preds_np = np.array(all_probs)
    f1 = f1_score(labels_np, preds_np, zero_division='warn')
    accuracy = accuracy_score(labels_np, preds_np)
    precision = precision_score(labels_np, preds_np, zero_division='warn')
    recall = recall_score(labels_np, preds_np, zero_division='warn')
    cm = confusion_matrix(labels_np, preds_np)

    print("\n📊 Métricas finales en TEST (2025):")
    print(f"   F1:        {f1:.4f}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   Matriz de confusión:\n{cm}")

    print("\n💰 Simulación de trading:")
    print(f"   Número de operaciones: {trades}")
    print(f"   Operaciones ganadoras: {total_wins}")
    print(f"   Operaciones perdedoras: {total_losses}")
    print(f"   Balance final: ${balance:.2f}")

    # Opinión del entrenamiento
    if f1 >= 0.6 and precision >= 0.5:
        print("\n✅ Opinión: El modelo muestra potencial para operar en producción.")
    else:
        print("\n❌ Opinión: El modelo no generaliza bien; considera revisar entrenamiento o ajustar hiperparámetros.")

if __name__ == "__main__":
    test_model()

