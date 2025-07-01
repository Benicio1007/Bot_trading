import torch
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, precision_recall_curve
from training.utils import CustomDataset, load_config
from modelos.feature_extractor import CNNFeatureExtractor
from modelos.sequence_model import SequenceModel
from modelos.attention_layer import AttentionBlock
from modelos.drl_agent import PPOAgent
from torch.utils.data import DataLoader

# Configuración
config_path = "data/config/config.yaml"
config = load_config(config_path)
device = torch.device(config['training']['device'])

# Dataset de test
dataset = CustomDataset(config)
total_samples = len(dataset)
test_size = int(total_samples * 0.1)
test_indices = list(range(total_samples - test_size, total_samples))
test_dataset = torch.utils.data.Subset(dataset, test_indices)
dataloader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False)

# Modelos
feature_extractor = CNNFeatureExtractor(config).to(device)
sequence_model = SequenceModel(config).to(device)
attention = AttentionBlock(config).to(device)
agent = PPOAgent(config).to(device)

# Cargar pesos
checkpoint_path = "checkpoints/model_last.pt"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"No se encontró el checkpoint en {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
feature_extractor.load_state_dict(checkpoint['feature_extractor'])
sequence_model.load_state_dict(checkpoint['sequence_model'])
attention.load_state_dict(checkpoint['attention'])
agent.load_state_dict(checkpoint['agent'])

feature_extractor.eval()
sequence_model.eval()
attention.eval()
agent.eval()

all_probs, all_labels = [], []

# Inference
with torch.no_grad():
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        features = feature_extractor(X)
        sequence = sequence_model(features)
        context = attention(sequence)
        logits = agent(context).squeeze()
        probs = torch.sigmoid(logits)

        probs_np = probs.cpu().numpy()
        if probs_np.size == 1:
            all_probs.append(probs_np.item())
        else:
            all_probs.extend(probs_np.tolist())
        all_labels.extend(y.cpu().numpy().tolist())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

# -------- 1️⃣ Umbral óptimo por F1 --------
best_f1, best_t_f1 = 0, 0.5
for t in np.linspace(0.3, 0.75, 100):
    preds = (all_probs >= t).astype(int)
    f1 = f1_score(all_labels, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_t_f1 = t

# -------- 2️⃣ Primer umbral con precision ≥ 0.50 --------
prec, rec, thresh = precision_recall_curve(all_labels, all_probs)
try:
    idx = np.where(prec >= 0.50)[0][0]
    best_t_prec = thresh[idx]
except IndexError:
    best_t_prec = 0.55  # fallback
print(f"\n🧠 Mejor umbral por F1 = {best_t_f1:.3f} (F1 = {best_f1:.4f})")
print(f"🎯 Primer umbral con Precision ≥ 50% = {best_t_prec:.3f}")

# -------- 3️⃣ Umbral balanceado (precision ≥ 0.50 y recall ≥ 0.50) --------
print("\n🔍 Buscando umbral intermedio balanceado (precision y recall ≥ 0.50):")
mejor = {'umbral': None, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
for t in np.linspace(0.60, 0.75, 100):
    preds = (all_probs >= t).astype(int)
    p = precision_score(all_labels, preds, zero_division='warn')
    r = recall_score(all_labels, preds, zero_division='warn')
    f = f1_score(all_labels, preds, zero_division='warn')
    if p >= 0.50 and r >= 0.50 and f > mejor['f1']:
        mejor = {'umbral': t, 'precision': p, 'recall': r, 'f1': f}

if mejor['umbral'] is not None:
    print(f"✅ Umbral balanceado encontrado: {mejor['umbral']:.3f} (Precision={mejor['precision']:.3f}, Recall={mejor['recall']:.3f}, F1={mejor['f1']:.4f})")
    umbral_final = mejor['umbral']
else:
    print("❌ No se encontró un umbral balanceado en el rango 0.60–0.75")
    umbral_final = best_t_prec  # fallback conservador

# -------- 4️⃣ Evaluación con umbral balanceado final --------
print("\n📊 MÉTRICAS CON UMBRAL BALANCEADO:")
preds_final = (all_probs >= umbral_final).astype(int)
print(classification_report(all_labels, preds_final, digits=4))
print("📉 Matriz de Confusión:\n", confusion_matrix(all_labels, preds_final))
