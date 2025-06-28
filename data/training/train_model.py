import torch
from torch.utils.data import DataLoader
import yaml
import os
import sys
from pathlib import Path

# Agregar el directorio padre al path para poder importar los módulos
sys.path.append(str(Path(__file__).parent.parent))

from modelos.feature_extractor import CNNFeatureExtractor
from modelos.sequence_model import SequenceModel
from modelos.attention_layer import AttentionBlock
from modelos.drl_agent import PPOAgent
from training.utils import CustomDataset, load_config
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import numpy as np



def focal_loss(logits, targets, alpha=0.45, gamma=2.0):
   bce = F.binary_cross_entropy_with_logits(logits.squeeze(), targets, reduction='none')
   pt  = torch.exp(-bce)
   return (alpha * (1 - pt)**gamma * bce).mean()


def main():
    config = load_config("data/config/config.yaml")
    device = torch.device(config['training']['device'])
    dataset = CustomDataset(config)
    print(f"Tamaño del dataset: {len(dataset)}")
    sample_X, sample_y = dataset[0]
    print(f"Shape de un sample: {sample_X.shape}, label: {sample_y}")
    
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)    
  

    labels = [int(y) for _, y in dataset]               
    counts = Counter(labels)
    weight_per_class = {cls: 1.0/count for cls, count in counts.items()}
    sample_weights = [weight_per_class[int(y)] for _, y in dataset]
    sampler = WeightedRandomSampler(sample_weights,
                                   num_samples=len(sample_weights),
                                   replacement=True)
    dataloader = DataLoader(dataset,
                            batch_size=config['training']['batch_size'],
                            sampler=sampler,
                            shuffle=False)  # shuffle=False porque usamos sampler

    feature_extractor = CNNFeatureExtractor(config).to(device)
    sequence_model = SequenceModel(config).to(device)
    attention = AttentionBlock(config).to(device)
    agent = PPOAgent(config).to(device)
    
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) +
                                 list(sequence_model.parameters()) +
                                 list(attention.parameters()) +
                                 list(agent.parameters()),
                                 lr=config['training']['learning_rate'])
    
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    checkpoint_path = "checkpoints/model_last.pt"
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print("✅ Checkpoint encontrado. Cargando modelo...")
        checkpoint = torch.load(checkpoint_path)
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        sequence_model.load_state_dict(checkpoint['sequence_model'])
        attention.load_state_dict(checkpoint['attention'])
        agent.load_state_dict(checkpoint['agent'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1

    log_file = "logs/training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "WinRate"])
    
    
    best_f1 = 0.0
    patience = 10  # cantidad de épocas sin mejora antes de parar
    patience_counter = 0

        # Entrenamiento
    for epoch in range(start_epoch, config['training']['epochs']):
        total_correct = 0
        total_samples = 0
        losses = []
        all_probs = []
        all_labels = []

        # ——— Loop de batches ———
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            features = feature_extractor(X)
            sequence = sequence_model(features)
            context = attention(sequence)
            action_probs = agent(context)  # shape [B,2] logits
            
            # 1) Cálculo de pérdida con Focal Loss
            loss = focal_loss(action_probs[:,1], y.float(), alpha=0.25, gamma=2.0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # 2) Acumula probabilidades y etiquetas
            probs_pos = torch.sigmoid(action_probs[:,1]).detach().cpu().numpy()
            all_probs.extend(probs_pos.tolist())
            all_labels.extend(y.cpu().numpy().tolist())

            # Para win_rate (opcional): cuenta aciertos usando prob >= 0.5
            total_correct += (probs_pos.round() == y.cpu().numpy()).sum()
            total_samples += len(y)

        avg_loss = sum(losses) / len(losses)
        win_rate = total_correct / total_samples

       # ——— 3) Uso de umbral fijo manual 0.41 ———
        labels_np = np.array(all_labels)
        probs_np  = np.array(all_probs)
        
        fixed_thr = 0.41
        
        final_preds = (probs_np >= fixed_thr).astype(int)
        
        accuracy  = accuracy_score(labels_np, final_preds)
        precision = precision_score(labels_np, final_preds, zero_division=0)
        recall    = recall_score(labels_np, final_preds, zero_division=0)
        f1_score_val = f1_score(labels_np, final_preds, zero_division=0)
        cm        = confusion_matrix(labels_np, final_preds)
        print(f"\n📈 Época {epoch+1} — Métricas con umbral fijo {fixed_thr:.2f}:")
        print(f"   F1:        {f1_score_val:.4f}")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   Matriz de confusión:\n{cm}")
        # ——— 4) Checkpoint & early stopping basados en F1 con umbral fijo ———
        if f1_score_val > best_f1:
            best_f1 = f1_score_val
            patience_counter = 0
            torch.save({
        'epoch': epoch,
        'feature_extractor': feature_extractor.state_dict(),
        'sequence_model': sequence_model.state_dict(),
        'attention': attention.state_dict(),
        'agent': agent.state_dict(),
        'optimizer': optimizer.state_dict()
    }, checkpoint_path)
            print(f"💾 Checkpoint guardado en época {epoch+1} (F1={best_f1:.4f})")
        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔️ Early stopping en época {epoch+1}. Mejor F1: {best_f1:.4f}")
                break

        # ➕ Guardado de logs en CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss, win_rate])

        # ——— 5) Impresión estándar cada época ———
        print(f"📊 Epoch {epoch+1}: Loss = {avg_loss:.4f}, Win Rate = {win_rate:.2%}")

        # ——— 6) Backup periódico ———
        if (epoch + 1) % 2 == 0 or (epoch + 1) == config['training']['epochs']:
            torch.save({
                'epoch': epoch,
                'feature_extractor': feature_extractor.state_dict(),
                'sequence_model': sequence_model.state_dict(),
                'attention': attention.state_dict(),
                'agent': agent.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            print(f"💾 Checkpoint (backup) guardado en época {epoch+1}")

if __name__ == "__main__":
    main()
