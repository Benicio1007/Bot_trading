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
from training.utils import CustomDataset, load_config, split_dataset_temporally
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter
from torch.utils.data import WeightedRandomSampler
import torch.nn.functional as F
import numpy as np



def focal_loss(logits, targets, alpha=0.45, gamma=2.0):
    logits = torch.clamp(logits, min=-20, max=20)  # evita logits extremos
    bce = F.binary_cross_entropy_with_logits(logits.squeeze(), targets, reduction='none')
    pt  = torch.exp(-bce)
    return (alpha * (1 - pt)**gamma * bce).mean()

def find_optimal_threshold(probs, labels, threshold_range=(0.40, 0.60), step=0.01):
    """
    Encuentra el umbral óptimo que maximiza F1 score
    """
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in np.arange(threshold_range[0], threshold_range[1] + step, step):
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(labels, preds, zero_division='warn')
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1



def main():
    config = load_config("data/config/config.yaml")
    device = torch.device(config['training']['device'])
    
    # División temporal del dataset
    train_indices, validation_indices = split_dataset_temporally(config, validation_split=0.2)
    
    # Crear datasets separados
    full_dataset = CustomDataset(config)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(full_dataset, validation_indices)
    
    print(f"Tamaño del dataset completo: {len(full_dataset)}")
    print(f"Tamaño del dataset de training: {len(train_dataset)}")
    print(f"Tamaño del dataset de validation: {len(validation_dataset)}")
    
    sample_X, sample_y = full_dataset[0]
    print(f"Shape de un sample: {sample_X.shape}, label: {sample_y}")
    
    # Crear dataloaders con weighted sampling para training
    labels = [int(full_dataset[i][1]) for i in train_indices]
    counts = Counter(labels)
    weight_per_class = {cls: 1.0/count for cls, count in counts.items()}
    sample_weights = [weight_per_class[int(full_dataset[i][1])] for i in train_indices]
    train_sampler = WeightedRandomSampler(sample_weights,
                                         num_samples=len(sample_weights),
                                         replacement=True)
    
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=config['training']['batch_size'],
                                 sampler=train_sampler,
                                 shuffle=False)
    
    validation_dataloader = DataLoader(validation_dataset,
                                      batch_size=config['training']['batch_size'],
                                      shuffle=False)

    feature_extractor = CNNFeatureExtractor(config).to(device)
    sequence_model = SequenceModel(config).to(device)
    attention = AttentionBlock(config).to(device)
    agent = PPOAgent(config).to(device)
    
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) +
                                 list(sequence_model.parameters()) +
                                 list(attention.parameters()) +
                                 list(agent.parameters()),
                                 lr=config['training']['learning_rate'],
                                 weight_decay=0.0005)
    
    # Ajustar learning rate a 0.0001
    optimizer.param_groups[0]["lr"] = 0.0001
    
    # Scheduler para reducir learning rate cuando F1 se estanca
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-5
    )
    
    print(f"🚀 Scheduler configurado:")
    print(f"   Tipo: ReduceLROnPlateau")
    print(f"   Modo: max (monitorea F1 score)")
    print(f"   Factor: 0.5 (reduce LR a la mitad)")
    print(f"   Patience: 3 épocas")
    print(f"   Min LR: 1e-5")
    print(f"   Learning Rate inicial: {optimizer.param_groups[0]['lr']:.6f}")
    print(f"   Weight Decay: {optimizer.param_groups[0]['weight_decay']:.6f}")
    print(f"   Early Stopping Patience: 5 épocas")
    
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
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    log_file = "logs/training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "WinRate", "ValidationF1", "LearningRate", "OptimalThreshold"])
    
    
    best_f1 = 0.0
    patience = 5  # cantidad de épocas sin mejora antes de parar
    patience_counter = 0

        # Entrenamiento
    for epoch in range(start_epoch, config['training']['epochs']):
        total_correct = 0
        total_samples = 0
        losses = []
        all_probs = []
        all_labels = []

        # ——— Loop de batches ———
        for X, y in train_dataloader:
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

        # ——— Evaluación en validation ———
        feature_extractor.eval()
        sequence_model.eval()
        attention.eval()
        agent.eval()
        
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for X, y in validation_dataloader:
                X, y = X.to(device), y.to(device)
                features = feature_extractor(X)
                sequence = sequence_model(features)
                context = attention(sequence)
                action_probs = agent(context)
                
                probs_pos = torch.sigmoid(action_probs[:,1]).cpu().numpy()
                val_probs.extend(probs_pos.tolist())
                val_labels.extend(y.cpu().numpy().tolist())
        
        # Métricas de validation
        val_labels_np = np.array(val_labels)
        val_probs_np = np.array(val_probs)
        
        # Encontrar umbral óptimo para validation
        optimal_threshold, optimal_f1 = find_optimal_threshold(val_probs_np, val_labels_np)
        val_preds = (val_probs_np >= optimal_threshold).astype(int)
        
        val_f1 = f1_score(val_labels_np, val_preds, zero_division='warn')
        val_accuracy = accuracy_score(val_labels_np, val_preds)
        val_precision = precision_score(val_labels_np, val_preds, zero_division='warn')
        val_recall = recall_score(val_labels_np, val_preds, zero_division='warn')
        
        print(f"\n📊 VALIDATION - Época {epoch+1} (umbral óptimo: {optimal_threshold:.3f}):")
        print(f"   F1:        {val_f1:.4f}")
        print(f"   Accuracy:  {val_accuracy:.4f}")
        print(f"   Precision: {val_precision:.4f}")
        print(f"   Recall:    {val_recall:.4f}")
        
        # Volver a modo training
        feature_extractor.train()
        sequence_model.train()
        attention.train()
        agent.train()

       # ——— 3) Uso de umbral fijo manual 0.41 ———
        labels_np = np.array(all_labels)
        probs_np  = np.array(all_probs)
        
        fixed_thr = 0.41
        
        final_preds = (probs_np >= fixed_thr).astype(int)
        
        accuracy  = accuracy_score(labels_np, final_preds)
        precision = precision_score(labels_np, final_preds, zero_division='warn')
        recall    = recall_score(labels_np, final_preds, zero_division='warn')
        f1_score_val = f1_score(labels_np, final_preds, zero_division='warn')
        cm        = confusion_matrix(labels_np, final_preds)
        print(f"\n📈 Época {epoch+1} — Métricas con umbral fijo {fixed_thr:.2f}:")
        print(f"   F1:        {f1_score_val:.4f}")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall:    {recall:.4f}")
        print(f"   Matriz de confusión:\n{cm}")
        # ——— 4) Checkpoint & early stopping basados en F1 de validation ———
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            torch.save({
        'epoch': epoch,
        'feature_extractor': feature_extractor.state_dict(),
        'sequence_model': sequence_model.state_dict(),
        'attention': attention.state_dict(),
        'agent': agent.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'optimal_threshold': optimal_threshold,
        'best_val_f1': val_f1
    }, checkpoint_path)
            print(f"💾 Checkpoint guardado en época {epoch+1} (Validation F1={best_f1:.4f}, Umbral={optimal_threshold:.3f})")
        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔️ Early stopping en época {epoch+1}. Mejor Validation F1: {best_f1:.4f}")
                break

        # Actualizar learning rate con scheduler basado en validation F1
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)
        new_lr = optimizer.param_groups[0]['lr']
        
        if new_lr != old_lr:
            print(f"🔄 Learning Rate reducido: {old_lr:.6f} → {new_lr:.6f}")
        else:
            print(f"📈 Learning Rate actual: {new_lr:.6f}")

        # ➕ Guardado de logs en CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss, win_rate, val_f1, new_lr, optimal_threshold])

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
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimal_threshold': optimal_threshold,
                'best_val_f1': val_f1
            }, checkpoint_path)
            print(f"💾 Checkpoint (backup) guardado en época {epoch+1}")

if __name__ == "__main__":
    main()
