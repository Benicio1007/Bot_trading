import torch
from torch.utils.data import DataLoader
import yaml
import os
import sys
from pathlib import Path
import torch.nn as nn
from math import pi, cos
import numpy as np
import argparse


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
from training.scheduler import WarmupCosineScheduler

def focal_loss(logits, targets, alpha=0.45, gamma=2.0):
    # Asegurar que las dimensiones sean correctas
    if logits.dim() == 0:
        logits = logits.unsqueeze(0)  # Agregar dimensión de batch si es necesario
    if targets.dim() == 0:
        targets = targets.unsqueeze(0)  # Agregar dimensión de batch si es necesario
    
    # Asegurar que ambos tensores tengan la misma forma
    if logits.shape != targets.shape:
        targets = targets.view_as(logits)
    
    logits = torch.clamp(logits, min=-20, max=20)  # evita logits extremos
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
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

# Paso 3: Pérdida con castigo a falsos positivos
def create_bce_criterion(device, labels):
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)]).to(device)
    print(f"[DEBUG] pos_weight dinámico: {pos_weight.item():.4f}")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def safe_bce_loss(logits, targets, criterion):
    """Maneja de forma segura las dimensiones para BCE loss"""
    # Asegurar que ambos tensores tengan la misma forma
    logits_flat = logits.squeeze()
    targets_flat = targets.float()
    
    # Si son escalares, agregar dimensión
    if logits_flat.dim() == 0:
        logits_flat = logits_flat.unsqueeze(0)
    if targets_flat.dim() == 0:
        targets_flat = targets_flat.unsqueeze(0)
    
    # Asegurar que tengan la misma forma
    if logits_flat.shape != targets_flat.shape:
        targets_flat = targets_flat.view_as(logits_flat)
    
    # Debug: mostrar shapes finales (solo en el primer batch)
    if hasattr(safe_bce_loss, 'debug_shown'):
        pass
    else:
        print(f"🔧 safe_bce_loss - Shapes finales:")
        print(f"   logits_flat: {logits_flat.shape}")
        print(f"   targets_flat: {targets_flat.shape}")
        safe_bce_loss.debug_shown = True
    
    return criterion(logits_flat, targets_flat)

def train_one_epoch(train_dataloader, feature_extractor, sequence_model, attention, agent, optimizer, criterion, device):
    feature_extractor.train()
    sequence_model.train()
    attention.train()
    agent.train()
    total_loss = 0
    all_probs = []
    all_labels = []
    all_logits = []
    total_correct = 0
    total_samples = 0
    for X, y in train_dataloader:
        X, y = X.to(device), y.to(device)
        features = feature_extractor(X)
        sequence = sequence_model(features)
        context = attention(sequence)
        logits = agent(context)
        loss = safe_bce_loss(logits, y, criterion)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # Acumula probabilidades, logits y etiquetas
        probs = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
        all_probs.extend(probs.tolist() if hasattr(probs, 'size') and probs.size > 1 else [float(probs)])
        all_logits.extend(logits.detach().cpu().numpy().squeeze().tolist() if hasattr(probs, 'size') and probs.size > 1 else [float(logits.detach().cpu().numpy().squeeze())])
        all_labels.extend(y.cpu().numpy().tolist())
        # Winrate (prob >= 0.51)
        total_correct += ((probs >= 0.51) == y.cpu().numpy()).sum() if hasattr(probs, 'size') and probs.size > 1 else int((probs >= 0.51) == y.cpu().numpy())
        total_samples += len(y)
    avg_loss = total_loss / len(train_dataloader)
    win_rate = total_correct / total_samples if total_samples else 0
    # Métricas
    labels_np = np.array(all_labels)
    if labels_np.dtype != int and labels_np.dtype != bool:
        labels_np = (labels_np >= 0.5).astype(int)
    probs_np  = np.array(all_probs)
    fixed_thr = 0.55
    final_preds = (probs_np >= fixed_thr).astype(int)
    accuracy  = accuracy_score(labels_np, final_preds)
    precision = precision_score(labels_np, final_preds, zero_division='warn')
    recall    = recall_score(labels_np, final_preds, zero_division='warn')
    f1_score_val = f1_score(labels_np, final_preds, zero_division='warn')
    cm        = confusion_matrix(labels_np, final_preds)
    print(f"\n📈 Métricas de entrenamiento (umbral fijo {fixed_thr:.2f}):")
    print(f"   F1:        {f1_score_val:.4f}")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   Win Rate:  {win_rate:.4f}")
    print(f"   Matriz de confusión:\n{cm}")
    print(f"[DEBUG] Media de predicciones (sigmoid): {np.mean(all_probs):.4f}")
    print(f"[DEBUG] Media de logits: {np.mean(all_logits):.4f}")
    return avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true', help='Congela y descongela capas en dos fases')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate para el optimizador')
    parser.add_argument('--epochs', type=int, default=None, help='Cantidad de épocas de entrenamiento')
    parser.add_argument('--resume', type=str, default=None, help='Ruta de checkpoint para reanudar entrenamiento')
    parser.add_argument('--save-as', type=str, default=None, help='Ruta para guardar el checkpoint final')
    args = parser.parse_args()
    config = load_config("data/config/config.yaml")
    device = torch.device(config['training']['device'])
    # División temporal del dataset
    train_indices, validation_indices = split_dataset_temporally(config, validation_split=0.2)
    # Crear datasets separados
    full_dataset = CustomDataset(config)
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(full_dataset, validation_indices)
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=config['training']['batch_size'],
                                 shuffle=True)
    validation_dataloader = DataLoader(validation_dataset,
                                      batch_size=config['training']['batch_size'],
                                      shuffle=False)
    feature_extractor = CNNFeatureExtractor(config).to(device)
    sequence_model = SequenceModel(config).to(device)
    attention = AttentionBlock(config).to(device)
    agent = PPOAgent(config).to(device)
    # Leer hiperparámetros del config
    pos_weight = torch.tensor([config['training'].get('pos_weight', 1.0)]).to(device)
    early_stopping_patience = config['training'].get('early_stopping_patience', 5)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Usar learning rate del argumento si se pasa, si no el del config
    lr = args.lr if args.lr is not None else config['training']['learning_rate']
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) +
                                 list(sequence_model.parameters()) +
                                 list(attention.parameters()) +
                                 list(agent.parameters()),
                                 lr=lr,
                                 weight_decay=0.0001)
    # Scheduler WarmupCosineScheduler
    scheduler_config = config['training'].get('scheduler', {})
    warmup_steps = scheduler_config.get('warmup_steps', 1)
    total_steps = scheduler_config.get('total_steps', config['training']['epochs'])
    min_lr = config['training'].get('min_learning_rate', 0.00009)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps, min_lr=min_lr)
    # Epochs
    epochs = args.epochs if args.epochs is not None else config['training']['epochs']
    # Checkpoint paths
    checkpoint_path = args.save_as if args.save_as is not None else "checkpoints/model_last.pt"
    resume_path = args.resume
    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        print(f"✅ Checkpoint encontrado en {resume_path}. Cargando modelo...")
        checkpoint = torch.load(resume_path, weights_only=False)
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        sequence_model.load_state_dict(checkpoint['sequence_model'])
        attention.load_state_dict(checkpoint['attention'])
        agent.load_state_dict(checkpoint['agent'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Checkpoint cargado exitosamente desde época {checkpoint.get('epoch', 0)}")
    if args.freeze:
        # 1. Congelar todo menos conv1 y fc2
        for name, param in feature_extractor.named_parameters():
            if not name.startswith("conv1"):
                param.requires_grad = False
        for name, param in agent.named_parameters():
            if not name.startswith("fc2"):
                param.requires_grad = False
        for name, param in sequence_model.named_parameters():
            param.requires_grad = False
        for name, param in attention.named_parameters():
            param.requires_grad = False
        # Optimizer solo con params entrenables
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(feature_extractor.parameters()) + list(sequence_model.parameters()) + list(attention.parameters()) + list(agent.parameters())),
                                     lr=1e-4, weight_decay=1e-4)
        print("🔒 Entrenando solo conv1 y fc2 (1 época)")
        train_one_epoch(train_dataloader, feature_extractor, sequence_model, attention, agent, optimizer, criterion, device)
        # 2. Descongelar todo y bajar LR
        for p in feature_extractor.parameters():
            p.requires_grad = True
        for p in sequence_model.parameters():
            p.requires_grad = True
        for p in attention.parameters():
            p.requires_grad = True
        for p in agent.parameters():
            p.requires_grad = True
        for g in optimizer.param_groups:
            g['lr'] = 3e-5
        print("🔓 Entrenando todo el modelo (1 época)")
        train_one_epoch(train_dataloader, feature_extractor, sequence_model, attention, agent, optimizer, criterion, device)
        print("✅ Fases de freezing/desfreezing completadas. Continúa entrenamiento normal si lo deseas.")
        return
    
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
    
    # Paso 1: Confirmar balance de clases
    print(f"📊 Balance de clases en training:")
    print(f"   Clase 0: {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"   Clase 1: {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")
    print(f"   Total: {len(labels)} muestras")
    
    # Verificar si hay clase 2 (no debería haber)
    if 2 in counts:
        print(f"   ⚠️ Clase 2 encontrada: {counts[2]} muestras")
    else:
        print(f"   ✅ Solo clases 0 y 1 (clasificación binaria correcta)")
    
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
    
    # Paso 5: Reiniciar la capa final (bias = 0, weight ~ N(0,0.01)) para quitar sesgo inicial
    agent.fc2.weight.data.normal_(0, 0.01)
    agent.fc2.bias.data.zero_()
    print("🔄 Capa final reiniciada: bias=0, weights~N(0,0.01)")
    
    # Paso 3: Crear criterio de pérdida con pos_weight fijo en 2.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.0]).to(device))
    
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) +
                                 list(sequence_model.parameters()) +
                                 list(attention.parameters()) +
                                 list(agent.parameters()),
                                 lr=config['training']['learning_rate'],
                                 weight_decay=0.0001)
    
    # Forzar learning rate inicial a 0.0008
    optimizer.param_groups[0]["lr"] = 0.0008
    
    Path("checkpoints").mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(parents=True, exist_ok=True)

    checkpoint_path = "checkpoints/model_last.pt"
    start_epoch = 0
    
    if os.path.exists(checkpoint_path):
        print("✅ Checkpoint encontrado. Cargando modelo...")
        # Cargar con weights_only=False para compatibilidad con checkpoints antiguos
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        feature_extractor.load_state_dict(checkpoint['feature_extractor'])
        sequence_model.load_state_dict(checkpoint['sequence_model'])
        attention.load_state_dict(checkpoint['attention'])
        agent.load_state_dict(checkpoint['agent'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Checkpoint cargado exitosamente desde época {checkpoint['epoch']}")

    log_file = "logs/training_log.csv"
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Epoch", "Loss", "WinRate", "ValidationF1", "LearningRate", "OptimalThreshold"])
    
    
    best_f1 = 0.0
    patience = 5  # cantidad de épocas sin mejora antes de parar
    patience_counter = 0

        # Entrenamiento
    for epoch in range(start_epoch, epochs):
        total_correct = 0
        total_samples = 0
        losses = []
        all_probs = []
        all_labels = []
        all_logits = []

        # ——— Loop de batches ———
        for X, y in train_dataloader:
            X, y = X.to(device), y.to(device)
            # No mixup en Q4 2024
            features = feature_extractor(X)
            sequence = sequence_model(features)
            context = attention(sequence)
            logits = agent(context)
            
            # Debug: mostrar shapes
            if epoch == 0 and len(losses) == 0:
                print(f"🔍 Debug shapes:")
                print(f"   X: {X.shape}")
                print(f"   y: {y.shape}")
                print(f"   logits: {logits.shape}")
                print(f"   logits.squeeze(): {logits.squeeze().shape}")
                print(f"   y.float(): {y.float().shape}")
                print(f"   Tipos - y: {y.dtype}, logits: {logits.dtype}")
                print(f"   Batch size: {X.shape[0]}")
            
            # Paso 3: Cálculo de pérdida con BCEWithLogitsLoss
            loss = safe_bce_loss(logits, y, criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # 2) Acumula probabilidades, logits y etiquetas
            probs = torch.sigmoid(logits).detach().cpu().numpy().squeeze()
            all_probs.extend(probs.tolist() if probs.size > 1 else [probs.item()])
            all_logits.extend(logits.detach().cpu().numpy().squeeze().tolist() if probs.size > 1 else [logits.detach().cpu().numpy().squeeze().item()])
            all_labels.extend(y.cpu().numpy().tolist())

            # Para win_rate (opcional): cuenta aciertos usando prob >= 0.51
            total_correct += ((probs >= 0.51) == y.cpu().numpy()).sum()
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
                logits = agent(context)
                
                probs = torch.sigmoid(logits).cpu().numpy().squeeze()
                val_probs.extend(probs.tolist() if probs.size > 1 else [probs.item()])
                val_labels.extend(y.cpu().numpy().tolist())
        
        # Métricas de validation
        val_labels_np = np.array(val_labels)
        val_probs_np = np.array(val_probs)
        
        # Paso 4: Umbral inicial ≥ 0.51
        initial_threshold = 0.55
        val_preds_initial = (val_probs_np >= initial_threshold).astype(int)
        
        # Encontrar umbral óptimo para validation (rango 0.51-0.55)
        optimal_threshold, optimal_f1 = find_optimal_threshold(val_probs_np, val_labels_np, threshold_range=(0.51, 0.55))
        val_preds = (val_probs_np >= optimal_threshold).astype(int)
        
        val_f1 = f1_score(val_labels_np, val_preds, zero_division='warn')
        val_accuracy = accuracy_score(val_labels_np, val_preds)
        val_precision = precision_score(val_labels_np, val_preds, zero_division='warn')
        val_recall = recall_score(val_labels_np, val_preds, zero_division='warn')
        
        print(f"\n📊 VALIDATION - Época {epoch+1}:")
        print(f"   Umbral inicial (0.51): F1={f1_score(val_labels_np, val_preds_initial, zero_division='warn'):.4f}")
        print(f"   Umbral óptimo ({optimal_threshold:.3f}): F1={val_f1:.4f}")
        print(f"   Accuracy:  {val_accuracy:.4f}")
        print(f"   Precision: {val_precision:.4f}")
        print(f"   Recall:    {val_recall:.4f}")
        
        # Volver a modo training
        feature_extractor.train()
        sequence_model.train()
        attention.train()
        agent.train()

       # ——— 3) Uso de umbral fijo manual 0.51 ———
        labels_np = np.array(all_labels)
        # Binarizar labels si son continuos (por mixup)
        if labels_np.dtype != int and labels_np.dtype != bool:
            labels_np = (labels_np >= 0.5).astype(int)
        probs_np  = np.array(all_probs)
        
        fixed_thr = 0.55
        
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
                'optimal_threshold': optimal_threshold,
                'best_val_f1': val_f1
            }, checkpoint_path)
            print(f"💾 Checkpoint guardado en época {epoch+1} (Validation F1={best_f1:.4f}, Umbral={optimal_threshold:.3f})")
        
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔️ Early stopping en época {epoch+1}. Mejor Validation F1: {best_f1:.4f}")
                break

        # ➕ Guardado de logs en CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss, win_rate, val_f1, lr, optimal_threshold])

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
                'optimal_threshold': optimal_threshold,
                'best_val_f1': val_f1
            }, checkpoint_path)
            print(f"💾 Checkpoint (backup) guardado en época {epoch+1}")

        # Debug: imprime media de predicciones y logits
        print(f"[DEBUG] Media de predicciones (sigmoid): {np.mean(all_probs):.4f}")
        print(f"[DEBUG] Media de logits: {np.mean(all_logits):.4f}")

if __name__ == "__main__":
    main()
