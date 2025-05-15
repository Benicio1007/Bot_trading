import torch
from torch.utils.data import DataLoader
import yaml
import os
from data.modelos.feature_extractor import CNNFeatureExtractor
from data.modelos.sequence_model import SequenceModel
from data.modelos.attention_layer import AttentionBlock
from data.modelos.drl_agent import PPOAgent
from data.training.utils import CustomDataset, load_config
import csv
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def main():
    config = load_config("data/config/config.yaml")
    device = torch.device(config['training']['device'])
    dataset = CustomDataset(config)
    print(dataset.X.shape)
    config['data']['features'] = list(range(dataset.X.shape[2]))
    dataloader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

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
    
    best_loss = float('inf')
    patience = 10  # cantidad de épocas sin mejora antes de parar
    patience_counter = 0

    # Entrenamiento
    for epoch in range(start_epoch, config['training']['epochs']):
        total_correct = 0
        total_samples = 0
        losses = []
        all_preds = []
        all_labels = []
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            features = feature_extractor(X)
            sequence = sequence_model(features)
            context = attention(sequence)
            action_probs = agent(context)

            loss = agent.compute_loss(action_probs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            predicted = torch.argmax(action_probs, dim=1).float()
            correct = (predicted == y.float()).sum().item()
            total_correct += correct
            total_samples += len(y)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
        
        avg_loss = sum(losses) / len(losses)
        win_rate = total_correct / total_samples  # esto es lo que ya mostrabas y se mantiene

        # Guardar mejor modelo si mejora el loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'feature_extractor': feature_extractor.state_dict(),
                'sequence_model': sequence_model.state_dict(),
                'attention': attention.state_dict(),
                'agent': agent.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            print(f"💾 Checkpoint guardado en época {epoch + 1}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⛔️ Early stopping activado en época {epoch+1}. Mejor loss: {best_loss:.4f}")
                break

        # Guardar en CSV
        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, avg_loss, win_rate])

        # 🔍 Mostrar métricas detalladas cada 2 épocas (cambiá el 2 si querés otro intervalo)
        if (epoch + 1) % 2 == 0:
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
            recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
            f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
            cm = confusion_matrix(all_labels, all_preds)

            print(f"\n📈 Métricas adicionales en época {epoch + 1}:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print("   Matriz de confusión:")
            print(cm)

        # ✅ Imprimir resultado cada época (esto no cambia)
        print(f"📊 Epoch {epoch+1}: Loss = {avg_loss:.4f}, Win Rate = {win_rate:.2%}")

        # Guardar cada 2 épocas o al final
        if (epoch + 1) % 2 == 0 or (epoch + 1) == config['training']['epochs']:
            torch.save({
                'epoch': epoch,
                'feature_extractor': feature_extractor.state_dict(),
                'sequence_model': sequence_model.state_dict(),
                'attention': attention.state_dict(),
                'agent': agent.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            print(f"💾 Checkpoint guardado en época {epoch + 1}")

if __name__ == "__main__":
    main()
