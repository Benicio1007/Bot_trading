import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOAgent(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config['model']['lstm_hidden_size'] * 2
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 1)  # Solo 1 logit para clasificación binaria
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits  # Retorna logits [B, 1]

    def compute_loss(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits.squeeze(), targets.float())
