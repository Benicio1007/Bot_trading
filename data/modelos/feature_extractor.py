import torch.nn as nn

class CNNFeatureExtractor(nn.Module):
    def __init__(self, config):
        super().__init__()
        in_channels = len(config['data']['features'])
        out_channels = config['model']['cnn_out_channels']
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config['model']['dropout'])

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [batch, seq_len, features] → [batch, features, seq_len]
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # → [batch, seq_len, cnn_out]
        return x
