import torch.nn as nn

class SequenceModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        input_size = config['model']['cnn_out_channels']
        hidden_size = config['model']['lstm_hidden_size']
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=config['model']['lstm_layers'],
                            batch_first=True,
                            bidirectional=True)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        output, _ = self.lstm(x)
        return output  # [batch, seq_len, hidden_size * 2]
