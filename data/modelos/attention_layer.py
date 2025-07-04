import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config['model']['lstm_hidden_size'] * 2
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim,
                                               num_heads=config['model']['attention_heads'],
                                               batch_first=True)
        
        for m in self.modules():
            if isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return attn_output.mean(dim=1)  # Context vector (mean pooled)
