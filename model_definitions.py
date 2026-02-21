# File: model_definitions.py
# 优化：统一模型定义，添加Dropout防过拟合，参数化input_size。合并TFT和NBeats为可复用类。添加注意力机制简单版提升TFT时序能力。

import torch
import torch.nn as nn

class EnhancedTFT(nn.Module):
    def __init__(self, input_size, hidden_size=128, output_size=1, dropout=0.2):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.attention = nn.MultiheadAttention(hidden_size, num_heads=4, dropout=dropout)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B,F) 或 (B,T,F)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = torch.relu(self.linear1(x))          # (B,T,H)
        x = x.permute(1, 0, 2)                   # (T,B,H)
        x, _ = self.attention(x, x, x)           # (T,B,H)
        x = x.permute(1, 0, 2)                   # (B,T,H)
        x = x[:, -1, :]                          # 取最后时刻 (B,H)
        x = torch.relu(self.linear2(x))          # (B,H)
        x = self.dropout(x)
        return self.output(x)                    # (B,1)

class EnhancedNBeats(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, dropout=0.2):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_size, 1))
        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)