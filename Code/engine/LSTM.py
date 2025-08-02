import torch
from torch import nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, hidden_size : int, input_feature: int = 1 , num_layers: int =1, dropout : float = 0.0):
        super(LSTM_Model, self).__init__()
        self.lstm = nn.LSTM(input_size = input_feature, hidden_size=hidden_size, num_layers= num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size,1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out


class StackedLSTM(nn.Module):
    def __init__(self, hidden_size: int, input_feature: int = 1, num_layers: int = 2, dropout: float = 0.3):
        super(StackedLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out

class BiLSTM(nn.Module):
    def __init__(self, hidden_size: int, input_feature: int = 1, num_layers: int = 1, dropout: float = 0.0):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)  # Because bidirectional doubles hidden size

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_time_step = lstm_out[:, -1, :]
        out = self.fc(last_time_step)
        return out

class AttentionLSTM(nn.Module):
    def __init__(self, hidden_size: int, input_feature: int = 1, num_layers: int = 1, dropout: float = 0.0):
        super(AttentionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # shape: (batch, seq_len, hidden_size)
        attn_weights = F.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        weighted = torch.sum(lstm_out * attn_weights, dim=1)      # (batch, hidden_size)
        out = self.fc(weighted)
        return out