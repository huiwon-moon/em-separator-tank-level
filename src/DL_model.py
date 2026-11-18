import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config.columns import Columns

class TimeSeriesDataset(Dataset):
    def __init__(self,
                 raw_data: pd.DataFrame,
                 data: pd.DataFrame,
                 feature_cols: list,
                 target_cols: list,
                 group_col: str,
                 lookback: int,
                 step_size: int,
                 ):

        self.lookback = lookback

        self.X_data = data[feature_cols].values.astype(np.float32)
        self.y_data = data[target_cols].values.astype(np.float32)
        self.y_raw_data = raw_data[[Columns.SEP_TANK_LVL.value]].values.astype(np.float32)

        self.indices = []
        self.target_indices = []

        grouped = data.groupby(group_col).indices

        for case_id, group_idx in sorted(grouped.items()):
            group_len = len(group_idx)
            
            if group_len < lookback:
                continue
            
            for i in range(lookback, group_len, step_size):
                seq_indices = group_idx[i - lookback : i]
                target_index = group_idx[i]

                self.indices.append((seq_indices, target_index))
                self.target_indices.append(target_index)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        seq_idx, target_idx = self.indices[idx]

        x = torch.from_numpy(self.X_data[seq_idx])
        y = torch.from_numpy(self.y_data[target_idx])
        y_true = torch.from_numpy(self.y_raw_data[target_idx])

        return x, y, y_true

class LSTMModel(nn.Module):
    def __init__(self, 
                 in_channels: int, 
                 num_horizons: int, 
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.2
                 ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        self.heads = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_horizons),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]

        return self.heads(last_step)


class GRUModel(nn.Module):
    def __init__(self,
                 in_channels: int,
                 num_horizons: int,
                 hidden_dim: int,
                 num_layers: int,
                 dropout: float = 0.2,
                 ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.heads = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim//2, num_horizons)
        )

    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]

        return self.heads(last_step)

class Conv1DModel(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 num_horizons: int,
                 ):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels=64, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128 * 2, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1)
            ) for _ in range(num_horizons)
        ])

    def forward(self, x):
        x = x.transpose(1, 2)

        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)

        avg_feat = self.global_avg_pool(c3).squeeze(-1)

        last_feat = c3[:, :, -1]

        combined = torch.cat([avg_feat, last_feat], dim=-1)
        outputs = [head(combined)for head in self.heads]

        return torch.cat(outputs, dim=-1)
