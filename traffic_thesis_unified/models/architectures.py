import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class GRURegressor(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


class CNNRegressor(nn.Module):
    def __init__(self, input_dim=13, seq_len=12, output_dim=1):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        self.fc = nn.Linear(64, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)   # [batch, features, seq_len]
        out = self.conv(x)
        out = out.squeeze(-1)
        out = self.fc(out)
        return out


class TrafficModel(nn.Module):
    def __init__(self, model_type="LSTM", input_dim=13, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2, seq_len=12):
        super().__init__()

        self.model_type = model_type

        if model_type == "LSTM":
            self.model = LSTMRegressor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout
            )

        elif model_type == "GRU":
            self.model = GRURegressor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                output_dim=output_dim,
                dropout=dropout
            )

        elif model_type == "CNN":
            self.model = CNNRegressor(
                input_dim=input_dim,
                seq_len=seq_len,
                output_dim=output_dim
            )

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    def forward(self, x):
        return self.model(x)