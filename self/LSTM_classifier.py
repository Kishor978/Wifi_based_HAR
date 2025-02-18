import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_dir = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.hidden2label = nn.Sequential(
            nn.Linear(hidden_dim * self.num_dir, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_dir * self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_dir * self.num_layers, batch_size, self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))

        # Mean pooling (better than just last timestep)
        y = self.hidden2label(lstm_out.mean(dim=1))  

        return y
