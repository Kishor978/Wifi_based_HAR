# import torch
# from torch import nn

# class CNN_BiLSTM_Attention(nn.Module):
#     def __init__(self, input_dim, hidden_dim, layer_dim, dropout_rate, bidirectional, output_dim, seq_dim):
#         super(CNN_BiLSTM_Attention, self).__init__()

#         # CNN layers with ReLU activation and BatchNorm
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.maxpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

#         # Compute H and W dynamically
#         self.H = input_dim // 4  # Approximate feature map size after pooling
#         self.W = seq_dim // 4

#         # BiLSTM
#         lstm_input_size = 64 * self.H
#         self.bilstm = nn.LSTM(
#             input_size=lstm_input_size,
#             hidden_size=hidden_dim,
#             num_layers=layer_dim,
#             dropout=dropout_rate if layer_dim > 1 else 0,
#             bidirectional=bidirectional,
#             batch_first=True
#         )

#         # Attention mechanism
#         self.attention = nn.Sequential(
#             nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128),
#             nn.Tanh(),
#             nn.Linear(128, 1)
#         )

#         # Fully connected layer
#         self.fc_out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout_rate)  # Extra dropout for regularization

#     def forward(self, x):
#         # CNN Processing
#         x = torch.relu(self.bn1(self.conv1(x)))
#         x = self.maxpool(x)
#         x = torch.relu(self.bn2(self.conv2(x)))
#         x = self.maxpool(x)  # Shape: (batch, 64, H, W)

#         # Reshape for BiLSTM
#         batch_size = x.size(0)
#         x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, self.W, -1)

#         # BiLSTM Processing
#         lstm_out, _ = self.bilstm(x)  # Shape: (batch, W, hidden_dim * num_directions)

#         # Attention Weights
#         attn_scores = self.attention(lstm_out)  # Shape: (batch, W, 1)
#         attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=-1)  # Shape: (batch, W)

#         # Context Vector
#         context_vector = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

#         # Fully Connected Layer
#         out = self.fc_out(self.dropout(context_vector))
#         return out

import torch
from torch import nn

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, dropout_rate, bidirectional, output_dim, seq_dim):
        super(CNN_BiLSTM_Attention, self).__init__()

        # CNN layers with ReLU activation, BatchNorm, and MaxPool
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Compute H and W dynamically
        self.H = input_dim // 4  # Approximate feature map size after pooling
        self.W = seq_dim // 4

        # BiLSTM
        lstm_input_size = 64 * self.H
        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            dropout=dropout_rate if layer_dim > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        # Fully connected layer
        self.fc_out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)  # Extra dropout for regularization

    def forward(self, x):
        # CNN Processing
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)  # Apply MaxPool after first conv block
        
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.maxpool2(x)  # Apply MaxPool after second conv block

        # Reshape for BiLSTM
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, self.W, -1)

        # BiLSTM Processing
        lstm_out, _ = self.bilstm(x)  # Shape: (batch, W, hidden_dim * num_directions)

        # Attention Weights
        attn_scores = self.attention(lstm_out)  # Shape: (batch, W, 1)
        attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=-1)  # Shape: (batch, W)

        # Context Vector
        context_vector = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)

        # Fully Connected Layer
        out = self.fc_out(self.dropout(context_vector))
        return out
