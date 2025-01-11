import torch
from torch import nn

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, dropout_rate, bidirectional, output_dim,seq_dim):
        super(CNN_BiLSTM_Attention, self).__init__()
        
        # CNN layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))
        
        # Calculate the flattened size after the CNN layers
        self.flattened_size = 64 * (input_dim // 4) * (seq_dim // 4)  # Adjust based on CNN operations
        
        # Fully connected layer to project the CNN output to the LSTM input size
        self.fc = nn.Linear(self.flattened_size, 128)
        
        # BiLSTM
        self.bilstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, num_layers=layer_dim,
                              dropout=dropout_rate, bidirectional=bidirectional, batch_first=True)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.fc_out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
    
    def forward(self, x):
        # CNN layers
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        
        # Flatten the CNN output
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Project to match LSTM input size
        x = self.fc(x)
        
        # Add a sequence dimension (assuming seq_len = 1)
        x = x.unsqueeze(1)
        
        # BiLSTM
        lstm_out, _ = self.bilstm(x)
        
        # Attention mechanism
        attn_weights = torch.softmax(self.attention(lstm_out).squeeze(-1), dim=1)
        context_vector = torch.sum(attn_weights.unsqueeze(-1) * lstm_out, dim=1)
        
        # Output layer
        out = self.fc_out(context_vector)
        
        return out
