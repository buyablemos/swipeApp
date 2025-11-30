import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, input_length, num_classes):
        super().__init__()
        
        # Architecture matching cnn.py as closely as possible
        # Keras Conv1D default: kernel=3, stride=1, padding='valid' (no padding)
        
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size dynamically
        self._flatten_size = self._get_flatten_size(input_channels, input_length)
        # print(f"Flattened size after conv and pooling: {self._flatten_size}")
        self.fc1 = nn.Linear(self._flatten_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def _get_flatten_size(self, in_channels, length):
        # Helper to pass a dummy input and see the size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, length)
            x = self.conv1(dummy)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.pool2(x)
            return x.numel()

    def forward(self, x):
        # x shape: (Batch, Channels, Length)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
