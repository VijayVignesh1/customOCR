from .base_model import BaseOCRModel
import torch.nn as nn
import torch

class CRNN(BaseOCRModel):
    def __init__(self, num_channels=3, num_classes=65, hidden_size=256, rnn_layers=2, dropout=0.5):
        """
        CRNN model for OCR.
        
        Args:
            num_channels (int): Number of input channels (3 for RGB, 1 for grayscale)
            num_classes (int): Number of output classes (vocabulary size + blank)
            hidden_size (int): Size of LSTM hidden state
        """
        super().__init__()

        self.cnn = nn.Sequential(
            # Input: (B, C, 32, 128)
            nn.Conv2d(num_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (B, 64, 16, 64)

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> (B, 128, 8, 32)

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # -> (B, 256, 4, 32)

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # -> (B, 256, 2, 32)

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),  # -> (B, 512, 1, 32)
        )

        self.rnn = nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True, num_layers=rnn_layers, dropout=dropout if rnn_layers > 1 else 0.0)

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # Input shape: (B, C, W, H)
        features = self.cnn(x)
        b, c, h, w = features.size()
        
        # Debug: print actual dimensions if assertion fails
        if h != 1:
            print(f"CNN output shape: {features.shape}")
            print(f"Expected height: 1, Got height: {h}")
        
        # Reshape for RNN: (B, W, C) where W is sequence length (W*H=W since H=1)
        features = features.squeeze(2).permute(0, 2, 1)  # (B, W, C)
        
        # RNN processing
        recurrent, (hidden, cell) = self.rnn(features)
        
        # Final classification
        output = self.fc(recurrent)
        
        # Return in CTC format: (T, N, C) where T is sequence length
        return output.permute(1, 0, 2)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
        return logits