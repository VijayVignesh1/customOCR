from src.models.base_model import BaseOCRModel
import torch.nn as nn
import torch

class CRNN(BaseOCRModel):
    def __init__(self, img_height, num_channels, num_classes, hidden_size=256):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2, 1))
        )

        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True),
            nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True, batch_first=True),
        )

        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        b, c, h, w = features.size()
        assert h == 1, "The height after conv must be 1"
        features = features.squeeze(2).permute(0, 2, 1)
        recurrent, _ = self.rnn(features)
        output = self.fc(recurrent)
        return output.permute(1, 0, 2)  # (T, N, C)

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            preds = logits.argmax(2)
        return preds