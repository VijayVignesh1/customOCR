from .base_model import BaseOCRModel
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class CRNN_MobileNetV3_Small(BaseOCRModel):
    def __init__(self, num_classes=65, hidden_size=256, rnn_layers=2, dropout=0.5, pretrained=True):
        """
        CRNN model using MobileNetV3-Small backbone for OCR fine-tuning.

        Args:
            num_classes (int): Number of output classes (charset size + blank)
            hidden_size (int): LSTM hidden state size
            rnn_layers (int): Number of LSTM layers
            dropout (float): Dropout in LSTM
            pretrained (bool): Whether to load pretrained ImageNet weights
        """
        super().__init__()

        # Load pretrained MobileNetV3-Small as CNN feature extractor
        mobilenet = mobilenet_v3_large(pretrained=pretrained)

        # Remove classifier and pooling layers
        self.cnn = nn.Sequential(*list(mobilenet.features.children()))
        cnn_output_channels = 960  # last conv output channels for MobileNetV3-Small

        with torch.inference_mode():
            out_shape = self.cnn(torch.zeros((1, *(3, 32, 128)))).shape

        # Add adaptive pooling to normalize height → 1
        # CRNN expects (B, C, 1, W)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 32))

        # BiLSTM sequence modeler
        self.rnn = nn.LSTM(
            input_size=cnn_output_channels,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # Linear projection to class logits
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Output in (T, N, C) format for CTC loss
        """
        features = self.cnn(x)                 # (B, C, H', W')
        features = self.adaptive_pool(features)  # (B, C, 1, W')
        b, c, h, w = features.size()

        # (B, W', C)
        features = features.squeeze(2).permute(0, 2, 1)

        # RNN sequence modeling
        recurrent, _ = self.rnn(features)

        # Classification
        output = self.fc(recurrent)  # (B, W', num_classes)

        # Convert to (T, N, C) for CTC
        return output.permute(1, 0, 2)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        logits = self.forward(x)
        preds = logits.argmax(2)
        return preds
