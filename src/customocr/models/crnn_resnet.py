from .base_model import BaseOCRModel
import torch
import torch.nn as nn
from torchvision.models import resnet34, resnet18, resnet50


class CRNN_ResNet(BaseOCRModel):
    def __init__(
        self,
        num_classes=65,
        hidden_size=256,
        rnn_layers=2,
        dropout=0.1,
        backbone="resnet34",
        pretrained=True,
        **kwargs,
    ):
        """
        CRNN model using ResNet backbone for OCR fine-tuning.

        Args:
            num_classes (int): Number of output classes (charset size + blank)
            hidden_size (int): LSTM hidden state size
            rnn_layers (int): Number of LSTM layers
            dropout (float): Dropout in LSTM
            backbone (str): ResNet variant ('resnet18', 'resnet34', 'resnet50')
            pretrained (bool): Whether to load pretrained ImageNet weights
        """
        super().__init__()

        # 1️⃣ Load pretrained ResNet backbone
        if backbone == "resnet18":
            resnet = resnet18(pretrained=pretrained)
            cnn_output_channels = 512
        elif backbone == "resnet34":
            resnet = resnet34(pretrained=pretrained)
            cnn_output_channels = 512
        elif backbone == "resnet50":
            resnet = resnet50(pretrained=pretrained)
            cnn_output_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Remove avgpool and fc layers (keep convolutional feature extractor)
        self.cnn = nn.Sequential(*list(resnet.children())[:-2])

        # Check CNN output shape dynamically
        with torch.inference_mode():
            dummy = torch.zeros((1, 3, 32, 128))
            out_shape = self.cnn(dummy).shape
        print(f"{backbone} CNN output shape:", out_shape)

        # Adaptive pooling to normalize height → 1
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
        features = self.cnn(x)                   # (B, C, H', W')
        features = self.adaptive_pool(features)  # (B, C, 1, W')
        b, c, h, w = features.size()

        # Flatten height and permute for RNN
        features = features.squeeze(2).permute(0, 2, 1)  # (B, W', C)

        # Sequence modeling
        recurrent, _ = self.rnn(features)

        # Classification
        output = self.fc(recurrent)  # (B, W', num_classes)

        # Convert to (T, N, C) for CTC
        return output.permute(1, 0, 2)

    @torch.no_grad()
    def predict(self, x):
        self.eval()
        logits = self.forward(x)
        return logits
