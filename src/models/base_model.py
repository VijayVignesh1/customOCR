import torch.nn as nn
from abc import ABC, abstractmethod

class BaseOCRModel(nn.Module, ABC):
    """Abstract Base Class for all OCR models"""

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        """Inference wrapper"""
        pass