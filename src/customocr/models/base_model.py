from abc import ABC
from abc import abstractmethod

import torch.nn as nn


class BaseOCRModel(nn.Module, ABC):
    """Abstract Base Class for all OCR models."""

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def predict(self, x):
        """Inference wrapper."""
