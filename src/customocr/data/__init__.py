from .collate import collate_fn
from .dataset import OCRDataset
from .generate_synthetic import predefined_strings
from .generate_synthetic import random_strings

__all__ = ["random_strings", "predefined_strings", "OCRDataset", "collate_fn"]
