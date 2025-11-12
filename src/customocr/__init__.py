"""
Source package for customOCR project.

A custom OCR implementation with synthetic data generation capabilities.
"""

from . import data
from . import models
from . import utils

__all__ = [
    "data",
    "models", 
    "utils"
]

__version__ = "1.0.0"
__author__ = "Vijay Vignesh"