from src.models.crnn import CRNN
from .crnn_mobilenet_small import CRNN_MobileNetV3_Small
from .crnn_resnet import CRNN_ResNet

def get_model(name, config):
    name = name.lower()
    if name == "crnn":
        return CRNN(**config)
    if name == "crnn_mobilenet_small":
        return CRNN_MobileNetV3_Small(**config)
    if name == "crnn_resnet":
        return CRNN_ResNet(**config)
    else:
        raise ValueError(f"Unknown model: {name}")