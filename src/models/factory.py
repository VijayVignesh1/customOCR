from src.models.crnn import CRNN

def get_model(name, config):
    name = name.lower()
    if name == "crnn":
        return CRNN(**config)
    else:
        raise ValueError(f"Unknown model: {name}")