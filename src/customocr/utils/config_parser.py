import yaml
from pathlib import Path

def load_config(config_path):
    """
    Load a YAML configuration file and return as a Python dictionary.
    Args:
        config_path (str or Path): path to YAML config file
    Returns:
        dict: parsed configuration
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg