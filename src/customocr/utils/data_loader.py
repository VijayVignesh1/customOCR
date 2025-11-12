from torch.utils.data import DataLoader
from customocr.data.dataset import OCRDataset
from customocr.data.collate import collate_fn  # assuming you already have this

def create_dataloaders(cfg):
    """
    Create train and validation dataloaders for OCR fine-tuning.
    Args:
        cfg (dict): configuration dict from YAML
    Returns:
        tuple: (train_loader, val_loader)
    """

    # Dataset paths
    train_dir = cfg["train_dir"]
    val_dir = cfg["val_dir"]

    # Instantiate datasets
    train_dataset = OCRDataset(train_dir)
    val_dataset = OCRDataset(val_dir)

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_loader, val_loader