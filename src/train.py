import torch
from torch import nn
import pytorch_lightning as pl
# from .utils.metrics import compute_cer, compute_wer
from pytorch_lightning import Trainer
from src.models.crnn import CRNN
from src.utils.data_loader import create_dataloaders
from src.utils.config_parser import load_config

class OCRLightningModule(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg["train"]["lr"]
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        """Forward pass."""
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Defines a single step in training loop."""
        imgs = batch["images"]
        labels = batch["labels"]
        label_lengths = batch["label_lengths"]

        preds = self(imgs)
        preds_log = preds.log_softmax(2)
        input_lengths = torch.full(
            (preds.size(1),), preds.size(0), dtype=torch.long, device=self.device
        )

        loss = self.criterion(preds_log, labels, input_lengths, label_lengths)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Runs validation + logs CER/WER."""
        imgs = batch["images"]
        labels = batch["labels"]
        label_lengths = batch["label_lengths"]

        preds = self(imgs)
        preds_log = preds.log_softmax(2)
        input_lengths = torch.full(
            (preds.size(1),), preds.size(0), dtype=torch.long, device=self.device
        )

        loss = self.criterion(preds_log, labels, input_lengths, label_lengths)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # Optional: Decode for CER/WER metrics
        # _, pred_indices = preds_log.max(2)
        # pred_indices = pred_indices.transpose(1, 0).contiguous().cpu().numpy()

        # cer = compute_cer(pred_indices, labels.cpu().numpy())
        # wer = compute_wer(pred_indices, labels.cpu().numpy())

        # self.log("val_cer", cer, prog_bar=True)
        # self.log("val_wer", wer, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Define optimizer (and optionally scheduler)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


cfg = load_config("configs/data.yaml")
train_loader, val_loader = create_dataloaders(cfg)

cfg = load_config("configs/crnn.yaml")
model = CRNN(num_classes=cfg["num_classes"])
lit_model = OCRLightningModule(model, cfg)

trainer = Trainer(
    max_epochs=cfg["train"]["epochs"],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision="16-mixed",  # optional for speedup
    log_every_n_steps=10
)

trainer.fit(lit_model, train_loader, val_loader)
