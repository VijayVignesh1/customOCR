import torch
from torch import nn
from customocr.data.collate import decode_text
import pytorch_lightning as pl
from customocr.utils.metrics import compute_cer, compute_wer

class OCRLightningModule(pl.LightningModule):
    def __init__(self, model, cfg):
        """
        Lightning module for OCR.
        
        Args:
            model: PyTorch OCR model (CRNN, MobileNet CRNN, etc.)
            cfg: Config dict
        """
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.lr = cfg["train"]["lr"]
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
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

        # Decode predictions
        _, pred_indices = preds_log.max(2)
        pred_indices = pred_indices.transpose(1, 0)

        pred_texts = [decode_text(torch.tensor(seq.cpu().numpy())) for seq in pred_indices]

        # Decode ground truth
        gt_texts = []
        start = 0
        for length in label_lengths:
            seq = labels[start:start+length].cpu()
            gt_texts.append(decode_text(seq))
            start += length

        cer = compute_cer(gt_texts, pred_texts)
        wer = compute_wer(gt_texts, pred_texts)
        self.log("val_cer", cer, prog_bar=True)
        self.log("val_wer", wer, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

# cfg = load_config("../configs/config.yaml")
# lit_model = OCRTrainer(cfg)
# lit_model.fit()
