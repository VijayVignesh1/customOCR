from PIL import Image
from torchvision import transforms
import pytorch_lightning as pl
import torch

from customocr.utils.config_parser import load_config
from customocr.models.factory import get_model
from customocr.utils.data_loader import create_dataloaders
from customocr.data.factory import get_generator
from customocr.train import OCRLightningModule
from customocr.data.collate import decode_text

class CustomOCR:
    """
    Trainer class for OCR model fine-tuning.
    Args:
        cfg (dict): Configuration dictionary.
    """
    def __init__(self, cfg: str):

        self.cfg = load_config(cfg)

        # Generate dataset if missing
        generator_name = self.cfg["data"]["generator"]["name"]
        generator = get_generator(generator_name)
        train_params = self.cfg["data"]["generator"]["train"]
        val_params = self.cfg["data"]["generator"]["val"]

        if self.cfg["data"]["generator"]["name"] != "none":
            generator(train_params["output_dir"], **self.cfg["data"]["generator"]["params"])
            generator(val_params["output_dir"], **self.cfg["data"]["generator"]["params"])

        # Create dataloaders
        self.train_loader, self.val_loader = create_dataloaders(self.cfg["data"])

        # Initialize model
        self.model = get_model(self.cfg["model"]["name"], self.cfg["model"]["configs"])

        # Wrap in LightningModule
        self.lit_model = OCRLightningModule(self.model, self.cfg)

        print("[INFO] Trainer initialized.")

    def fit(self):
        trainer = pl.Trainer(
            max_epochs=self.cfg["train"]["epochs"],
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None
        )
        trainer.fit(self.lit_model, train_dataloaders=self.train_loader, val_dataloaders=self.val_loader)
        trainer.validate(self.lit_model, self.val_loader)

    def predict(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((32, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        preds = self.lit_model.model.predict(image_tensor)
        preds = preds.log_softmax(2)
        _, pred_indices = preds.max(2)
        pred_indices = pred_indices.transpose(1, 0)
        return decode_text(pred_indices[0])


if __name__ == "__main__":

    # Initialize trainer and start training
    config_path = "../../configs/config.yaml"
    ocr_trainer = CustomOCR(config_path)
    ocr_trainer.fit()

    # Make predictions
    test_image_path = "../../data/generated/train/000001.png"
    predicted_text = ocr_trainer.predict(test_image_path)
    print(f"Predicted Text: {predicted_text}")
