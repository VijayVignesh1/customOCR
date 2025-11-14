import json

from PIL import Image
from torch.utils.data import Dataset


class OCRDataset(Dataset):
    def __init__(self, label_file):
        with open(label_file, "r") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = list(self.samples.keys())[idx]
        text = self.samples[img_path]
        image = Image.open(img_path).convert("RGB")
        return image, text
