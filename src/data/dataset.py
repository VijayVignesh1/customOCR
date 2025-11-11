from torch.utils.data import Dataset
from PIL import Image

class OCRDataset(Dataset):
    def __init__(self, label_file):
        with open(label_file, 'r') as f:
            self.samples = [line.strip().split('\t') for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        return image, text
