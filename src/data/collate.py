import torch
import torch.nn.functional as F

def collate_fn(batch):
    """
    Custom collate function for OCR.
    Handles variable image widths and text lengths.
    
    Args:
        batch: List of samples from the Dataset. 
               Each sample = {"image": Tensor, "label": Tensor}
    Returns:
        Dict with batched images, labels, and label lengths.
    """
    # Separate images and labels
    images = [item["image"] for item in batch]
    labels = [item["label"] for item in batch]
    
    # Find max width and height for this batch
    heights = [img.size(1) for img in images]
    widths = [img.size(2) for img in images]
    max_h = max(heights)
    max_w = max(widths)

    # Pad images to same size (for batching)
    padded_images = []
    for img in images:
        c, h, w = img.size()
        pad_h = max_h - h
        pad_w = max_w - w
        # pad format: (left, right, top, bottom)
        img = F.pad(img, (0, pad_w, 0, pad_h), value=0.0)
        padded_images.append(img)
    padded_images = torch.stack(padded_images)

    # Concatenate all labels into a flat tensor (for CTC)
    labels_concat = torch.cat(labels)
    label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    return {
        "images": padded_images,       # (B, C, H, W)
        "labels": labels_concat,       # (sum of label lengths)
        "label_lengths": label_lengths # (B,)
    }