import torch

from customocr.utils.functions import encode_text
from customocr.utils.functions import transform_image


def collate_fn(batch):
    """Custom collate function for OCR. Handles variable image widths and text
    lengths.

    Args:
        batch: List of samples from the Dataset.
               Each sample = {"image": Tensor, "label": Tensor}
    Returns:
        Dict with batched images, labels, and label lengths.
    """

    # Separate images and labels
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # print(images[0])

    # Find max width and height for this batch
    # Handle both PIL Images and PyTorch tensors

    # Resize/pad images to fixed size (32x128)
    processed_images = []
    for img in images:
        img = transform_image(img)
        processed_images.append(img)

    padded_images = torch.stack(processed_images)

    # Concatenate all labels into a flat tensor (for CTC)
    # Convert string labels to tensor indices
    encoded_labels = []
    if isinstance(labels[0], str):
        # Labels are strings, encode them to indices
        for label in labels:
            encoded_labels.append(encode_text(label))
        labels_concat = torch.cat(encoded_labels)
        label_lengths = torch.tensor(
            [len(label) for label in encoded_labels], dtype=torch.long
        )

    else:
        # Labels are already tensors
        labels_concat = torch.cat(labels)
        label_lengths = torch.tensor(
            [len(label) for label in labels], dtype=torch.long
        )

    return {
        "images": padded_images,  # (B, C, 32, 128) - fixed size
        "labels": labels_concat,  # (sum of label lengths)
        "label_lengths": label_lengths,  # (B,)
    }
