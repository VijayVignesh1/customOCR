import random

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from customocr.data.charset import CHAR_TO_IDX
from customocr.data.charset import IDX_TO_CHAR


# ---------------------------
# Encode text -> indices
# ---------------------------
def encode_text(text: str, char_to_idx: dict = CHAR_TO_IDX) -> torch.Tensor:
    """Convert text string to tensor of character indices."""
    indices = []
    for char in text:  # keep case as-is
        if char in char_to_idx:
            indices.append(char_to_idx[char])
        else:
            indices.append(char_to_idx["<unk>"])
    return torch.tensor(indices, dtype=torch.long)


# ---------------------------
# Decode indices -> text (CTC greedy)
# ---------------------------
def decode_text(indices: list, idx_to_char: dict = IDX_TO_CHAR) -> str:
    """
    Convert a sequence of indices to string using CTC rules:
    - Remove repeated characters
    - Skip <blank>
    - Works for torch tensors or lists
    """
    chars = []
    last_idx = None
    for idx in indices:
        # Convert tensor element to int
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        # Skip blanks and repeated characters
        if idx != last_idx and idx in idx_to_char:
            char = idx_to_char[idx]
            if char != "<blank>":
                chars.append(char)
        last_idx = idx
    return "".join(chars)


def transform_image(image: Image.Image) -> torch.Tensor:
    """Transform a PIL image to a tensor suitable for model input. Resizes and
    normalizes the image.

    Args:
        image: PIL Image
    Returns:
        Tensor: Transformed image tensor
    """
    # Convert to tensor first
    to_tensor = transforms.ToTensor()
    img = to_tensor(image)

    c, h, w = img.shape
    target_height, target_width = 32, 128

    # Resize if larger than target (matching collate_fn logic)
    if h > target_height or w > target_width:
        scale_h = target_height / h
        scale_w = target_width / w
        scale = min(scale_h, scale_w)

        new_h = int(h * scale)
        new_w = int(w * scale)

        img = F.interpolate(
            img.unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        )
        img = img.squeeze(0)
        h, w = new_h, new_w

    # Pad to exact target size (matching collate_fn logic)
    pad_h = target_height - h
    pad_w = target_width - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
    return img


def generate_random_word():
    """Generate a random word consisting of lowercase letters and digits."""
    letters = "abcdefghijklmnopqrstuvwxyz0123456789"
    word_length = random.randint(5, 10)
    return "".join(random.choice(letters) for _ in range(word_length))
