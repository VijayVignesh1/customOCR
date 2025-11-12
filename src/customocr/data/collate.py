import torch
import torch.nn.functional as F
from torchvision import transforms
from customocr.data.charset import CHAR_TO_IDX, IDX_TO_CHAR

# ---------------------------
# Encode text -> indices
# ---------------------------
def encode_text(text, char_to_idx=CHAR_TO_IDX):
    """Convert text string to tensor of character indices."""
    indices = []
    for char in text:  # keep case as-is
        if char in char_to_idx:
            indices.append(char_to_idx[char])
        else:
            indices.append(char_to_idx['<unk>'])
    return torch.tensor(indices, dtype=torch.long)

# ---------------------------
# Decode indices -> text (CTC greedy)
# ---------------------------
def decode_text(indices, idx_to_char=IDX_TO_CHAR):
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
            if char != '<blank>':
                chars.append(char)
        last_idx = idx
    return ''.join(chars)

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
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    # print(images[0])
    
    # Find max width and height for this batch
    # Handle both PIL Images and PyTorch tensors
    target_height = 32
    target_width = 128

    # Resize/pad images to fixed size (32x128)
    processed_images = []
    for img in images:
        # Convert to tensor if it's a PIL Image
        if not torch.is_tensor(img):
            # Assuming it's a PIL Image, convert to tensor
            to_tensor = transforms.ToTensor()
            img = to_tensor(img)
        
        c, h, w = img.shape  # Use shape instead of size for consistency
        
        # First resize if image is larger than target
        if h > target_height or w > target_width:
            # Calculate scaling factor to fit within target dimensions
            scale_h = target_height / h
            scale_w = target_width / w
            scale = min(scale_h, scale_w)  # Use smaller scale to maintain aspect ratio
            
            new_h = int(h * scale)
            new_w = int(w * scale)
            
            # Resize using interpolation
            img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
            img = img.squeeze(0)
            h, w = new_h, new_w
        
        # Now pad to exact target size
        pad_h = target_height - h
        pad_w = target_width - w
        
        # Center the image by padding equally on both sides when possible
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # pad format: (left, right, top, bottom)
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
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
        label_lengths = torch.tensor([len(l) for l in encoded_labels], dtype=torch.long)
    else:
        # Labels are already tensors
        labels_concat = torch.cat(labels)
        label_lengths = torch.tensor([len(l) for l in labels], dtype=torch.long)

    return {
        "images": padded_images,       # (B, C, 32, 128) - fixed size
        "labels": labels_concat,       # (sum of label lengths)
        "label_lengths": label_lengths # (B,)
    }