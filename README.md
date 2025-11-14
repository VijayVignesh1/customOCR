# CustomOCR

A quick and easy pipeline for generating OCR dataset and finetuning a model.

## Overview

CustomOCR is a complete end-to-end OCR training pipeline built with PyTorch and PyTorch Lightning. It provides tools for synthetic text image generation, data preprocessing, and training CRNN (Convolutional Recurrent Neural Network) models for text recognition.

## Features

- **Synthetic Data Generation**: Generate text images using TRDG (Text Recognition Data Generator)
- **CRNN Architecture**: Implemented CNN + LSTM architecture for text recognition
- **CTC Loss**: Connectionist Temporal Classification for sequence-to-sequence learning
- **Configurable Pipeline**: YAML-based configuration for easy experimentation
- **PyTorch Lightning**: Modern training framework with built-in logging and checkpointing
- **Flexible Image Sizes**: Automatic resizing and padding to standardized dimensions (32x128)

## Project Structure

```
customOCR/
├── configs/
│   └── config.yaml               # Main configuration file
├── data/                         # Generated datasets directory
├── src/
│   ├── ${root_dir}/              # Root directory reference
│   ├── __init__.py               # Package initialization
│   ├── main.py                   # Main entry point
│   ├── train.py                  # Training script
│   ├── data/
│   │   ├── __init__.py           # Data module exports
│   │   ├── charset.py            # Character set definitions
│   │   ├── collate.py            # Custom collate function with text encoding
│   │   ├── dataset.py            # OCR dataset class
│   │   ├── factory.py            # Data factory for different generators
│   │   └── generate_synthetic.py # Synthetic data generation
│   ├── models/
│   │   ├── __init__.py           # Model module exports
│   │   ├── base_model.py         # Abstract base class for OCR models
│   │   ├── crnn.py               # Basic CRNN implementation
│   │   ├── crnn_mobilenet_small.py # MobileNet-based CRNN
│   │   ├── crnn_resnet.py        # ResNet-based CRNN
│   │   └── factory.py            # Model factory
│   └── utils/
│       ├── __init__.py           # Utils module exports
│       ├── config_parser.py      # YAML configuration parser
│       ├── data_loader.py        # DataLoader creation utilities
│       ├── functions.py          # Utility functions
│       └── metrics.py            # Evaluation metrics (CER/WER)
├── customocr.egg-info/           # Package metadata
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup configuration
├── LICENSE                       # MIT License
└── README.md                     # This file
```


## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VijayVignesh1/customOCR.git
   cd customOCR
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install system dependencies** (for TRDG and PIL):
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install libfreetype6-dev libjpeg-dev libpng-dev libwebp-dev libopenjp2-7-dev build-essential

   # CentOS/RHEL/Fedora
   sudo dnf install freetype-devel libjpeg-devel libpng-devel libwebp-devel openjpeg2-devel gcc gcc-c++
   ```

4. **Install the package**:
   ```bash
   pip install -e .
   ```

## Quick Start

### 1. Generate Synthetic Dataset

```bash
# Generate training and validation data
python -c "
from src.data.generate_synthetic import predefined_strings
predefined_strings(
    output_dir='data/synthetic/train',
    strings=[f'Sample Text {i}' for i in range(100)]
)
predefined_strings(
    output_dir='data/synthetic/val',
    strings=[f'Validation {i}' for i in range(20)]
)
"
```

### 2. Configure Training

Edit `configs/config.yaml` to customize your training:

```yaml
data:
  train_dir: "data/synthetic/train/labels.json"
  val_dir: "data/synthetic/val/labels.json"
  batch_size: 32

model:
  name: "crnn_resnet"  # or "crnn", "crnn_mobilenet_small"
  configs:
    num_classes: 98
    hidden_size: 256

train:
  epochs: 20
  lr: 0.001
```

### 3. Train the Model

```bash
# Run training
python src/main.py

# Or use the training script directly
python src/train.py
```

### 4. Use custom config file

```bash
from customOCR import CustomOCR

config_path = "/path/to/config.yaml"
ocr_trainer = CustomOCR(config_path)
ocr_trainer.fit()
```

### 5. Quick Test

```python
from src.models import get_model
from src.utils import load_config
from PIL import Image

# Load trained model
config = load_config("configs/config.yaml")
model = get_model(config["model"]["name"], config["model"]["configs"])

# Load and predict on an image
image = Image.open("path/to/your/image.png")
prediction = model.predict(image)
print(f"Predicted text: {prediction}")
```


## Architecture Details

### CRNN Model
- **CNN Backbone**: Custom CNN / ResNet / MobileNet
- **RNN Component**: 2-layer bidirectional LSTM for sequence modeling
- **Output Layer**: Linear layer mapping to character vocabulary
- **Input**: RGB images of size 32×128
- **Output**: Character sequences via CTC decoding

### Character Vocabulary
The model supports:
- Lowercase letters: a-z (indices 2-27)
- Digits: 0-9 (indices 54-63)
- Space: (index 1)
- CTC Blank: (index 0)

### Data Processing
- **Image Preprocessing**: Automatic resize/padding to 32×128
- **Text Encoding**: Character-to-index mapping for CTC loss
- **Collate Function**: Custom batching with variable text lengths


### Configuration (`configs/config.yaml`)
```yaml

data:
  dataset_dir: "../data/generated/synthetic"
  num_images: 10
  img_height: 32
  img_width: 128
  batch_size: 32
  train_split: 0.9
  shuffle: true
  num_workers: 4
  generator:
    name: "predefined_strings"   # options: "predefined_strings", "random_words", etc.
    train:
      output_dir: "../data/generated/train"
    val:
      output_dir: "../data/generated/val"
    params:
      strings:
        - "Hello"
        - "World"
      count: 5
      size: 100
      language: "en"
      text_color: "#000000"
      skewing_angle: 3
      blur: 1
  train_dir: "../data/generated/train/labels.json"
  val_dir: "../data/generated/val/labels.json"

model:
  name: "crnn_resnet"  # model class name in src/models/
  configs:
    num_classes: 38  # Updated: 26 lowercase + 10 digits + 2 blank = 38
    hidden_size: 256
    rnn_layers: 2
    dropout: 0.1
    backbone: "resnet34"
    pretrained: True

train:
  epochs: 20
  lr: 0.0001
  optimizer: "Adam"
  weight_decay: 0.0
  device: "cuda"

checkpoint:
  save_dir: "outputs/checkpoints"
  save_every_n_epochs: 5
  resume_from: null

logging:
  log_dir: "outputs/logs"
  log_every_n_steps: 10
```

## Usage Examples [WIP]

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- PyTorch Lightning
- Pillow (PIL)
- NumPy
- TRDG (Text Recognition Data Generator)
- PyYAML
- tqdm

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [TRDG](https://github.com/Belval/TextRecognitionDataGenerator) for synthetic text image generation
- [PyTorch Lightning](https://www.pytorchlightning.ai/) for the training framework
- CRNN architecture based on ["An End-to-End Trainable Neural OCR"](https://arxiv.org/abs/1507.05717)
