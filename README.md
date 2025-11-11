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

## Project Structure [WIP]

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
3. **Install the package**:
    ```bash
    pip install -e .
    ```

## Quick Start [WIP]


## Architecture Details

### CRNN Model
- **CNN Backbone**: 5 convolutional layers with MaxPooling to reduce height to 1
- **RNN Component**: 2-layer bidirectional LSTM for sequence modeling
- **Output Layer**: Linear layer mapping to character vocabulary
- **Input**: RGB images of size 32×128
- **Output**: Character sequences via CTC decoding

### Character Vocabulary
The model supports:
- Lowercase letters: a-z (indices 2-27)
- Uppercase letters: A-Z (indices 28-53)
- Digits: 0-9 (indices 54-63)
- Special characters: `!@#$%^&*(){}[]|\~`'".,;:?/-_+=<>` (indices 64-97)
- Space: (index 1)
- CTC Blank: (index 0)

### Data Processing
- **Image Preprocessing**: Automatic resize/padding to 32×128
- **Text Encoding**: Character-to-index mapping for CTC loss
- **Collate Function**: Custom batching with variable text lengths

## Configuration

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
