# GAN - Synthetic Image Generator

A Vanilla GAN implementation for generating synthetic images using PyTorch.

## Project Structure

```
GAN PROJECT/
├── config.yaml          # Configuration file
├── data_loader.py       # Data pipeline & preprocessing
├── generator.py         # Generator network
├── discriminator.py     # Discriminator network
├── vanilla_gan.py       # Combined GAN model
└── preprocessed_data/   # Preprocessed dataset
    ├── train_data.pt    # Training data (95%)
    ├── test_data.pt     # Test data (5%)
    ├── train_samples.png
    └── test_samples.png
```

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pandas
- numpy
- matplotlib
- tqdm
- PyYAML

## Installation

```bash
# Clone repository
git clone https://github.com/saivighnesh2190/GAN---Synthetic-Image-Generator-.git
cd "GAN---Synthetic-Image-Generator-"

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision pandas numpy matplotlib tqdm pyyaml
```

## Dataset Setup

This project uses the **Amazon Berkeley Objects (ABO) Dataset**.

### Download the Dataset

1. **Download ABO Small Images** from:
   - Official: https://amazon-berkeley-objects.s3.amazonaws.com/index.html
   - Direct link: `abo-images-small.tar` (~3GB)

2. **Download metadata**:
   - `images.csv.gz` from the listings metadata

3. **Extract and organize** the files:
```bash
# Create dataset folder
mkdir -p abo_dataset/metadata

# Extract images
tar -xf abo-images-small.tar -C abo_dataset/
mv abo_dataset/images abo_dataset/small

# Move metadata
mv images.csv.gz abo_dataset/metadata/
gunzip abo_dataset/metadata/images.csv.gz
```

4. **Final folder structure should look like**:
```
abo_dataset/
├── metadata/
│   └── images.csv       # Image metadata
└── small/
    ├── 00/
    │   └── 00xxxxx.jpg
    ├── 01/
    │   └── 01xxxxx.jpg
    └── ...              # More image folders
```

### Alternative: Use Your Own Dataset

You can use any image dataset by modifying `config.yaml`:
```yaml
data:
  csv_file: "your_dataset/metadata.csv"  # CSV with 'path' column
  img_dir: "your_dataset/images"
```

Your CSV should have a `path` column with relative image paths.

## Running the Project

### Module 1: Data Pipeline

```bash
# Preprocess data and create train/test split
python data_loader.py
```

This will:
- Load images from `abo_dataset/`
- Resize to 64x64, normalize to [-1, 1]
- Split into train (95%) and test (5%)
- Save to `preprocessed_data/`

### Module 2: Test GAN Architecture

```bash
# Test Generator
python generator.py

# Test Discriminator
python discriminator.py

# Test complete Vanilla GAN
python vanilla_gan.py
```

## Configuration

Edit `config.yaml` to change settings:

```yaml
preprocessing:
  image_size: 64
  channels: 3
  normalization: "tanh"

split:
  train_split: 0.95

dataloader:
  batch_size: 16
  max_samples: 5000
```

## Model Architecture

### Generator
- Input: Latent vector z ∈ R^100
- FC layer → 4×4×512 → ConvTranspose blocks → 64×64×3
- Activation: ReLU (hidden), Tanh (output)

### Discriminator
- Input: Image 64×64×3
- 4 Conv blocks → Dense → Sigmoid
- Activation: LeakyReLU(0.2)

### Training
- Loss: Binary Cross Entropy
- Optimizer: Adam (lr=0.0002, β1=0.5)

## Usage in Your Code

```python
from data_loader import get_preprocessed_dataloader
from vanilla_gan import VanillaGAN

# Load data
train_loader, _ = get_preprocessed_dataloader('preprocessed_data/train_data.pt')

# Initialize GAN
gan = VanillaGAN(latent_dim=100, device='cuda')

# Training loop (Module 3)
for epoch in range(num_epochs):
    for batch in train_loader:
        real_images = batch['image'].to(device)
        d_loss, _, _ = gan.train_discriminator(real_images)
        g_loss = gan.train_generator(batch_size=real_images.size(0))
```

## License

MIT License
