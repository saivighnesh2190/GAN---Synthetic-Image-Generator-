# 🌿 GAN Synthetic Leaf Disease Image Generator

A complete Vanilla GAN implementation for generating synthetic plant leaf disease images using the PlantDoc dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/vighnesh2190/generate-leaves-images)

## 🌐 Live Demo

**Try it now:** [https://huggingface.co/spaces/vighnesh2190/generate-leaves-images](https://huggingface.co/spaces/vighnesh2190/generate-leaves-images)

Generate synthetic leaf disease images directly in your browser - no installation required!

## 📋 Overview

This project implements a Generative Adversarial Network (GAN) that generates synthetic images of diseased plant leaves. It can be used for:
- **Data Augmentation** for plant disease detection models
- **Research** in agricultural AI
- **Education** on GAN architectures

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/saivighnesh2190/GAN---Synthetic-Image-Generator-.git
cd GAN---Synthetic-Image-Generator-
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Dataset
```bash
mkdir -p leaf_dataset
cd leaf_dataset
git clone https://github.com/pratikkayal/PlantDoc-Dataset.git --depth 1
cd ..
```

### 4. Train the Model
```bash
# Preprocess data
python plantdoc_loader.py

# Train GAN (100 epochs)
python train.py
```

### 5. Generate Images
```bash
# Command line
python inference.py --num_images 10

# Web interface
streamlit run app.py

# REST API
uvicorn api:app --port 8000
```

## 📁 Project Structure

```
├── train.py              # Training pipeline
├── inference.py          # Image generation
├── evaluation.py         # Quality metrics
├── monitoring.py         # Production monitoring
├── app.py                # Streamlit web UI
├── api.py                # FastAPI backend
├── generator.py          # Generator network
├── discriminator.py      # Discriminator network
├── vanilla_gan.py        # Combined GAN model
├── plantdoc_loader.py    # Data loader
├── config.yaml           # Configuration
├── requirements.txt      # Dependencies
├── Dockerfile            # Container config
└── docs/                 # Documentation
    ├── SRS.md            # Software Requirements
    ├── HLD.md            # High-Level Design
    ├── LLD.md            # Low-Level Design
    └── User_Manual.md    # Usage Guide
```

## 🧠 Model Architecture

| Component | Architecture | Parameters |
|-----------|-------------|------------|
| Generator | 4 ConvTranspose layers | 3.6M |
| Discriminator | 4 Conv layers | 2.7M |
| **Total** | | **6.3M** |

**Input**: 100-dim latent vector → **Output**: 64×64 RGB image

## 📊 Dataset

**PlantDoc Leaf Disease Dataset**
- 2,342 training images
- 236 test images  
- 28 disease classes
- 13 plant species

## 🎯 Usage

### Streamlit Web Interface
```bash
streamlit run app.py
```
Open http://localhost:8501

Features:
- Generate 1-64 images
- Download as ZIP
- Latent interpolation

### REST API
```bash
uvicorn api:app --port 8000
```
Open http://localhost:8000/docs for Swagger UI

**Endpoints:**
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/generate` | Generate images (base64) |
| GET | `/generate/zip` | Download as ZIP |
| POST | `/interpolate` | Latent interpolation |

### Command Line
```bash
# Generate 10 images
python inference.py --num_images 10 --output_dir generated_images

# With seed for reproducibility
python inference.py --num_images 5 --seed 42

# Create ZIP file
python inference.py --num_images 50 --zip
```

## ⚙️ Training

### Configuration (`config.yaml`)
```yaml
training:
  num_epochs: 100
  batch_size: 64
  lr: 0.0002
  label_smoothing: true
  gradient_clip: true
```

### Monitor Training
```bash
tensorboard --logdir=logs/
```

### Resume Training
Checkpoints are saved every 10 epochs in `checkpoints/`.

## 📈 Evaluation

Run evaluation metrics:
```bash
python evaluation.py --checkpoint checkpoints/gan_final.pt
```

Metrics:
- Classifier Realism Score
- Diversity Score
- FID (Fréchet Inception Distance)
- Mode Collapse Detection

## 🐳 Docker

```bash
# Build image
docker build -t gan-generator .

# Run FastAPI
docker run -p 8000:8000 gan-generator

# Run Streamlit
docker run -p 8501:8501 gan-generator streamlit run app.py --server.port 8501
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [SRS](docs/SRS.md) | Software Requirements Specification |
| [HLD](docs/HLD.md) | High-Level Design |
| [LLD](docs/LLD.md) | Low-Level Design |
| [User Manual](docs/User_Manual.md) | Detailed usage guide |
| [Architecture](docs/Model_Architecture.md) | Model diagrams |

## 🛠️ Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

See `requirements.txt` for full list.

## 🚀 Deployment

### Hugging Face Spaces (Recommended)

The app is deployed at: **[https://huggingface.co/spaces/vighnesh2190/generate-leaves-images](https://huggingface.co/spaces/vighnesh2190/generate-leaves-images)**

To deploy your own:
1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Docker** SDK with **Streamlit** template
3. Upload: `app.py`, `G_final.pt`, `requirements.txt`
4. Update `Dockerfile` to copy your files
5. Wait for build to complete

### Local Docker
```bash
docker build -t gan-generator .
docker run -p 8501:8501 gan-generator
```

## 📜 License

MIT License

## 🙏 Acknowledgments

- [PlantDoc Dataset](https://github.com/pratikkayal/PlantDoc-Dataset) by Pratik Kayal
- PyTorch team for the deep learning framework

---

Made with ❤️ for Agricultural AI Research
