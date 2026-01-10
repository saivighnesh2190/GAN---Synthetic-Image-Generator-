# User Manual
## Synthetic Leaf Disease Image Generator

---

## 1. Introduction

Welcome to the Synthetic Leaf Disease Image Generator! This application uses a Generative Adversarial Network (GAN) to create realistic synthetic images of diseased plant leaves.

### What Can You Do?
- Generate synthetic leaf disease images
- Download images individually or as ZIP
- Visualize latent space interpolation
- Access via web interface or REST API

---

## 2. Quick Start

### 2.1 Starting the Application

**Option 1: Streamlit Web Interface (Recommended)**
```bash
cd /path/to/Project1
streamlit run app.py
```
Open http://localhost:8501 in your browser.

**Option 2: FastAPI REST API**
```bash
uvicorn api:app --reload --port 8000
```
API docs at http://localhost:8000/docs

**Option 3: Command Line**
```bash
python3 inference.py --num_images 10 --output_dir my_images
```

---

## 3. Using the Streamlit Interface

### 3.1 Main Interface

![Streamlit Interface](images/streamlit_ui.png)

### 3.2 Controls

| Control | Description |
|---------|-------------|
| **Number of Images** | Slider to select 1-64 images |
| **Gallery Columns** | Dropdown to adjust grid layout |
| **Use Random Seed** | Checkbox for reproducible generation |
| **Seed Value** | Number input for specific seed |
| **Generate Images** | Button to create new images |

### 3.3 Step-by-Step Guide

1. **Adjust Settings**
   - Use the sidebar to set the number of images
   - Optionally enable random seed for reproducibility

2. **Generate Images**
   - Click the "🚀 Generate Images" button
   - Wait for images to appear in the gallery

3. **Download Images**
   - Click "📥 Download All Images (ZIP)" for all images
   - Or expand "Download Individual Images" for single files

4. **Latent Interpolation**
   - Enable "Show Latent Interpolation" in sidebar
   - View smooth morphing between random images

---

## 4. Using the REST API

### 4.1 API Documentation

Access interactive docs at: http://localhost:8000/docs

### 4.2 Common Endpoints

**Generate Images:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{"num_images": 5, "seed": 42}'
```

**Download as ZIP:**
```bash
curl "http://localhost:8000/generate/zip?num_images=10" \
     --output images.zip
```

**Health Check:**
```bash
curl "http://localhost:8000/health"
```

### 4.3 Python Client Example

```python
import requests

# Generate images
response = requests.post(
    "http://localhost:8000/generate",
    json={"num_images": 5}
)
data = response.json()

# Images are base64 encoded
for i, img_data in enumerate(data['images']):
    print(f"Image {i+1}: {img_data[:50]}...")
```

---

## 5. Command Line Interface

### 5.1 Basic Usage

```bash
# Generate 10 images
python3 inference.py --num_images 10

# Generate with specific seed
python3 inference.py --num_images 5 --seed 42

# Save to custom directory
python3 inference.py --num_images 20 --output_dir my_images

# Create ZIP file
python3 inference.py --num_images 50 --zip
```

### 5.2 Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | `checkpoints/G_final.pt` | Model file |
| `--num_images` | 10 | Number to generate |
| `--output_dir` | `generated_images` | Output folder |
| `--seed` | None | Random seed |
| `--zip` | False | Create ZIP file |

---

## 6. Training Your Own Model

### 6.1 Prepare Dataset

1. Place images in `leaf_dataset/PlantDoc-Dataset/train/`
2. Organize by class: `train/ClassName/image.jpg`

### 6.2 Preprocess Data

```bash
python3 plantdoc_loader.py
```

### 6.3 Start Training

```bash
python3 train.py
```

### 6.4 Monitor Training

```bash
tensorboard --logdir=logs/
```

---

## 7. Troubleshooting

### Problem: Images look like noise

**Cause:** Model not trained enough or wrong checkpoint loaded.

**Solution:**
1. Check that `checkpoints/G_final.pt` exists
2. Ensure it's from the correct training session
3. Train for more epochs if needed

### Problem: Streamlit won't start

**Cause:** Port already in use or dependencies missing.

**Solution:**
```bash
pip install streamlit
streamlit run app.py --server.port 8502  # Different port
```

### Problem: API returns 503 error

**Cause:** Model failed to load.

**Solution:**
1. Check checkpoint file exists
2. Restart the API server
3. Check logs for errors

---

## 8. FAQ

**Q: How long does generation take?**
A: ~50-100ms per image on CPU, ~10-20ms on GPU.

**Q: Can I use my own dataset?**
A: Yes! Follow the dataset structure in Section 6.1.

**Q: What image size is generated?**
A: 64x64 RGB images by default.

**Q: Is it safe to use generated images?**
A: The system includes memorization detection to ensure privacy.

---

## 9. Support

For issues or questions:
1. Check the troubleshooting section above
2. Review logs in `logs/` directory
3. Contact the development team

---

*Last Updated: January 2026*
