# Low-Level Design (LLD) Document
## Synthetic Leaf Disease Image Generator

---

## 1. Model Architecture Details

### 1.1 Generator Architecture

```
Input: z ∈ R^100 (latent vector)
Output: Image ∈ R^(3×64×64)

Layer Structure:
┌────────────────────────────────────────────────────────────────┐
│  Input: z (100) ──▶ Linear ──▶ Reshape (512×4×4)              │
├────────────────────────────────────────────────────────────────┤
│  ConvTranspose2d(512→256, 4×4, stride=2, padding=1)           │
│  BatchNorm2d(256) ──▶ ReLU                                    │
│  Output: (256×8×8)                                             │
├────────────────────────────────────────────────────────────────┤
│  ConvTranspose2d(256→128, 4×4, stride=2, padding=1)           │
│  BatchNorm2d(128) ──▶ ReLU                                    │
│  Output: (128×16×16)                                           │
├────────────────────────────────────────────────────────────────┤
│  ConvTranspose2d(128→64, 4×4, stride=2, padding=1)            │
│  BatchNorm2d(64) ──▶ ReLU                                     │
│  Output: (64×32×32)                                            │
├────────────────────────────────────────────────────────────────┤
│  ConvTranspose2d(64→3, 4×4, stride=2, padding=1)              │
│  Tanh                                                          │
│  Output: (3×64×64) ∈ [-1, 1]                                   │
└────────────────────────────────────────────────────────────────┘

Total Parameters: 3,600,256
```

### 1.2 Discriminator Architecture

```
Input: Image ∈ R^(3×64×64)
Output: Probability ∈ [0, 1]

Layer Structure:
┌────────────────────────────────────────────────────────────────┐
│  Input: (3×64×64)                                              │
├────────────────────────────────────────────────────────────────┤
│  Conv2d(3→64, 4×4, stride=2, padding=1)                       │
│  LeakyReLU(0.2)                                                │
│  Output: (64×32×32)                                            │
├────────────────────────────────────────────────────────────────┤
│  Conv2d(64→128, 4×4, stride=2, padding=1)                     │
│  BatchNorm2d(128) ──▶ LeakyReLU(0.2)                          │
│  Output: (128×16×16)                                           │
├────────────────────────────────────────────────────────────────┤
│  Conv2d(128→256, 4×4, stride=2, padding=1)                    │
│  BatchNorm2d(256) ──▶ LeakyReLU(0.2)                          │
│  Output: (256×8×8)                                             │
├────────────────────────────────────────────────────────────────┤
│  Conv2d(256→512, 4×4, stride=2, padding=1)                    │
│  BatchNorm2d(512) ──▶ LeakyReLU(0.2)                          │
│  Output: (512×4×4)                                             │
├────────────────────────────────────────────────────────────────┤
│  Flatten ──▶ Linear(8192→1) ──▶ Sigmoid                       │
│  Output: Probability                                           │
└────────────────────────────────────────────────────────────────┘

Total Parameters: 2,765,569
```

---

## 2. Training Algorithm

### 2.1 Training Loop Pseudocode

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        real_images = batch['image']
        batch_size = real_images.size(0)
        
        # ========== Train Discriminator ==========
        # Real images
        real_labels = smooth_real_label  # 0.9 with label smoothing
        real_output = D(real_images)
        d_loss_real = BCELoss(real_output, real_labels)
        
        # Fake images
        z = random_normal(batch_size, latent_dim)
        fake_images = G(z)
        fake_labels = smooth_fake_label  # 0.0
        fake_output = D(fake_images.detach())
        d_loss_fake = BCELoss(fake_output, fake_labels)
        
        # Update D
        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        clip_gradients(D, max_norm=1.0)
        optimizer_D.step()
        
        # ========== Train Generator ==========
        z = random_normal(batch_size, latent_dim)
        fake_images = G(z)
        fake_output = D(fake_images)
        g_loss = BCELoss(fake_output, real_labels)  # Fool D
        
        # Update G
        optimizer_G.zero_grad()
        g_loss.backward()
        clip_gradients(G, max_norm=1.0)
        optimizer_G.step()
        
        log_metrics(g_loss, d_loss)
    
    if epoch % save_interval == 0:
        save_checkpoint()
        save_sample_images()
```

### 2.2 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| latent_dim | 100 | Latent vector dimension |
| lr | 0.0002 | Learning rate |
| beta1 | 0.5 | Adam beta1 |
| beta2 | 0.999 | Adam beta2 |
| batch_size | 64 | Training batch size |
| num_epochs | 100 | Total training epochs |
| smooth_real | 0.9 | Label smoothing for real |
| smooth_fake | 0.0 | Label smoothing for fake |
| gradient_clip | 1.0 | Max gradient norm |

---

## 3. API Endpoint Specifications

### 3.1 POST /generate

**Request:**
```json
{
    "num_images": 10,
    "seed": 42
}
```

**Response:**
```json
{
    "success": true,
    "num_images": 10,
    "images": ["data:image/png;base64,...", ...],
    "message": "Successfully generated 10 images"
}
```

### 3.2 GET /generate/zip

**Query Parameters:**
- `num_images` (int): Number of images (1-100)
- `seed` (int, optional): Random seed

**Response:** ZIP file (application/zip)

### 3.3 POST /interpolate

**Request:**
```json
{
    "num_steps": 10,
    "seed1": 42,
    "seed2": 123
}
```

**Response:** Array of base64 images showing interpolation

---

## 4. Database Schema (Monitoring)

### 4.1 Inference Metrics (JSON)

```json
{
    "total_requests": 1000,
    "successful_requests": 995,
    "failed_requests": 5,
    "total_images_generated": 5000,
    "latency_history": [
        {
            "timestamp": "2026-01-10T12:00:00",
            "num_images": 10,
            "latency_ms": 250.5
        }
    ],
    "daily_stats": {
        "2026-01-10": {
            "requests": 100,
            "images": 500,
            "avg_latency_ms": 245.3
        }
    },
    "errors": []
}
```

### 4.2 Model Versions (JSON)

```json
{
    "versions": [
        {
            "version": "v1",
            "created_at": "2026-01-10T10:00:00",
            "metrics": {
                "fid_score": 150.5,
                "diversity_score": 0.85
            },
            "notes": "Initial training",
            "generator_hash": "a1b2c3d4",
            "discriminator_hash": "e5f6g7h8"
        }
    ],
    "current": "v1"
}
```

---

## 5. Error Handling

| Error Code | Description | Resolution |
|------------|-------------|------------|
| 400 | Invalid input parameters | Validate num_images range |
| 500 | Generation failed | Check model loading |
| 503 | Model not loaded | Restart service |

---

## 6. File Structure

```
Project1/
├── app.py              # Streamlit UI
├── api.py              # FastAPI backend
├── config.yaml         # Configuration
├── data_loader.py      # Original data loader
├── plantdoc_loader.py  # PlantDoc data loader
├── discriminator.py    # D network
├── generator.py        # G network
├── vanilla_gan.py      # Combined GAN
├── train.py            # Training pipeline
├── inference.py        # Production inference
├── evaluation.py       # Metrics & visualization
├── monitoring.py       # Runtime monitoring
├── requirements.txt    # Dependencies
├── Dockerfile          # Container config
├── checkpoints/        # Model weights
├── samples/            # Generated samples
├── logs/               # Training logs
├── figures/            # Evaluation figures
├── docs/               # Documentation
└── leaf_dataset/       # PlantDoc dataset
```
