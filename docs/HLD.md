# High-Level Design (HLD) Document
## Synthetic Leaf Disease Image Generator

---

## 1. System Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Streamlit   │  │   FastAPI    │  │     CLI      │          │
│  │     UI       │  │     API      │  │  Interface   │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
└─────────┼─────────────────┼─────────────────┼──────────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌────────────────────────────────────────────────────────────────┐
│                    INFERENCE LAYER                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 GANInference Engine                       │  │
│  │  • Load trained Generator model                          │  │
│  │  • Generate images from latent vectors                   │  │
│  │  • Latent space interpolation                            │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                                  │
│  ┌─────────────────┐           ┌─────────────────┐             │
│  │    Generator    │           │  Discriminator  │             │
│  │   (G_final.pt)  │           │   (D_final.pt)  │             │
│  └─────────────────┘           └─────────────────┘             │
└────────────────────────────────────────────────────────────────┘
          │
          ▼
┌────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  PlantDoc    │  │ Preprocessed │  │  Generated   │          │
│  │   Dataset    │  │   Tensors    │  │   Outputs    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         GAN SYSTEM                               │
│                                                                  │
│   ┌───────────────┐          ┌───────────────┐                  │
│   │  data_loader  │─────────▶│    train      │                  │
│   │     .py       │          │     .py       │                  │
│   └───────────────┘          └───────┬───────┘                  │
│          │                           │                          │
│          ▼                           ▼                          │
│   ┌───────────────┐          ┌───────────────┐                  │
│   │   generator   │◀─────────│  vanilla_gan  │                  │
│   │     .py       │          │     .py       │                  │
│   └───────────────┘          └───────────────┘                  │
│          │                           ▲                          │
│          │                           │                          │
│   ┌───────────────┐          ┌───────────────┐                  │
│   │ discriminator │─────────▶│  inference    │                  │
│   │     .py       │          │     .py       │                  │
│   └───────────────┘          └───────┬───────┘                  │
│                                      │                          │
│                                      ▼                          │
│   ┌───────────────┐          ┌───────────────┐                  │
│   │  evaluation   │◀─────────│     app       │                  │
│   │     .py       │          │     .py       │                  │
│   └───────────────┘          └───────────────┘                  │
│          │                           │                          │
│          ▼                           ▼                          │
│   ┌───────────────┐          ┌───────────────┐                  │
│   │  monitoring   │          │     api       │                  │
│   │     .py       │          │     .py       │                  │
│   └───────────────┘          └───────────────┘                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow Diagram

### Training Flow
```
Raw Images ──▶ Preprocessing ──▶ DataLoader ──▶ Training Loop
                                                      │
                    ┌─────────────────────────────────┘
                    ▼
              ┌─────────┐      ┌─────────────┐
              │ Real    │─────▶│             │
              │ Images  │      │ Discriminator│──▶ D_loss
              └─────────┘      │     (D)     │
                    ▲          └─────────────┘
                    │                ▲
              ┌─────────┐            │
              │ Latent  │──▶ Generator ──▶ Fake Images
              │ Noise z │      (G)              │
              └─────────┘                       │
                                                ▼
                                           G_loss
```

### Inference Flow
```
User Request ──▶ Generate N images ──▶ Random z vectors
                                              │
                                              ▼
                                       ┌───────────┐
                                       │ Generator │
                                       │    (G)    │
                                       └─────┬─────┘
                                             │
                                             ▼
                                    Synthetic Images
                                             │
                    ┌───────────┬────────────┼────────────┐
                    ▼           ▼            ▼            ▼
               Display     Download       ZIP         API
              (Gallery)     (PNG)      Archive     Response
```

---

## 4. Module Descriptions

| Module | File | Responsibility |
|--------|------|----------------|
| Data Loader | `data_loader.py`, `plantdoc_loader.py` | Load and preprocess images |
| Generator | `generator.py` | Create synthetic images from noise |
| Discriminator | `discriminator.py` | Classify real vs fake images |
| GAN Model | `vanilla_gan.py` | Combine G and D with training logic |
| Training | `train.py` | Training loop with logging |
| Inference | `inference.py` | Production image generation |
| Evaluation | `evaluation.py` | Quality metrics and visualization |
| Monitoring | `monitoring.py` | Runtime monitoring and versioning |
| Web UI | `app.py` | Streamlit interface |
| REST API | `api.py` | FastAPI endpoints |

---

## 5. Technology Stack

| Layer | Technology |
|-------|------------|
| Deep Learning | PyTorch 2.x |
| Data Processing | NumPy, Pandas, PIL |
| Visualization | Matplotlib |
| Web Framework | Streamlit, FastAPI |
| API Server | Uvicorn |
| Logging | TensorBoard, Python logging |
| Containerization | Docker |

---

## 6. Deployment Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        HOST MACHINE                           │
│                                                               │
│   ┌─────────────────┐    ┌─────────────────┐                 │
│   │   Streamlit     │    │    FastAPI      │                 │
│   │   Port 8501     │    │   Port 8000     │                 │
│   └────────┬────────┘    └────────┬────────┘                 │
│            │                      │                          │
│            └──────────┬───────────┘                          │
│                       ▼                                       │
│              ┌─────────────────┐                             │
│              │  GAN Inference  │                             │
│              │     Engine      │                             │
│              └────────┬────────┘                             │
│                       │                                       │
│              ┌────────▼────────┐                             │
│              │   Model Files   │                             │
│              │  (checkpoints/) │                             │
│              └─────────────────┘                             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```
