# Software Requirements Specification (SRS)
## Synthetic Leaf Disease Image Generator using Vanilla GAN

### 1. Introduction

#### 1.1 Purpose
This document describes the software requirements for the Synthetic Leaf Disease Image Generator system, a GAN-based application for generating realistic plant disease images.

#### 1.2 Scope
The system generates synthetic leaf disease images using a trained Vanilla GAN model. It includes training pipelines, evaluation tools, and deployment interfaces.

#### 1.3 Definitions
| Term | Definition |
|------|------------|
| GAN | Generative Adversarial Network |
| Generator (G) | Neural network that creates synthetic images |
| Discriminator (D) | Neural network that classifies real vs fake images |
| Latent Space | Random noise vector space used as Generator input |
| FID | Fréchet Inception Distance (quality metric) |

---

### 2. System Overview

#### 2.1 System Architecture
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Training   │────▶│  Trained     │────▶│  Inference  │
│  Pipeline   │     │  GAN Model   │     │  Pipeline   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                                        │
       ▼                                        ▼
┌─────────────┐                          ┌─────────────┐
│  Evaluation │                          │  Deployment │
│  Pipeline   │                          │  (UI/API)   │
└─────────────┘                          └─────────────┘
```

#### 2.2 User Classes
| User Class | Description |
|------------|-------------|
| Researcher | Uses generated images for ML training |
| Developer | Integrates API into applications |
| End User | Generates images via web interface |

---

### 3. Functional Requirements

#### 3.1 Data Pipeline (FR-1)
| ID | Requirement |
|----|-------------|
| FR-1.1 | System shall load images from PlantDoc dataset |
| FR-1.2 | System shall preprocess images to 64x64 RGB |
| FR-1.3 | System shall normalize images to [-1, 1] range |
| FR-1.4 | System shall split data into train/test sets |

#### 3.2 Training Pipeline (FR-2)
| ID | Requirement |
|----|-------------|
| FR-2.1 | System shall train Generator and Discriminator alternately |
| FR-2.2 | System shall save checkpoints every N epochs |
| FR-2.3 | System shall log metrics to TensorBoard and CSV |
| FR-2.4 | System shall apply label smoothing and gradient clipping |
| FR-2.5 | System shall generate sample images during training |

#### 3.3 Inference Pipeline (FR-3)
| ID | Requirement |
|----|-------------|
| FR-3.1 | System shall generate N images from random noise |
| FR-3.2 | System shall support reproducible generation with seeds |
| FR-3.3 | System shall export images as PNG files |
| FR-3.4 | System shall create ZIP archives of generated images |
| FR-3.5 | System shall perform latent space interpolation |

#### 3.4 Evaluation Pipeline (FR-4)
| ID | Requirement |
|----|-------------|
| FR-4.1 | System shall compute Classifier Realism Score |
| FR-4.2 | System shall compute Diversity Score |
| FR-4.3 | System shall compute FID Proxy Score |
| FR-4.4 | System shall detect mode collapse |
| FR-4.5 | System shall generate t-SNE visualization |

#### 3.5 Deployment (FR-5)
| ID | Requirement |
|----|-------------|
| FR-5.1 | System shall provide Streamlit web interface |
| FR-5.2 | System shall provide FastAPI REST endpoints |
| FR-5.3 | System shall support image download functionality |

#### 3.6 Monitoring (FR-6)
| ID | Requirement |
|----|-------------|
| FR-6.1 | System shall log inference latency |
| FR-6.2 | System shall track request frequency |
| FR-6.3 | System shall support model versioning |
| FR-6.4 | System shall detect training data memorization |

---

### 4. Non-Functional Requirements

#### 4.1 Performance
| ID | Requirement |
|----|-------------|
| NFR-1 | Image generation shall complete in <500ms per image on GPU |
| NFR-2 | Web interface shall load within 3 seconds |
| NFR-3 | API response time shall be <1 second for single image |

#### 4.2 Reliability
| ID | Requirement |
|----|-------------|
| NFR-4 | System shall handle invalid inputs gracefully |
| NFR-5 | Training shall resume from checkpoints on failure |
| NFR-6 | System shall log all errors for debugging |

#### 4.3 Usability
| ID | Requirement |
|----|-------------|
| NFR-7 | Web interface shall be intuitive with clear controls |
| NFR-8 | API documentation shall be auto-generated (Swagger) |

#### 4.4 Security
| ID | Requirement |
|----|-------------|
| NFR-9 | Generated images shall not memorize training data |
| NFR-10 | API shall support CORS for cross-origin requests |

---

### 5. System Interfaces

#### 5.1 User Interface
- Streamlit web application at port 8501
- Controls: Number slider, Generate button, Download button
- Display: Image gallery grid

#### 5.2 API Interface
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Generate images (base64) |
| `/generate/zip` | GET | Generate and download ZIP |
| `/health` | GET | Health check |
| `/interpolate` | POST | Latent interpolation |

---

### 6. Constraints
- Training requires significant computational resources
- CPU training is slower than GPU training
- Image quality depends on training data diversity

---

### 7. Appendix

#### 7.1 Dataset Information
- **Name**: PlantDoc Leaf Disease Dataset
- **Size**: 2,342 training images, 236 test images
- **Classes**: 28 disease categories
- **Source**: GitHub (pratikkayal/PlantDoc-Dataset)
