"""
Module 5: FastAPI Backend
==========================
REST API for the GAN Synthetic Image Generator.

Run with: uvicorn api:app --reload --port 8000
"""

import os
import io
import base64
import zipfile
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import inference engine
from inference import GANInference


# =============================================================================
# FASTAPI APP CONFIGURATION
# =============================================================================

app = FastAPI(
    title="GAN Synthetic Image Generator API",
    description="""
    REST API for generating synthetic images using a trained Vanilla GAN.
    
    ## Features
    - Generate single or multiple synthetic images
    - Download images as ZIP
    - Latent space interpolation
    - Reproducible generation with seeds
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# GLOBAL MODEL INSTANCE
# =============================================================================

# Initialize model on startup
engine: Optional[GANInference] = None


@app.on_event("startup")
async def load_model():
    """Load the GAN model on startup"""
    global engine
    try:
        engine = GANInference(checkpoint_path='checkpoints/G_final.pt')
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Failed to load model: {e}")
        engine = GANInference()  # Use uninitialized model


# =============================================================================
# REQUEST/RESPONSE MODELS
# =============================================================================

class GenerationRequest(BaseModel):
    """Request model for image generation"""
    num_images: int = Field(default=1, ge=1, le=100, description="Number of images to generate")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class GenerationResponse(BaseModel):
    """Response model for image generation"""
    success: bool
    num_images: int
    images: List[str]  # Base64 encoded images
    message: str


class InterpolationRequest(BaseModel):
    """Request model for latent interpolation"""
    num_steps: int = Field(default=10, ge=2, le=50, description="Number of interpolation steps")
    seed1: Optional[int] = Field(default=None, description="Seed for first latent vector")
    seed2: Optional[int] = Field(default=None, description="Seed for second latent vector")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    device: str


# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/", response_class=JSONResponse)
async def root():
    """API root endpoint"""
    return {
        "name": "GAN Synthetic Image Generator API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "generate": "/generate",
            "generate_zip": "/generate/zip",
            "interpolate": "/interpolate"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=engine is not None,
        device=engine.device if engine else "unknown"
    )


@app.post("/generate", response_model=GenerationResponse)
async def generate_images(request: GenerationRequest):
    """
    Generate synthetic images.
    
    Returns base64-encoded PNG images.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate images
        images = engine.generate_pil(
            num_images=request.num_images,
            seed=request.seed
        )
        
        # Convert to base64
        base64_images = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode('utf-8')
            base64_images.append(f"data:image/png;base64,{b64}")
        
        return GenerationResponse(
            success=True,
            num_images=len(images),
            images=base64_images,
            message=f"Successfully generated {len(images)} images"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/single")
async def generate_single_image(seed: Optional[int] = None):
    """
    Generate a single image and return it directly as PNG.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        images = engine.generate_pil(num_images=1, seed=seed)
        
        buffer = io.BytesIO()
        images[0].save(buffer, format='PNG')
        buffer.seek(0)
        
        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=generated_image.png"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/zip")
async def generate_zip(
    num_images: int = Query(default=10, ge=1, le=100),
    seed: Optional[int] = None
):
    """
    Generate multiple images and return as ZIP file.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        zip_buffer = engine.create_zip(num_images=num_images, seed=seed)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": "attachment; filename=generated_images.zip"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/interpolate", response_model=GenerationResponse)
async def interpolate_latent_space(request: InterpolationRequest):
    """
    Generate interpolation between two latent vectors.
    
    Shows smooth morphing between two random images.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        images = engine.interpolate(
            num_steps=request.num_steps,
            seed1=request.seed1,
            seed2=request.seed2
        )
        
        # Convert to base64
        base64_images = []
        for img in images:
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            buffer.seek(0)
            b64 = base64.b64encode(buffer.read()).decode('utf-8')
            base64_images.append(f"data:image/png;base64,{b64}")
        
        return GenerationResponse(
            success=True,
            num_images=len(images),
            images=base64_images,
            message=f"Generated {len(images)} interpolation steps"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/batch")
async def generate_batch(
    num_images: int = Query(default=100, ge=1, le=1000),
    batch_size: int = Query(default=32, ge=1, le=64),
    seed: Optional[int] = None
):
    """
    Generate a large batch of images efficiently.
    
    Returns as ZIP file for larger batches.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if num_images > 100:
        return await generate_zip(num_images=num_images, seed=seed)
    
    try:
        images = engine.generate_batch(
            num_images=num_images,
            batch_size=batch_size,
            seed=seed
        )
        
        # Create ZIP for batch
        zip_buffer = io.BytesIO()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, img in enumerate(images):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                zf.writestr(f"batch_{timestamp}_{i:04d}.png", img_buffer.read())
        
        zip_buffer.seek(0)
        
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=batch_{num_images}_images.zip"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
