# =============================================================================
# GAN Synthetic Image Generator - Docker Configuration
# =============================================================================

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py ./
COPY config.yaml ./
COPY checkpoints/ ./checkpoints/

# Expose ports
EXPOSE 8000 8501

# Default command (FastAPI)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

# Alternative commands:
# For Streamlit: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
