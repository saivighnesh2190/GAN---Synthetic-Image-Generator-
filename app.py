"""
Module 5: Streamlit UI
=======================
Interactive web interface for the GAN Synthetic Image Generator.

Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime

# Import inference engine
from inference import GANInference


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="GAN Synthetic Image Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
    .image-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 10px;
        margin: 5px;
        background: #f9f9f9;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# INITIALIZE MODEL (cached)
# =============================================================================

@st.cache_resource
def load_model():
    """Load the GAN model (cached for performance)"""
    return GANInference(checkpoint_path='checkpoints/G_final.pt')


# =============================================================================
# SIDEBAR CONTROLS
# =============================================================================

def sidebar_controls():
    """Render sidebar with generation controls"""
    st.sidebar.markdown("## ⚙️ Generation Settings")
    
    # Number of images
    num_images = st.sidebar.slider(
        "Number of Images",
        min_value=1,
        max_value=64,
        value=9,
        step=1,
        help="How many synthetic images to generate"
    )
    
    # Grid columns
    grid_cols = st.sidebar.selectbox(
        "Gallery Columns",
        options=[2, 3, 4, 5, 6],
        index=1,
        help="Number of columns in the image gallery"
    )
    
    # Seed control
    use_seed = st.sidebar.checkbox("Use Random Seed", value=False)
    seed = None
    if use_seed:
        seed = st.sidebar.number_input(
            "Seed Value",
            min_value=0,
            max_value=999999,
            value=42,
            help="Set seed for reproducible generation"
        )
    
    st.sidebar.markdown("---")
    
    # Advanced options
    st.sidebar.markdown("## 🔧 Advanced Options")
    
    show_interpolation = st.sidebar.checkbox(
        "Show Latent Interpolation",
        value=False,
        help="Generate smooth morphing between two images"
    )
    
    interp_steps = 10
    if show_interpolation:
        interp_steps = st.sidebar.slider(
            "Interpolation Steps",
            min_value=5,
            max_value=20,
            value=10
        )
    
    return {
        'num_images': num_images,
        'grid_cols': grid_cols,
        'seed': seed,
        'show_interpolation': show_interpolation,
        'interp_steps': interp_steps
    }


# =============================================================================
# MAIN CONTENT
# =============================================================================

def display_image_gallery(images, cols=3):
    """Display images in a responsive grid gallery"""
    rows = (len(images) + cols - 1) // cols
    
    for row in range(rows):
        columns = st.columns(cols)
        for col_idx in range(cols):
            img_idx = row * cols + col_idx
            if img_idx < len(images):
                with columns[col_idx]:
                    st.image(images[img_idx], use_container_width=True)


def create_download_button(images, filename="generated_images.zip"):
    """Create a download button for ZIP file of images"""
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    import zipfile
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        for i, img in enumerate(images):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            zf.writestr(f"generated_{timestamp}_{i:04d}.png", img_buffer.read())
    
    zip_buffer.seek(0)
    
    st.download_button(
        label="📥 Download All Images (ZIP)",
        data=zip_buffer.getvalue(),
        file_name=filename,
        mime="application/zip"
    )


def display_interpolation(engine, steps=10):
    """Display latent space interpolation"""
    st.markdown("### 🔄 Latent Space Interpolation")
    st.caption("Smooth morphing between two random latent vectors")
    
    images = engine.interpolate(num_steps=steps)
    
    # Display in a single row
    cols = st.columns(steps)
    for i, (col, img) in enumerate(zip(cols, images)):
        with col:
            st.image(img, use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 GAN Synthetic Image Generator</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Generate realistic synthetic images using a trained Vanilla GAN</p>', 
                unsafe_allow_html=True)
    
    # Load model
    try:
        engine = load_model()
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.info("Make sure you have trained the GAN and saved the checkpoint.")
        model_loaded = False
    
    if not model_loaded:
        return
    
    # Get sidebar controls
    settings = sidebar_controls()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### 🖼️ Generated Images")
    
    with col2:
        generate_btn = st.button("🚀 Generate Images", type="primary")
    
    # Session state for generated images
    if 'generated_images' not in st.session_state:
        st.session_state.generated_images = None
    
    # Generate images
    if generate_btn:
        with st.spinner("Generating images..."):
            images = engine.generate_pil(
                num_images=settings['num_images'],
                seed=settings['seed']
            )
            st.session_state.generated_images = images
    
    # Display gallery
    if st.session_state.generated_images:
        images = st.session_state.generated_images
        
        # Stats
        st.success(f"✓ Generated {len(images)} images")
        
        # Gallery
        display_image_gallery(images, cols=settings['grid_cols'])
        
        # Download button
        st.markdown("---")
        create_download_button(images)
        
        # Individual image download
        with st.expander("📁 Download Individual Images"):
            cols = st.columns(4)
            for i, img in enumerate(images):
                with cols[i % 4]:
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    st.download_button(
                        label=f"Image {i+1}",
                        data=img_buffer.getvalue(),
                        file_name=f"generated_{i+1}.png",
                        mime="image/png",
                        key=f"download_{i}"
                    )
    else:
        st.info("👆 Click 'Generate Images' to create synthetic images")
    
    # Interpolation section
    if settings['show_interpolation']:
        st.markdown("---")
        display_interpolation(engine, settings['interp_steps'])
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #888; padding: 1rem;'>
            Built with Streamlit | Vanilla GAN - Synthetic Image Generator
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
