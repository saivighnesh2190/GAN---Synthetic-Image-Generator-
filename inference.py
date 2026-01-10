"""
Module 5: Inference Script
===========================
Packaged model for deployment - load trained Generator and generate images.
"""

import os
import io
import zipfile
import torch
import numpy as np
from PIL import Image
from datetime import datetime

# Import Generator
from generator import Generator


class GANInference:
    """
    Inference wrapper for trained GAN Generator.
    Provides easy-to-use interface for image generation.
    """
    
    def __init__(self, checkpoint_path='checkpoints/G_final.pt', 
                 latent_dim=100, img_channels=3, feature_maps=64, device=None):
        """
        Initialize the inference engine.
        
        Args:
            checkpoint_path: Path to Generator weights
            latent_dim: Latent vector dimension (default: 100)
            img_channels: Number of image channels (default: 3 for RGB)
            feature_maps: Feature map multiplier (default: 64)
            device: 'cuda' or 'cpu' (auto-detect if None)
        """
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.latent_dim = latent_dim
        
        # Initialize Generator
        self.generator = Generator(
            latent_dim=latent_dim,
            img_channels=img_channels,
            feature_maps=feature_maps
        ).to(self.device)
        
        # Load weights
        if os.path.exists(checkpoint_path):
            self.generator.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print(f"✓ Loaded Generator from: {checkpoint_path}")
        else:
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            print("  Using randomly initialized Generator")
        
        # Set to evaluation mode
        self.generator.eval()
        
        print(f"✓ Inference engine ready (device: {self.device})")
    
    def generate(self, num_images=1, seed=None):
        """
        Generate synthetic images.
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility (optional)
            
        Returns:
            numpy array of images with shape (N, H, W, C) in range [0, 255]
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        with torch.no_grad():
            # Sample random latent vectors
            z = torch.randn(num_images, self.latent_dim, device=self.device)
            
            # Generate images
            fake_images = self.generator(z)
            
            # Denormalize from [-1, 1] to [0, 1]
            fake_images = fake_images * 0.5 + 0.5
            
            # Convert to numpy and change to (N, H, W, C)
            images = fake_images.cpu().numpy()
            images = np.transpose(images, (0, 2, 3, 1))
            
            # Clip and convert to uint8
            images = np.clip(images * 255, 0, 255).astype(np.uint8)
        
        return images
    
    def generate_pil(self, num_images=1, seed=None):
        """
        Generate images as PIL Image objects.
        
        Args:
            num_images: Number of images to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of PIL Image objects
        """
        images = self.generate(num_images, seed)
        return [Image.fromarray(img) for img in images]
    
    def generate_batch(self, num_images=100, batch_size=32, seed=None):
        """
        Generate a large batch of images efficiently.
        
        Args:
            num_images: Total number of images
            batch_size: Batch size for generation
            seed: Random seed
            
        Returns:
            List of PIL Image objects
        """
        if seed is not None:
            torch.manual_seed(seed)
        
        all_images = []
        remaining = num_images
        
        while remaining > 0:
            batch = min(batch_size, remaining)
            images = self.generate_pil(batch)
            all_images.extend(images)
            remaining -= batch
        
        return all_images
    
    def save_images(self, num_images=1, output_dir='generated_images', 
                    prefix='gen', seed=None):
        """
        Generate and save images to disk.
        
        Args:
            num_images: Number of images to generate
            output_dir: Output directory
            prefix: Filename prefix
            seed: Random seed
            
        Returns:
            List of saved file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        images = self.generate_pil(num_images, seed)
        paths = []
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for i, img in enumerate(images):
            filename = f"{prefix}_{timestamp}_{i:04d}.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            paths.append(filepath)
        
        print(f"✓ Saved {len(paths)} images to {output_dir}/")
        return paths
    
    def create_zip(self, num_images=10, seed=None):
        """
        Generate images and create a ZIP file in memory.
        
        Args:
            num_images: Number of images to include
            seed: Random seed
            
        Returns:
            BytesIO object containing the ZIP file
        """
        images = self.generate_pil(num_images, seed)
        
        zip_buffer = io.BytesIO()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for i, img in enumerate(images):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                zf.writestr(f"generated_{timestamp}_{i:04d}.png", img_buffer.read())
        
        zip_buffer.seek(0)
        return zip_buffer
    
    def interpolate(self, num_steps=10, seed1=None, seed2=None):
        """
        Generate interpolation between two latent vectors.
        
        Args:
            num_steps: Number of interpolation steps
            seed1: Seed for first latent vector
            seed2: Seed for second latent vector
            
        Returns:
            List of PIL Image objects showing smooth morphing
        """
        # Generate two random latent vectors
        if seed1 is not None:
            torch.manual_seed(seed1)
        z1 = torch.randn(1, self.latent_dim, device=self.device)
        
        if seed2 is not None:
            torch.manual_seed(seed2)
        z2 = torch.randn(1, self.latent_dim, device=self.device)
        
        images = []
        
        with torch.no_grad():
            for i in range(num_steps):
                alpha = i / (num_steps - 1)
                z = (1 - alpha) * z1 + alpha * z2
                
                fake = self.generator(z)
                fake = fake * 0.5 + 0.5
                fake = fake[0].cpu().numpy().transpose(1, 2, 0)
                fake = np.clip(fake * 255, 0, 255).astype(np.uint8)
                
                images.append(Image.fromarray(fake))
        
        return images


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='GAN Inference - Generate Synthetic Images')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/G_final.pt',
                        help='Path to Generator checkpoint')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--output_dir', type=str, default='generated_images',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--zip', action='store_true',
                        help='Create ZIP file instead of individual images')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("GAN Inference - Synthetic Image Generator")
    print("=" * 60 + "\n")
    
    # Initialize inference engine
    engine = GANInference(checkpoint_path=args.checkpoint)
    
    if args.zip:
        # Create ZIP file
        zip_buffer = engine.create_zip(args.num_images, args.seed)
        zip_path = os.path.join(args.output_dir, 'generated_images.zip')
        os.makedirs(args.output_dir, exist_ok=True)
        with open(zip_path, 'wb') as f:
            f.write(zip_buffer.read())
        print(f"✓ ZIP file saved: {zip_path}")
    else:
        # Save individual images
        paths = engine.save_images(
            num_images=args.num_images,
            output_dir=args.output_dir,
            seed=args.seed
        )
    
    print("\n✓ Generation complete!")


if __name__ == "__main__":
    main()
