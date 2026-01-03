import torch
import torch.nn as nn
import torch.optim as optim
from generator import Generator
from discriminator import Discriminator


class VanillaGAN:
    """
    Vanilla GAN - Combined Generator and Discriminator
    Loss: Binary Cross Entropy
    Optimizer: Adam (lr=0.0002, beta1=0.5)
    """
    
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64, 
                 lr=0.0002, beta1=0.5, device='cpu'):
        
        self.latent_dim = latent_dim
        self.device = device
        
        # Initialize Generator and Discriminator
        self.G = Generator(latent_dim, img_channels, feature_maps).to(device)
        self.D = Discriminator(img_channels, feature_maps).to(device)
        
        # Loss function: Binary Cross Entropy
        self.criterion = nn.BCELoss()
        
        # Optimizers: Adam with lr=0.0002, beta1=0.5
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, 0.999))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, 0.999))
        
        # Labels
        self.real_label = 1.0
        self.fake_label = 0.0
    
    def train_discriminator(self, real_images):
        """
        Train Discriminator: D_loss = BCE(real) + BCE(fake)
        """
        batch_size = real_images.size(0)
        
        # Labels
        real_labels = torch.full((batch_size, 1), self.real_label, device=self.device)
        fake_labels = torch.full((batch_size, 1), self.fake_label, device=self.device)
        
        # Zero gradients
        self.optimizer_D.zero_grad()
        
        # Train on real images
        real_outputs = self.D(real_images)
        d_loss_real = self.criterion(real_outputs, real_labels)
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.G(z)
        
        # Train on fake images
        fake_outputs = self.D(fake_images.detach())
        d_loss_fake = self.criterion(fake_outputs, fake_labels)
        
        # Total D loss
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        self.optimizer_D.step()
        
        return d_loss.item(), real_outputs.mean().item(), fake_outputs.mean().item()
    
    def train_generator(self, batch_size):
        """
        Train Generator: G_loss = BCE(fake labels=1)
        """
        # Labels (we want discriminator to think fake images are real)
        real_labels = torch.full((batch_size, 1), self.real_label, device=self.device)
        
        # Zero gradients
        self.optimizer_G.zero_grad()
        
        # Generate fake images
        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake_images = self.G(z)
        
        # Get discriminator output
        outputs = self.D(fake_images)
        
        # G wants D to think fake images are real
        g_loss = self.criterion(outputs, real_labels)
        g_loss.backward()
        self.optimizer_G.step()
        
        return g_loss.item()
    
    def generate(self, num_images=1):
        """Generate fake images"""
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.latent_dim, device=self.device)
            fake_images = self.G(z)
        self.G.train()
        return fake_images
    
    def save(self, path):
        """Save model checkpoints"""
        torch.save({
            'generator': self.G.state_dict(),
            'discriminator': self.D.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D': self.optimizer_D.state_dict(),
        }, path)
        print(f"✓ Model saved to {path}")
    
    def load(self, path):
        """Load model checkpoints"""
        checkpoint = torch.load(path, map_location=self.device)
        self.G.load_state_dict(checkpoint['generator'])
        self.D.load_state_dict(checkpoint['discriminator'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D'])
        print(f"✓ Model loaded from {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Vanilla GAN - Model Test")
    print("=" * 60)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Initialize GAN
    gan = VanillaGAN(
        latent_dim=100,
        img_channels=3,
        feature_maps=64,
        lr=0.0002,
        beta1=0.5,
        device=device
    )
    
    print(f"\n✓ Generator initialized")
    print(f"✓ Discriminator initialized")
    print(f"✓ Loss: BCELoss")
    print(f"✓ Optimizer: Adam (lr=0.0002, beta1=0.5)")
    
    # Test with fake batch
    print("\n" + "=" * 60)
    print("Testing Training Step")
    print("=" * 60)
    
    fake_real_images = torch.randn(16, 3, 64, 64).to(device)
    
    # Train D
    d_loss, d_real, d_fake = gan.train_discriminator(fake_real_images)
    print(f"\nDiscriminator Loss: {d_loss:.4f}")
    print(f"  D(real): {d_real:.4f}")
    print(f"  D(fake): {d_fake:.4f}")
    
    # Train G
    g_loss = gan.train_generator(batch_size=16)
    print(f"\nGenerator Loss: {g_loss:.4f}")
    
    # Generate images
    print("\n" + "=" * 60)
    print("Testing Image Generation")
    print("=" * 60)
    
    generated = gan.generate(num_images=4)
    print(f"\nGenerated images shape: {generated.shape}")
    print(f"Value range: [{generated.min():.3f}, {generated.max():.3f}]")
    
    # Model summary
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    
    g_params = sum(p.numel() for p in gan.G.parameters())
    d_params = sum(p.numel() for p in gan.D.parameters())
    
    print(f"\nGenerator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    print(f"Total parameters: {g_params + d_params:,}")
    
    print("\n✓ Vanilla GAN ready for training!")
