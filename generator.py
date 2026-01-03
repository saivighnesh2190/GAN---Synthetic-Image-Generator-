import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Vanilla GAN Generator
    Input: latent noise vector z ∈ R^100
    Output: synthetic image (3, 64, 64)
    """
    
    def __init__(self, latent_dim=100, img_channels=3, feature_maps=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Project and reshape: 100 -> 4*4*512
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * feature_maps * 8),
            nn.BatchNorm1d(4 * 4 * feature_maps * 8),
            nn.ReLU(True)
        )
        
        # ConvTranspose blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.conv_blocks = nn.Sequential(
            # Block 1: 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # Block 2: 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # Block 3: 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # Output: 32x32 -> 64x64
            nn.ConvTranspose2d(feature_maps, img_channels, 
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()  # Output range: [-1, 1]
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        # z: (batch_size, latent_dim)
        x = self.fc(z)
        x = x.view(-1, 512, 4, 4)  # Reshape to (batch, 512, 4, 4)
        x = self.conv_blocks(x)
        return x  # (batch_size, 3, 64, 64)


if __name__ == "__main__":
    # Test Generator
    print("=" * 50)
    print("Testing Generator")
    print("=" * 50)
    
    G = Generator(latent_dim=100, img_channels=3)
    print(f"\nGenerator Architecture:")
    print(G)
    
    # Test forward pass
    z = torch.randn(4, 100)  # Batch of 4 noise vectors
    fake_images = G(z)
    
    print(f"\nInput shape: {z.shape}")
    print(f"Output shape: {fake_images.shape}")
    print(f"Output range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in G.parameters())
    print(f"\nTotal parameters: {total_params:,}")
