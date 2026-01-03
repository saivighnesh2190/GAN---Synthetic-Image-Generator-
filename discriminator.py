import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """
    Vanilla GAN Discriminator
    Input: image (real/fake) of shape (3, 64, 64)
    Output: probability (real=1, fake=0)
    """
    
    def __init__(self, img_channels=3, feature_maps=64):
        super(Discriminator, self).__init__()
        
        # Conv blocks: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv_blocks = nn.Sequential(
            # Block 1: 64x64 -> 32x32
            nn.Conv2d(img_channels, feature_maps, 
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 2: 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, 
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 3: 16x16 -> 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Block 4: 8x8 -> 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Dense + Sigmoid output
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_maps * 8 * 4 * 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        # img: (batch_size, 3, 64, 64)
        x = self.conv_blocks(img)
        x = self.fc(x)
        return x  # (batch_size, 1)


if __name__ == "__main__":
    # Test Discriminator
    print("=" * 50)
    print("Testing Discriminator")
    print("=" * 50)
    
    D = Discriminator(img_channels=3)
    print(f"\nDiscriminator Architecture:")
    print(D)
    
    # Test forward pass
    fake_images = torch.randn(4, 3, 64, 64)  # Batch of 4 images
    outputs = D(fake_images)
    
    print(f"\nInput shape: {fake_images.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output values: {outputs.squeeze().tolist()}")
    
    # Count parameters
    total_params = sum(p.numel() for p in D.parameters())
    print(f"\nTotal parameters: {total_params:,}")
