import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np

class PreprocessedDataset(Dataset):
    """Dataset that loads from preprocessed tensors"""
    
    def __init__(self, preprocessed_file):
        print(f"Loading preprocessed data from {preprocessed_file}...")
        
        # Load preprocessed data
        data = torch.load(preprocessed_file)
        
        self.images = data['images']
        self.image_ids = data['image_ids']
        self.img_size = data['img_size']
        
        print(f"✓ Loaded {len(self.images)} preprocessed images")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Memory usage: {self.images.element_size() * self.images.nelement() / (1024**2):.2f} MB\n")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {
            'image': self.images[idx],
            'image_id': self.image_ids[idx]
        }


def get_preprocessed_dataloader(batch_size=16, shuffle=True):
    """
    Create DataLoader from preprocessed data
    Much faster than loading from raw images!
    """
    dataset = PreprocessedDataset('preprocessed_data/preprocessed_images.pt')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )
    
    return dataloader


# Test the preprocessed dataloader
if __name__ == "__main__":
    print("="*70)
    print("Testing Preprocessed DataLoader")
    print("="*70 + "\n")
    
    # Create dataloader
    dataloader = get_preprocessed_dataloader(batch_size=16, shuffle=True)
    
    # Test loading
    print("Testing batch loading...")
    for i, batch in enumerate(dataloader):
        images = batch['image']
        print(f"Batch {i+1}:")
        print(f"  Shape: {images.shape}")
        print(f"  Range: [{images.min():.3f}, {images.max():.3f}]")
        
        if i == 2:  # Test 3 batches
            break
    
    print("\n✓ Preprocessed dataloader working correctly!")
    print("You can now use this for GAN training.")
