"""
PlantDoc Leaf Disease Data Loader
==================================
Custom data loader for PlantDoc dataset (folder-based structure).
"""

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

# PlantDoc paths
PLANTDOC_TRAIN = "leaf_dataset/PlantDoc-Dataset/train"
PLANTDOC_TEST = "leaf_dataset/PlantDoc-Dataset/test"
OUTPUT_DIR = "preprocessed_data"

# Image settings
IMG_SIZE = 64
CHANNELS = 3


# =============================================================================
# DATASET CLASS
# =============================================================================

class PlantDocDataset(Dataset):
    """Custom PyTorch Dataset for PlantDoc leaf images"""
    
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        print(f"Loading images from {root_dir}...")
        
        # Get all class folders
        class_folders = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        
        self.classes = class_folders
        self.class_to_idx = {c: i for i, c in enumerate(class_folders)}
        
        # Collect all images
        for class_name in class_folders:
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_path, img_name))
                    self.labels.append(class_name)
        
        # Limit samples if specified
        if max_samples and len(self.image_paths) > max_samples:
            self.image_paths = self.image_paths[:max_samples]
            self.labels = self.labels[:max_samples]
        
        print(f"✓ Found {len(self.image_paths)} images in {len(class_folders)} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='green')
        
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'label': label,
            'path': img_path
        }


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms(img_size=IMG_SIZE):
    """Get image transforms for GAN training ([-1, 1] range)"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def create_preprocessed_dataset(train_dir=PLANTDOC_TRAIN, test_dir=PLANTDOC_TEST,
                                  output_dir=OUTPUT_DIR, max_samples=None):
    """
    Load PlantDoc images and save as preprocessed tensors
    """
    print("=" * 70)
    print("Creating Preprocessed PlantDoc Dataset")
    print("=" * 70 + "\n")
    
    os.makedirs(output_dir, exist_ok=True)
    
    transform = get_transforms(IMG_SIZE)
    
    # Process train data
    print("[1/4] Loading training data...")
    train_dataset = PlantDocDataset(train_dir, transform=transform, max_samples=max_samples)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    train_images = []
    train_labels = []
    
    for batch in tqdm(train_loader, desc="Processing train batches"):
        train_images.append(batch['image'])
        train_labels.extend(batch['label'])
    
    train_images = torch.cat(train_images, dim=0)
    
    # Process test data
    print("\n[2/4] Loading test data...")
    test_dataset = PlantDocDataset(test_dir, transform=transform, max_samples=max_samples)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    test_images = []
    test_labels = []
    
    for batch in tqdm(test_loader, desc="Processing test batches"):
        test_images.append(batch['image'])
        test_labels.extend(batch['label'])
    
    test_images = torch.cat(test_images, dim=0)
    
    # Save datasets
    print(f"\n[3/4] Saving preprocessed data...")
    
    torch.save({
        'images': train_images,
        'image_ids': train_labels,
        'img_size': IMG_SIZE,
        'num_images': len(train_images),
        'split': 'train',
        'dataset': 'PlantDoc'
    }, f'{output_dir}/train_data.pt')
    print(f"✓ Saved: {output_dir}/train_data.pt ({len(train_images)} images)")
    
    torch.save({
        'images': test_images,
        'image_ids': test_labels,
        'img_size': IMG_SIZE,
        'num_images': len(test_images),
        'split': 'test',
        'dataset': 'PlantDoc'
    }, f'{output_dir}/test_data.pt')
    print(f"✓ Saved: {output_dir}/test_data.pt ({len(test_images)} images)")
    
    # Save sample visualization
    print(f"\n[4/4] Saving sample images...")
    save_sample_images(train_images, test_images, output_dir)
    
    print("\n" + "=" * 70)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nDataset: PlantDoc Leaf Disease")
    print(f"  Train images: {len(train_images)}")
    print(f"  Test images: {len(test_images)}")
    print(f"  Classes: {len(train_dataset.classes)}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"\nReady for GAN training!")
    
    return train_images, test_images


def save_sample_images(train_images, test_images, output_dir):
    """Save viewable sample images"""
    train_samples = train_images[:16] * 0.5 + 0.5
    test_samples = test_images[:8] * 0.5 + 0.5
    
    # Train samples
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flatten()):
        if i < len(train_samples):
            img = train_samples[i].permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
    plt.suptitle('PlantDoc Training Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/train_samples.png")
    
    # Test samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        if i < len(test_samples):
            img = test_samples[i].permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
    plt.suptitle('PlantDoc Test Samples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/test_samples.png")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    create_preprocessed_dataset()
