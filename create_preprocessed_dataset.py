import torch
import numpy as np
from torch.utils.data import DataLoader
from preprocess_abo import ABODataset, get_transforms
from tqdm import tqdm
import os

def create_preprocessed_dataset():
    """
    Load all images, apply transformations, and save as preprocessed tensors
    """
    print("="*70)
    print("Creating Preprocessed Dataset")
    print("="*70 + "\n")
    
    # Configuration
    CSV_FILE = 'abo_dataset/metadata/images.csv'
    IMG_DIR = 'abo_dataset/small'
    IMG_SIZE = 64
    MAX_SAMPLES = 5000  # Adjust as needed
    OUTPUT_DIR = 'preprocessed_data'
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Configuration:")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Max samples: {MAX_SAMPLES}")
    print(f"  Output: {OUTPUT_DIR}/\n")
    
    # Load dataset
    print("[1/3] Loading original dataset...")
    transform = get_transforms(img_size=IMG_SIZE)
    dataset = ABODataset(
        csv_file=CSV_FILE,
        img_dir=IMG_DIR,
        transform=transform,
        max_samples=MAX_SAMPLES
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,  # Larger batch for preprocessing
        shuffle=False,
        num_workers=0
    )
    
    # Process and save
    print(f"\n[2/3] Processing {len(dataset)} images...")
    all_images = []
    all_ids = []
    
    for batch in tqdm(dataloader, desc="Preprocessing batches"):
        all_images.append(batch['image'])
        all_ids.extend(batch['image_id'])
    
    # Concatenate all batches
    all_images = torch.cat(all_images, dim=0)
    
    print(f"\n[3/3] Saving preprocessed data...")
    print(f"  Total images: {all_images.shape[0]}")
    print(f"  Image shape: {all_images.shape[1:]}")
    print(f"  Data type: {all_images.dtype}")
    print(f"  Size in memory: {all_images.element_size() * all_images.nelement() / (1024**2):.2f} MB")
    
    # Save as PyTorch tensor (recommended for PyTorch projects)
    torch.save({
        'images': all_images,
        'image_ids': all_ids,
        'img_size': IMG_SIZE,
        'num_images': len(all_images)
    }, f'{OUTPUT_DIR}/preprocessed_images.pt')
    
    print(f"\n✓ Saved: {OUTPUT_DIR}/preprocessed_images.pt")
    
    # Also save as compressed numpy (optional, for compatibility)
    np.savez_compressed(
        f'{OUTPUT_DIR}/preprocessed_images.npz',
        images=all_images.numpy(),
        image_ids=np.array(all_ids)
    )
    
    print(f"✓ Saved: {OUTPUT_DIR}/preprocessed_images.npz (compressed)")
    
    # Save metadata
    with open(f'{OUTPUT_DIR}/dataset_info.txt', 'w') as f:
        f.write(f"Preprocessed ABO Dataset\n")
        f.write(f"========================\n")
        f.write(f"Total images: {len(all_images)}\n")
        f.write(f"Image size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Channels: 3 (RGB)\n")
        f.write(f"Data range: [-1, 1] (normalized)\n")
        f.write(f"Original dataset: ABO small\n")
    
    print(f"✓ Saved: {OUTPUT_DIR}/dataset_info.txt")
    
    print("\n" + "="*70)
    print("✓ PREPROCESSING COMPLETE!")
    print("="*70)
    print(f"\nPreprocessed files saved in: {OUTPUT_DIR}/")
    print("You can now use these files for faster training.\n")

if __name__ == "__main__":
    create_preprocessed_dataset()
