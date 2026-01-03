import os
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION - Load from config.yaml
# =============================================================================

def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        # Default configuration if YAML not found
        return {
            'preprocessing': {'image_size': 64, 'channels': 3, 'normalization': 'tanh'},
            'split': {'train_split': 0.95},
            'dataloader': {'batch_size': 16, 'max_samples': 5000},
            'data': {'csv_file': 'abo_dataset/metadata/images.csv', 'img_dir': 'abo_dataset/small', 'output_dir': 'preprocessed_data'}
        }

# Load config
CONFIG = load_config()

# Extract configuration values
IMG_SIZE = CONFIG['preprocessing']['image_size']
CHANNELS = CONFIG['preprocessing']['channels']
NORMALIZATION = CONFIG['preprocessing']['normalization']
TRAIN_SPLIT = CONFIG['split']['train_split']
BATCH_SIZE = CONFIG['dataloader']['batch_size']
MAX_SAMPLES = CONFIG['dataloader']['max_samples']
CSV_FILE = CONFIG['data']['csv_file']
IMG_DIR = CONFIG['data']['img_dir']
OUTPUT_DIR = CONFIG['data']['output_dir']


# =============================================================================
# DATASET CLASSES
# =============================================================================

class ABODataset(Dataset):
    """Custom PyTorch Dataset for ABO images"""

    def __init__(self, csv_file, img_dir, transform=None, max_samples=None):
        print(f"Loading metadata from {csv_file}...")

        if csv_file.endswith('.gz'):
            self.img_metadata = pd.read_csv(csv_file, compression='gzip')
        else:
            self.img_metadata = pd.read_csv(csv_file)

        print(f"Total images in CSV: {len(self.img_metadata)}")

        if max_samples:
            self.img_metadata = self.img_metadata.head(max_samples)
            print(f"Limited to {max_samples} samples for CPU training")

        self.img_dir = img_dir
        self.transform = transform

        self.img_metadata = self._filter_existing_images()
        print(f"✓ Dataset loaded: {len(self.img_metadata)} valid images\n")

    def _filter_existing_images(self):
        """Keep only images that exist"""
        print("Filtering existing images...")
        existing = []

        for idx, row in self.img_metadata.iterrows():
            img_path = os.path.join(self.img_dir, row['path'])

            if os.path.exists(img_path):
                existing.append(row)

            if (idx + 1) % 5000 == 0:
                print(f"  Checked {idx + 1} images...")

        return pd.DataFrame(existing)

    def __len__(self):
        return len(self.img_metadata)

    def __getitem__(self, idx):
        img_info = self.img_metadata.iloc[idx]
        img_path = os.path.join(self.img_dir, img_info['path'])

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='white')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'image_id': img_info.get('image_id', str(idx)),
        }


class PreprocessedDataset(Dataset):
    """Dataset that loads from preprocessed tensors"""

    def __init__(self, preprocessed_file):
        print(f"Loading preprocessed data from {preprocessed_file}...")

        data = torch.load(preprocessed_file, weights_only=False)

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


# =============================================================================
# TRANSFORMS
# =============================================================================

def get_transforms(img_size=IMG_SIZE):
    """
    Get image transforms for GAN training
    Normalization: [-1, 1] for TANH output
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


# =============================================================================
# DATALOADER FUNCTIONS
# =============================================================================

def get_raw_dataloader(csv_file, img_dir, batch_size=BATCH_SIZE, 
                       max_samples=MAX_SAMPLES, shuffle=True):
    """
    Create DataLoader from raw images
    Use this for initial preprocessing
    """
    transform = get_transforms(img_size=IMG_SIZE)
    
    dataset = ABODataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=transform,
        max_samples=max_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )

    return dataloader, dataset


def get_preprocessed_dataloader(preprocessed_file='preprocessed_data/preprocessed_images.pt',
                                 batch_size=BATCH_SIZE, shuffle=True):
    """
    Create DataLoader from preprocessed data
    Much faster than loading from raw images!
    """
    dataset = PreprocessedDataset(preprocessed_file)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False
    )

    return dataloader, dataset


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================

def create_preprocessed_dataset(csv_file='abo_dataset/metadata/images.csv',
                                 img_dir='abo_dataset/small',
                                 output_dir='preprocessed_data',
                                 max_samples=MAX_SAMPLES,
                                 train_split=TRAIN_SPLIT):
    """
    Load all images, apply transformations, and save as preprocessed tensors
    Creates separate train and test files
    """
    print("=" * 70)
    print("Creating Preprocessed Dataset (Train/Test Split)")
    print("=" * 70 + "\n")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Configuration:")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Channels: {CHANNELS}")
    print(f"  Max samples: {max_samples}")
    print(f"  Train/Test split: {train_split}/{1-train_split}")
    print(f"  Normalization: {NORMALIZATION} [-1, 1]")
    print(f"  Output: {output_dir}/\n")

    print("[1/4] Loading original dataset...")
    transform = get_transforms(img_size=IMG_SIZE)
    dataset = ABODataset(
        csv_file=csv_file,
        img_dir=img_dir,
        transform=transform,
        max_samples=max_samples
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    print(f"\n[2/4] Processing {len(dataset)} images...")
    all_images = []
    all_ids = []

    for batch in tqdm(dataloader, desc="Preprocessing batches"):
        all_images.append(batch['image'])
        all_ids.extend(batch['image_id'])

    all_images = torch.cat(all_images, dim=0)
    all_ids = list(all_ids)

    print(f"\n[3/4] Splitting into train/test...")
    total_size = len(all_images)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size

    # Shuffle indices for random split
    indices = torch.randperm(total_size)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_images = all_images[train_indices]
    test_images = all_images[test_indices]
    train_ids = [all_ids[i] for i in train_indices.tolist()]
    test_ids = [all_ids[i] for i in test_indices.tolist()]

    print(f"  Train set: {len(train_images)} images ({train_split*100:.0f}%)")
    print(f"  Test set: {len(test_images)} images ({(1-train_split)*100:.0f}%)")

    print(f"\n[4/4] Saving preprocessed data...")
    print(f"  Total images: {all_images.shape[0]}")
    print(f"  Image shape: {all_images.shape[1:]}")
    print(f"  Data type: {all_images.dtype}")

    # Save combined dataset
    torch.save({
        'images': all_images,
        'image_ids': all_ids,
        'img_size': IMG_SIZE,
        'num_images': len(all_images)
    }, f'{output_dir}/preprocessed_images.pt')
    print(f"\n✓ Saved: {output_dir}/preprocessed_images.pt (all data)")

    # Save TRAIN dataset
    torch.save({
        'images': train_images,
        'image_ids': train_ids,
        'img_size': IMG_SIZE,
        'num_images': len(train_images),
        'split': 'train'
    }, f'{output_dir}/train_data.pt')
    print(f"✓ Saved: {output_dir}/train_data.pt ({len(train_images)} images)")

    # Save TEST dataset
    torch.save({
        'images': test_images,
        'image_ids': test_ids,
        'img_size': IMG_SIZE,
        'num_images': len(test_images),
        'split': 'test'
    }, f'{output_dir}/test_data.pt')
    print(f"✓ Saved: {output_dir}/test_data.pt ({len(test_images)} images)")

    # Save as compressed numpy
    np.savez_compressed(
        f'{output_dir}/train_data.npz',
        images=train_images.numpy(),
        image_ids=np.array(train_ids)
    )
    np.savez_compressed(
        f'{output_dir}/test_data.npz',
        images=test_images.numpy(),
        image_ids=np.array(test_ids)
    )
    print(f"✓ Saved: {output_dir}/train_data.npz & test_data.npz (compressed)")

    # Save metadata
    with open(f'{output_dir}/dataset_info.txt', 'w') as f:
        f.write(f"Preprocessed ABO Dataset\n")
        f.write(f"========================\n")
        f.write(f"Total images: {len(all_images)}\n")
        f.write(f"Train images: {len(train_images)}\n")
        f.write(f"Test images: {len(test_images)}\n")
        f.write(f"Train/Test split: {train_split}/{1-train_split}\n")
        f.write(f"Image size: {IMG_SIZE}x{IMG_SIZE}\n")
        f.write(f"Channels: {CHANNELS} (RGB)\n")
        f.write(f"Data range: [-1, 1] (normalized with tanh)\n")
        f.write(f"Original dataset: ABO small\n")

    print(f"✓ Saved: {output_dir}/dataset_info.txt")

    print("\n" + "=" * 70)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"\nPreprocessed files saved in: {output_dir}/")
    print(f"  - train_data.pt ({len(train_images)} images)")
    print(f"  - test_data.pt ({len(test_images)} images)")
    print("You can now use these files for GAN training.\n")

    return train_images, test_images, train_ids, test_ids


# =============================================================================
# VISUALIZATION
# =============================================================================

def visualize_batch(dataloader, num_images=8, save_path='sample_images.png'):
    """Visualize sample images from dataloader"""
    batch = next(iter(dataloader))
    images = batch['image']
    images = images * 0.5 + 0.5  # Denormalize from [-1,1] to [0,1]

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(min(num_images, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Sample images saved to '{save_path}'")
    plt.close()


def save_sample_images(output_dir='preprocessed_data'):
    """
    Save viewable sample images from train and test datasets
    Creates PNG files showing samples from each split
    """
    train_file = f'{output_dir}/train_data.pt'
    test_file = f'{output_dir}/test_data.pt'
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("ERROR: Preprocessed train/test data not found!")
        return
    
    print("=" * 70)
    print("Saving Sample Images from Train/Test Data")
    print("=" * 70 + "\n")
    
    # Load train data
    train_data = torch.load(train_file, weights_only=False)
    train_images = train_data['images']
    
    # Load test data
    test_data = torch.load(test_file, weights_only=False)
    test_images = test_data['images']
    
    # Denormalize images from [-1,1] to [0,1]
    train_samples = train_images[:16] * 0.5 + 0.5
    test_samples = test_images[:8] * 0.5 + 0.5
    
    # Save train samples
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()
    for i in range(16):
        img = train_samples[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Train #{i+1}', fontsize=10)
    plt.suptitle('Training Data Samples (16 images)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/train_samples.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/train_samples.png (16 train images)")
    plt.close()
    
    # Save test samples
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i in range(8):
        img = test_samples[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f'Test #{i+1}', fontsize=10)
    plt.suptitle('Test Data Samples (8 images)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_samples.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/test_samples.png (8 test images)")
    plt.close()
    
    print(f"\n✓ Sample images saved to {output_dir}/")
    print("  - train_samples.png")
    print("  - test_samples.png")


# =============================================================================
# TRAIN/TEST SPLIT
# =============================================================================

def get_train_test_split(dataset, train_ratio=TRAIN_SPLIT):
    """
    Split dataset into train and test sets
    Note: GAN does not need labels, split is only for evaluation
    """
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    test_size = total_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    print(f"Dataset split:")
    print(f"  Train: {train_size} images ({train_ratio*100:.0f}%)")
    print(f"  Test: {test_size} images ({(1-train_ratio)*100:.0f}%)")

    return train_dataset, test_dataset


# =============================================================================
# MAIN - TEST THE DATA LOADER
# =============================================================================

def main():
    print("=" * 70)
    print("ABO Dataset Data Loader for GAN Training")
    print("=" * 70 + "\n")

    print(f"Configuration:")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Channels: {CHANNELS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Train split: {TRAIN_SPLIT}")
    print(f"  Normalization: {NORMALIZATION} [-1, 1]\n")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

    # Check if preprocessed train/test data exists
    train_file = 'preprocessed_data/train_data.pt'
    test_file = 'preprocessed_data/test_data.pt'

    if os.path.exists(train_file) and os.path.exists(test_file):
        print("✓ Preprocessed train/test data found!\n")
        print("Loading train data...")
        train_loader, train_dataset = get_preprocessed_dataloader(
            preprocessed_file=train_file,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        print("Loading test data...")
        test_loader, test_dataset = get_preprocessed_dataloader(
            preprocessed_file=test_file,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    else:
        print("Preprocessed train/test data NOT found.")
        print("Creating preprocessed dataset...\n")

        # Find CSV file
        possible_csv = [
            'abo_dataset/metadata/images.csv.gz',
            'abo_dataset/images.csv.gz',
            'abo_dataset/images.csv',
            'abo_dataset/metadata/images.csv'
        ]

        csv_file = None
        for path in possible_csv:
            if os.path.exists(path):
                csv_file = path
                print(f"✓ Found CSV: {path}")
                break

        if csv_file is None:
            print("ERROR: Could not find images.csv file!")
            print("Please ensure abo_dataset folder exists with metadata.")
            return

        # Create preprocessed train/test data
        create_preprocessed_dataset(
            csv_file=csv_file,
            img_dir='abo_dataset/small',
            output_dir='preprocessed_data',
            max_samples=MAX_SAMPLES,
            train_split=TRAIN_SPLIT
        )

        # Now load the created data
        print("\nLoading newly created train data...")
        train_loader, train_dataset = get_preprocessed_dataloader(
            preprocessed_file=train_file,
            batch_size=BATCH_SIZE,
            shuffle=True
        )
        print("Loading newly created test data...")
        test_loader, test_dataset = get_preprocessed_dataloader(
            preprocessed_file=test_file,
            batch_size=BATCH_SIZE,
            shuffle=False
        )

    print("=" * 70)
    print("Testing Train DataLoader")
    print("=" * 70 + "\n")

    for i, batch in enumerate(train_loader):
        images = batch['image']
        print(f"Train Batch {i + 1}:")
        print(f"  Shape: {images.shape}")
        print(f"  Value range: [{images.min():.3f}, {images.max():.3f}]")

        if i == 2:
            break

    print("\n" + "=" * 70)
    print("Testing Test DataLoader")
    print("=" * 70 + "\n")

    for i, batch in enumerate(test_loader):
        images = batch['image']
        print(f"Test Batch {i + 1}:")
        print(f"  Shape: {images.shape}")
        print(f"  Value range: [{images.min():.3f}, {images.max():.3f}]")

        if i == 1:
            break

    print("\n" + "=" * 70)
    print("Generating sample visualization...")
    print("=" * 70)
    visualize_batch(train_loader)

    print("\n" + "=" * 70)
    print("✓ DATA LOADER READY!")
    print("=" * 70)
    print(f"\nDataset Statistics:")
    print(f"  Train images: {len(train_dataset)}")
    print(f"  Test images: {len(test_dataset)}")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Image shape: [{CHANNELS}, {IMG_SIZE}, {IMG_SIZE}]")
    print(f"  Data range: [-1, 1] (normalized)")
    print(f"\nYour dataloaders are ready for GAN training!\n")


if __name__ == "__main__":
    main()
