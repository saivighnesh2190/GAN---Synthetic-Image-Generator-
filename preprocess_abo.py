import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, dataset
from torchvision import transforms
import matplotlib

matplotlib.use('Agg')  # For non-GUI environments
import matplotlib.pyplot as plt


class ABODataset(Dataset):
    """Custom PyTorch Dataset for ABO images"""

    def __init__(self, csv_file, img_dir, transform=None, max_samples=None):
        print(f"Loading metadata from {csv_file}...")

        # Try different ways to load the CSV
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
            # The path in CSV is like "12/12b40293.jpg"
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
            image = Image.new('RGB', (64, 64), color='white')

        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'image_id': img_info.get('image_id', str(idx)),
        }


def get_transforms(img_size=64):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


def visualize_batch(dataloader, num_images=8):
    """Visualize sample images"""
    batch = next(iter(dataloader))
    images = batch['image']
    images = images * 0.5 + 0.5  # Denormalize

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(min(num_images, len(images))):
        img = images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('sample_images.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Sample images saved to 'sample_images.png'")
    plt.close()


def main():
    print("=" * 60)
    print("ABO Dataset Preprocessing for GAN Training")
    print("=" * 60 + "\n")

    # CORRECTED PATHS based on your structure
    # First, let's auto-detect the CSV location
    possible_csv = [
        'abo_dataset/metadata/images.csv.gz',
        'abo_dataset/images.csv.gz',
        'abo_dataset/images.csv',
        'abo_dataset/metadata/images.csv'
    ]

    CSV_FILE = None
    for path in possible_csv:
        if os.path.exists(path):
            CSV_FILE = path
            print(f"✓ Found CSV: {path}")
            break

    if CSV_FILE is None:
        print("ERROR: Could not find images.csv file!")
        print("Please check manually:")
        os.system("find abo_dataset -name '*.csv*'")
        return

    IMG_DIR = 'abo_dataset/small'  # Your structure
    IMG_SIZE = 64
    BATCH_SIZE = 16
    MAX_SAMPLES = 5000

    print(f"\nConfiguration:")
    print(f"  CSV: {CSV_FILE}")
    print(f"  Images: {IMG_DIR}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max samples: {MAX_SAMPLES}\n")

    # Verify image directory
    if not os.path.exists(IMG_DIR):
        print(f"ERROR: Image directory not found: {IMG_DIR}")
        return

    sample_count = len([f for f in os.listdir(IMG_DIR) if os.path.isdir(os.path.join(IMG_DIR, f))])
    print(f"Found {sample_count} subdirectories in {IMG_DIR}\n")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Running on: {'GPU' if torch.cuda.is_available() else 'CPU'}\n")

    # Create dataset
    transform = get_transforms(img_size=IMG_SIZE)
    from load_preprocessed import get_preprocessed_dataloader
    dataloader = get_preprocessed_dataloader(batch_size=16, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    print("=" * 60)
    print("Testing DataLoader")
    print("=" * 60 + "\n")

    for i, batch in enumerate(dataloader):
        images = batch['image']
        print(f"Batch {i + 1}:")
        print(f"  Shape: {images.shape}")
        print(f"  Value range: [{images.min():.3f}, {images.max():.3f}]")

        if i == 2:
            break

    print("\n" + "=" * 60)
    print("Generating sample visualization...")
    print("=" * 60)
    visualize_batch(dataloader)

    print("\n" + "=" * 60)
    print("✓ PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"\nDataset Statistics:")
    print(f"  Total images: {len(dataset)}")
    print(f"  Total batches: {len(dataloader)}")
    print(f"  Image shape: [3, {IMG_SIZE}, {IMG_SIZE}]")
    print(f"\nYour dataloader is ready for GAN training!\n")


if __name__ == "__main__":
    main()
