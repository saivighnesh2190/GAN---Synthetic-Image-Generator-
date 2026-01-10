"""
Module 4: GAN Evaluation & Quality Assurance
==============================================
- Quantitative Metrics: Classifier Realism Score, Diversity Score, FID Proxy
- Qualitative Evaluation: Mode collapse detection, diversity assessment
- Visualizations: Loss curves, image grids, latent interpolation, t-SNE
"""

import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from PIL import Image

# For FID and feature extraction
import torchvision.models as models
import torchvision.transforms as transforms
from scipy import linalg
from sklearn.manifold import TSNE

# Project imports
from vanilla_gan import VanillaGAN
from data_loader import get_preprocessed_dataloader, load_config


# =============================================================================
# CONFIGURATION
# =============================================================================

FIGURES_DIR = 'figures'
os.makedirs(FIGURES_DIR, exist_ok=True)


# =============================================================================
# FEATURE EXTRACTOR (for FID and embeddings)
# =============================================================================

class FeatureExtractor:
    """Extract features using pretrained InceptionV3 or ResNet for FID calculation"""
    
    def __init__(self, model_name='resnet18', device='cpu'):
        self.device = device
        
        if model_name == 'inception':
            self.model = models.inception_v3(weights='IMAGENET1K_V1', transform_input=False)
            self.model.fc = nn.Identity()
            self.feature_dim = 2048
        else:  # resnet18 (lighter weight)
            self.model = models.resnet18(weights='IMAGENET1K_V1')
            self.model.fc = nn.Identity()
            self.feature_dim = 512
        
        self.model = self.model.to(device)
        self.model.eval()
        
        # Transform for pretrained models (ImageNet normalization)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, images):
        """Extract feature vectors from images"""
        # Convert from [-1, 1] to [0, 1]
        images = images * 0.5 + 0.5
        
        # Apply ImageNet normalization
        images = self.transform(images)
        
        with torch.no_grad():
            features = self.model(images.to(self.device))
        
        return features.cpu().numpy()


# =============================================================================
# QUANTITATIVE METRICS
# =============================================================================

class GANEvaluator:
    """Comprehensive GAN evaluation with multiple metrics"""
    
    def __init__(self, gan, device='cpu'):
        self.gan = gan
        self.device = device
        self.feature_extractor = FeatureExtractor(model_name='resnet18', device=device)
        
        # For classifier-based realism
        self.classifier = models.resnet18(weights='IMAGENET1K_V1').to(device)
        self.classifier.eval()
        
        print("✓ GANEvaluator initialized")
        print(f"  Feature extractor: ResNet18 (dim={self.feature_extractor.feature_dim})")
    
    # -------------------------------------------------------------------------
    # 1. Classifier-based Realism Score
    # -------------------------------------------------------------------------
    def classifier_realism_score(self, real_images, num_fake=100):
        """
        Compare classifier confidence on real vs generated images.
        Higher confidence on generated images = more realistic.
        """
        print("\n[1/4] Computing Classifier-based Realism Score...")
        
        # Generate fake images
        self.gan.G.eval()
        with torch.no_grad():
            z = torch.randn(num_fake, self.gan.latent_dim, device=self.device)
            fake_images = self.gan.G(z)
        self.gan.G.train()
        
        # Prepare transforms for classifier
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        def get_confidence(images):
            # Convert to [0, 1] then apply ImageNet normalization
            images = images * 0.5 + 0.5
            images = transform(images)
            
            with torch.no_grad():
                outputs = self.classifier(images.to(self.device))
                probs = torch.softmax(outputs, dim=1)
                max_probs = probs.max(dim=1)[0]
            
            return max_probs.cpu().numpy()
        
        # Get confidences
        real_conf = get_confidence(real_images[:num_fake])
        fake_conf = get_confidence(fake_images)
        
        results = {
            'real_mean_confidence': float(np.mean(real_conf)),
            'real_std_confidence': float(np.std(real_conf)),
            'fake_mean_confidence': float(np.mean(fake_conf)),
            'fake_std_confidence': float(np.std(fake_conf)),
            'realism_ratio': float(np.mean(fake_conf) / np.mean(real_conf))
        }
        
        print(f"  Real images confidence: {results['real_mean_confidence']:.4f} ± {results['real_std_confidence']:.4f}")
        print(f"  Fake images confidence: {results['fake_mean_confidence']:.4f} ± {results['fake_std_confidence']:.4f}")
        print(f"  Realism ratio: {results['realism_ratio']:.4f}")
        
        return results
    
    # -------------------------------------------------------------------------
    # 2. Diversity Score
    # -------------------------------------------------------------------------
    def diversity_score(self, num_samples=100):
        """
        Measure diversity of generated images using feature space variance.
        Higher variance = more diverse outputs.
        """
        print("\n[2/4] Computing Diversity Score...")
        
        # Generate fake images
        self.gan.G.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.gan.latent_dim, device=self.device)
            fake_images = self.gan.G(z)
        self.gan.G.train()
        
        # Extract features
        features = self.feature_extractor.extract_features(fake_images)
        
        # Compute pairwise distances
        mean_feature = np.mean(features, axis=0)
        distances = np.linalg.norm(features - mean_feature, axis=1)
        
        # Compute variance in feature space
        feature_variance = np.var(features, axis=0).mean()
        
        # Count unique clusters (approximate diversity)
        from sklearn.cluster import KMeans
        n_clusters = min(10, num_samples // 5)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        unique_clusters = len(np.unique(labels))
        
        results = {
            'mean_distance_from_center': float(np.mean(distances)),
            'feature_variance': float(feature_variance),
            'unique_clusters': unique_clusters,
            'cluster_diversity': unique_clusters / n_clusters
        }
        
        print(f"  Mean distance from center: {results['mean_distance_from_center']:.4f}")
        print(f"  Feature variance: {results['feature_variance']:.4f}")
        print(f"  Unique clusters: {results['unique_clusters']}/{n_clusters}")
        
        return results
    
    # -------------------------------------------------------------------------
    # 3. FID Proxy (Fréchet Inception Distance)
    # -------------------------------------------------------------------------
    def fid_score(self, real_images, num_fake=100):
        """
        Compute FID-like score between real and generated images.
        Lower FID = better quality and diversity.
        """
        print("\n[3/4] Computing FID Proxy Score...")
        
        # Generate fake images
        self.gan.G.eval()
        with torch.no_grad():
            z = torch.randn(num_fake, self.gan.latent_dim, device=self.device)
            fake_images = self.gan.G(z)
        self.gan.G.train()
        
        # Extract features
        real_features = self.feature_extractor.extract_features(real_images[:num_fake])
        fake_features = self.feature_extractor.extract_features(fake_images)
        
        # Compute statistics
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        # Compute FID
        diff = mu_real - mu_fake
        
        # Numerical stability
        eps = 1e-6
        sigma_real += np.eye(sigma_real.shape[0]) * eps
        sigma_fake += np.eye(sigma_fake.shape[0]) * eps
        
        # Matrix square root
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        results = {
            'fid_score': float(fid),
            'mean_distance': float(np.linalg.norm(diff)),
        }
        
        print(f"  FID Score: {results['fid_score']:.4f}")
        print(f"  Mean distance: {results['mean_distance']:.4f}")
        
        return results
    
    # -------------------------------------------------------------------------
    # 4. Mode Collapse Detection
    # -------------------------------------------------------------------------
    def detect_mode_collapse(self, num_samples=50, threshold=0.1):
        """
        Detect mode collapse by checking if generated images are too similar.
        """
        print("\n[4/4] Checking for Mode Collapse...")
        
        # Generate images
        self.gan.G.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.gan.latent_dim, device=self.device)
            fake_images = self.gan.G(z)
        self.gan.G.train()
        
        # Compute pairwise pixel differences
        fake_flat = fake_images.view(num_samples, -1).cpu().numpy()
        
        pairwise_diffs = []
        for i in range(num_samples):
            for j in range(i+1, num_samples):
                diff = np.mean(np.abs(fake_flat[i] - fake_flat[j]))
                pairwise_diffs.append(diff)
        
        mean_diff = np.mean(pairwise_diffs)
        min_diff = np.min(pairwise_diffs)
        
        # Mode collapse if images are too similar
        mode_collapse_detected = mean_diff < threshold
        
        results = {
            'mean_pairwise_difference': float(mean_diff),
            'min_pairwise_difference': float(min_diff),
            'mode_collapse_detected': mode_collapse_detected,
            'threshold': threshold
        }
        
        status = "⚠️ MODE COLLAPSE DETECTED" if mode_collapse_detected else "✓ No mode collapse"
        print(f"  Mean pairwise difference: {results['mean_pairwise_difference']:.4f}")
        print(f"  Min pairwise difference: {results['min_pairwise_difference']:.4f}")
        print(f"  Status: {status}")
        
        return results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

class GANVisualizer:
    """Visualization tools for GAN evaluation"""
    
    def __init__(self, gan, device='cpu'):
        self.gan = gan
        self.device = device
        self.feature_extractor = FeatureExtractor(model_name='resnet18', device=device)
    
    # -------------------------------------------------------------------------
    # 1. Loss Curves
    # -------------------------------------------------------------------------
    def plot_loss_curves(self, log_file='logs/training_history.csv', save_path=None):
        """Plot training loss curves from CSV log"""
        import csv
        
        epochs, g_losses, d_losses, d_real, d_fake = [], [], [], [], []
        
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                epochs.append(int(row['epoch']))
                g_losses.append(float(row['g_loss']))
                d_losses.append(float(row['d_loss']))
                d_real.append(float(row['d_real']))
                d_fake.append(float(row['d_fake']))
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(epochs, g_losses, label='Generator', color='blue')
        axes[0].plot(epochs, d_losses, label='Discriminator', color='orange')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scores plot
        axes[1].plot(epochs, d_real, label='D(real)', color='green')
        axes[1].plot(epochs, d_fake, label='D(fake)', color='red')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Discriminator Scores')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(FIGURES_DIR, 'loss_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Loss curves saved: {save_path}")
        return save_path
    
    # -------------------------------------------------------------------------
    # 2. Generated Image Grid
    # -------------------------------------------------------------------------
    def plot_image_grid(self, num_images=64, save_path=None):
        """Generate and plot a grid of generated images"""
        self.gan.G.eval()
        with torch.no_grad():
            z = torch.randn(num_images, self.gan.latent_dim, device=self.device)
            fake_images = self.gan.G(z)
        self.gan.G.train()
        
        # Denormalize
        fake_images = fake_images * 0.5 + 0.5
        fake_images = fake_images.cpu().numpy()
        
        nrow = int(np.sqrt(num_images))
        ncol = nrow
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
        
        for i, ax in enumerate(axes.flatten()):
            if i < len(fake_images):
                img = np.transpose(fake_images[i], (1, 2, 0))
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            ax.axis('off')
        
        plt.suptitle('Generated Images', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(FIGURES_DIR, 'generated_grid.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Image grid saved: {save_path}")
        return save_path
    
    # -------------------------------------------------------------------------
    # 3. Latent Space Interpolation
    # -------------------------------------------------------------------------
    def plot_latent_interpolation(self, num_steps=10, num_pairs=4, save_path=None):
        """
        Interpolate between pairs of latent vectors to show smooth morphing.
        """
        self.gan.G.eval()
        
        fig, axes = plt.subplots(num_pairs, num_steps, figsize=(num_steps * 1.5, num_pairs * 1.5))
        
        for pair_idx in range(num_pairs):
            # Random start and end points
            z1 = torch.randn(1, self.gan.latent_dim, device=self.device)
            z2 = torch.randn(1, self.gan.latent_dim, device=self.device)
            
            for step in range(num_steps):
                # Linear interpolation
                alpha = step / (num_steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                with torch.no_grad():
                    img = self.gan.G(z_interp)
                
                # Denormalize and display
                img = img * 0.5 + 0.5
                img = img[0].cpu().numpy().transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                axes[pair_idx, step].imshow(img)
                axes[pair_idx, step].axis('off')
        
        self.gan.G.train()
        
        plt.suptitle('Latent Space Interpolation (Smooth Morphing)', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(FIGURES_DIR, 'latent_interpolation.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Latent interpolation saved: {save_path}")
        return save_path
    
    # -------------------------------------------------------------------------
    # 4. t-SNE Visualization
    # -------------------------------------------------------------------------
    def plot_tsne(self, real_images, num_fake=100, save_path=None):
        """
        t-SNE visualization of real vs fake image embeddings.
        """
        print("Computing t-SNE embeddings (this may take a moment)...")
        
        # Generate fake images
        self.gan.G.eval()
        with torch.no_grad():
            z = torch.randn(num_fake, self.gan.latent_dim, device=self.device)
            fake_images = self.gan.G(z)
        self.gan.G.train()
        
        # Extract features
        real_features = self.feature_extractor.extract_features(real_images[:num_fake])
        fake_features = self.feature_extractor.extract_features(fake_images)
        
        # Combine features
        all_features = np.vstack([real_features, fake_features])
        labels = np.array([0] * num_fake + [1] * num_fake)  # 0=real, 1=fake
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings = tsne.fit_transform(all_features)
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        real_emb = embeddings[labels == 0]
        fake_emb = embeddings[labels == 1]
        
        ax.scatter(real_emb[:, 0], real_emb[:, 1], c='blue', label='Real', alpha=0.6, s=50)
        ax.scatter(fake_emb[:, 0], fake_emb[:, 1], c='red', label='Generated', alpha=0.6, s=50)
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title('t-SNE: Real vs Generated Image Embeddings')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path is None:
            save_path = os.path.join(FIGURES_DIR, 'tsne_visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ t-SNE visualization saved: {save_path}")
        return save_path
    
    # -------------------------------------------------------------------------
    # 5. Real vs Fake Comparison
    # -------------------------------------------------------------------------
    def plot_real_vs_fake(self, real_images, num_samples=8, save_path=None):
        """Side-by-side comparison of real vs generated images"""
        self.gan.G.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.gan.latent_dim, device=self.device)
            fake_images = self.gan.G(z)
        self.gan.G.train()
        
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            # Real image
            real_img = real_images[i] * 0.5 + 0.5
            real_img = real_img.cpu().numpy().transpose(1, 2, 0)
            real_img = np.clip(real_img, 0, 1)
            axes[0, i].imshow(real_img)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Real', fontsize=12)
            
            # Fake image
            fake_img = fake_images[i] * 0.5 + 0.5
            fake_img = fake_img.cpu().numpy().transpose(1, 2, 0)
            fake_img = np.clip(fake_img, 0, 1)
            axes[1, i].imshow(fake_img)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Generated', fontsize=12)
        
        plt.suptitle('Real vs Generated Images', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(FIGURES_DIR, 'real_vs_fake.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Real vs fake comparison saved: {save_path}")
        return save_path


# =============================================================================
# EVALUATION REPORT GENERATOR
# =============================================================================

def generate_evaluation_report(metrics, figures, save_path='figures/evaluation_report.txt'):
    """Generate a text-based evaluation report"""
    
    report = []
    report.append("=" * 70)
    report.append("GAN EVALUATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    report.append("\n1. CLASSIFIER-BASED REALISM SCORE")
    report.append("-" * 40)
    report.append(f"   Real images confidence:    {metrics['realism']['real_mean_confidence']:.4f} ± {metrics['realism']['real_std_confidence']:.4f}")
    report.append(f"   Fake images confidence:    {metrics['realism']['fake_mean_confidence']:.4f} ± {metrics['realism']['fake_std_confidence']:.4f}")
    report.append(f"   Realism ratio:             {metrics['realism']['realism_ratio']:.4f}")
    
    report.append("\n2. DIVERSITY SCORE")
    report.append("-" * 40)
    report.append(f"   Mean distance from center: {metrics['diversity']['mean_distance_from_center']:.4f}")
    report.append(f"   Feature variance:          {metrics['diversity']['feature_variance']:.4f}")
    report.append(f"   Unique clusters:           {metrics['diversity']['unique_clusters']}")
    report.append(f"   Cluster diversity:         {metrics['diversity']['cluster_diversity']:.2%}")
    
    report.append("\n3. FID SCORE (Fréchet Inception Distance Proxy)")
    report.append("-" * 40)
    report.append(f"   FID Score:                 {metrics['fid']['fid_score']:.4f}")
    report.append(f"   Mean feature distance:     {metrics['fid']['mean_distance']:.4f}")
    
    report.append("\n4. MODE COLLAPSE DETECTION")
    report.append("-" * 40)
    report.append(f"   Mean pairwise difference:  {metrics['mode_collapse']['mean_pairwise_difference']:.4f}")
    report.append(f"   Min pairwise difference:   {metrics['mode_collapse']['min_pairwise_difference']:.4f}")
    status = "⚠️ DETECTED" if metrics['mode_collapse']['mode_collapse_detected'] else "✓ Not detected"
    report.append(f"   Mode collapse status:      {status}")
    
    report.append("\n5. GENERATED FIGURES")
    report.append("-" * 40)
    for name, path in figures.items():
        report.append(f"   {name}: {path}")
    
    report.append("\n" + "=" * 70)
    report.append("END OF REPORT")
    report.append("=" * 70)
    
    report_text = "\n".join(report)
    
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n✓ Evaluation report saved: {save_path}")
    return report_text


# =============================================================================
# MAIN EVALUATION PIPELINE
# =============================================================================

def run_full_evaluation(checkpoint_path=None, num_samples=64):
    """Run complete evaluation pipeline"""
    
    print("=" * 70)
    print("Module 4: GAN Evaluation & Quality Assurance")
    print("=" * 70 + "\n")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Initialize GAN
    gan = VanillaGAN(latent_dim=100, device=device)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint: {checkpoint_path}")
        gan.load(checkpoint_path)
    else:
        print("\n⚠️ No checkpoint provided, using untrained model")
    
    # Load real images for comparison
    print("\nLoading real images...")
    train_loader, _ = get_preprocessed_dataloader(
        preprocessed_file='preprocessed_data/train_data.pt',
        batch_size=num_samples,
        shuffle=True
    )
    
    real_batch = next(iter(train_loader))
    real_images = real_batch['image'].to(device)
    
    # Initialize evaluator and visualizer
    evaluator = GANEvaluator(gan, device=device)
    visualizer = GANVisualizer(gan, device=device)
    
    # =========================================================================
    # QUANTITATIVE METRICS
    # =========================================================================
    print("\n" + "=" * 70)
    print("QUANTITATIVE METRICS")
    print("=" * 70)
    
    metrics = {}
    metrics['realism'] = evaluator.classifier_realism_score(real_images, num_fake=num_samples)
    metrics['diversity'] = evaluator.diversity_score(num_samples=num_samples)
    metrics['fid'] = evaluator.fid_score(real_images, num_fake=num_samples)
    metrics['mode_collapse'] = evaluator.detect_mode_collapse(num_samples=num_samples)
    
    # =========================================================================
    # VISUALIZATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("VISUALIZATIONS")
    print("=" * 70 + "\n")
    
    figures = {}
    
    # Loss curves (if log file exists)
    if os.path.exists('logs/training_history.csv'):
        figures['loss_curves'] = visualizer.plot_loss_curves()
    
    # Generated image grid
    figures['generated_grid'] = visualizer.plot_image_grid(num_images=64)
    
    # Latent interpolation
    figures['latent_interpolation'] = visualizer.plot_latent_interpolation(num_steps=10, num_pairs=4)
    
    # t-SNE visualization
    figures['tsne'] = visualizer.plot_tsne(real_images, num_fake=num_samples)
    
    # Real vs fake comparison
    figures['real_vs_fake'] = visualizer.plot_real_vs_fake(real_images, num_samples=8)
    
    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    
    report = generate_evaluation_report(metrics, figures)
    print("\n" + report)
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"\n✓ All figures saved to: {FIGURES_DIR}/")
    print(f"✓ Evaluation report: {FIGURES_DIR}/evaluation_report.txt")
    
    return metrics, figures


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='GAN Evaluation Pipeline')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/gan_final.pt',
                        help='Path to GAN checkpoint file')
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    run_full_evaluation(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples
    )
