"""
Module 3: GAN Training Pipeline
================================
- Training loop with configurable epochs and batch size
- TensorBoard logging (G_loss, D_loss, real/fake scores)
- CSV logging for training history
- Checkpoint saving after each epoch
- Sample image generation every N epochs
- Anti-failure mechanisms (label smoothing, gradient clipping)
"""

import os
import csv
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Project imports
from vanilla_gan import VanillaGAN
from data_loader import get_preprocessed_dataloader, load_config


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_training_config(config_path='config.yaml'):
    """Load training configuration from YAML file"""
    config = load_config(config_path)
    
    # Add training defaults if not present
    training_defaults = {
        'training': {
            'num_epochs': 100,
            'batch_size': 64,
            'latent_dim': 100,
            'lr': 0.0002,
            'beta1': 0.5,
            'save_interval': 10,  # Save samples every N epochs
            'checkpoint_interval': 10,  # Save checkpoint every N epochs
            'label_smoothing': True,
            'smooth_real_label': 0.9,
            'smooth_fake_label': 0.0,
            'gradient_clip': True,
            'gradient_clip_value': 1.0,
        }
    }
    
    if 'training' not in config:
        config.update(training_defaults)
    
    return config


# =============================================================================
# TRAINING CLASS
# =============================================================================

class GANTrainer:
    """
    GAN Training Pipeline with logging and checkpointing
    """
    
    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        
        # Training parameters
        train_cfg = config.get('training', {})
        self.num_epochs = train_cfg.get('num_epochs', 100)
        self.batch_size = train_cfg.get('batch_size', 64)
        self.latent_dim = train_cfg.get('latent_dim', 100)
        self.lr = train_cfg.get('lr', 0.0002)
        self.beta1 = train_cfg.get('beta1', 0.5)
        self.save_interval = train_cfg.get('save_interval', 10)
        self.checkpoint_interval = train_cfg.get('checkpoint_interval', 10)
        
        # Anti-failure mechanisms
        self.label_smoothing = train_cfg.get('label_smoothing', True)
        self.smooth_real_label = train_cfg.get('smooth_real_label', 0.9)
        self.smooth_fake_label = train_cfg.get('smooth_fake_label', 0.0)
        self.gradient_clip = train_cfg.get('gradient_clip', True)
        self.gradient_clip_value = train_cfg.get('gradient_clip_value', 1.0)
        
        # Initialize GAN
        self.gan = VanillaGAN(
            latent_dim=self.latent_dim,
            img_channels=3,
            feature_maps=64,
            lr=self.lr,
            beta1=self.beta1,
            device=device
        )
        
        # Apply label smoothing
        if self.label_smoothing:
            self.gan.real_label = self.smooth_real_label
            self.gan.fake_label = self.smooth_fake_label
            print(f"✓ Label smoothing enabled: real={self.smooth_real_label}, fake={self.smooth_fake_label}")
        
        # Create directories
        self.setup_directories()
        
        # Initialize logging
        self.setup_logging()
        
        # Training history
        self.history = {
            'epoch': [],
            'g_loss': [],
            'd_loss': [],
            'd_real': [],
            'd_fake': [],
        }
        
        # Fixed noise for consistent sample generation
        self.fixed_noise = torch.randn(64, self.latent_dim, device=device)
    
    def setup_directories(self):
        """Create output directories"""
        self.checkpoint_dir = 'checkpoints'
        self.samples_dir = 'samples'
        self.logs_dir = 'logs'
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        print(f"✓ Created directories: {self.checkpoint_dir}/, {self.samples_dir}/, {self.logs_dir}/")
    
    def setup_logging(self):
        """Setup TensorBoard and CSV logging"""
        # TensorBoard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tensorboard_dir = os.path.join(self.logs_dir, f'tensorboard_{timestamp}')
        self.writer = SummaryWriter(self.tensorboard_dir)
        print(f"✓ TensorBoard logging: {self.tensorboard_dir}")
        
        # CSV logging
        self.csv_file = os.path.join(self.logs_dir, f'training_log_{timestamp}.csv')
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'g_loss', 'd_loss', 'd_real', 'd_fake'])
        print(f"✓ CSV logging: {self.csv_file}")
    
    def save_checkpoint(self, epoch, is_final=False):
        """Save model checkpoint"""
        if is_final:
            g_path = os.path.join(self.checkpoint_dir, 'G_final.pt')
            d_path = os.path.join(self.checkpoint_dir, 'D_final.pt')
            full_path = os.path.join(self.checkpoint_dir, 'gan_final.pt')
        else:
            g_path = os.path.join(self.checkpoint_dir, f'G_epoch_{epoch:03d}.pt')
            d_path = os.path.join(self.checkpoint_dir, f'D_epoch_{epoch:03d}.pt')
            full_path = os.path.join(self.checkpoint_dir, f'gan_epoch_{epoch:03d}.pt')
        
        # Save Generator
        torch.save(self.gan.G.state_dict(), g_path)
        
        # Save Discriminator
        torch.save(self.gan.D.state_dict(), d_path)
        
        # Save full checkpoint (for resuming training)
        torch.save({
            'epoch': epoch,
            'generator': self.gan.G.state_dict(),
            'discriminator': self.gan.D.state_dict(),
            'optimizer_G': self.gan.optimizer_G.state_dict(),
            'optimizer_D': self.gan.optimizer_D.state_dict(),
            'history': self.history,
        }, full_path)
        
        print(f"  ✓ Checkpoint saved: epoch {epoch}")
    
    def save_samples(self, epoch, num_samples=64):
        """Save generated sample images"""
        self.gan.G.eval()
        with torch.no_grad():
            fake_images = self.gan.G(self.fixed_noise[:num_samples])
        self.gan.G.train()
        
        # Denormalize from [-1, 1] to [0, 1]
        fake_images = fake_images * 0.5 + 0.5
        fake_images = fake_images.cpu().numpy()
        
        # Create grid
        nrow = 8
        ncol = 8
        fig, axes = plt.subplots(nrow, ncol, figsize=(12, 12))
        
        for i, ax in enumerate(axes.flatten()):
            if i < len(fake_images):
                img = np.transpose(fake_images[i], (1, 2, 0))
                img = np.clip(img, 0, 1)
                ax.imshow(img)
            ax.axis('off')
        
        plt.suptitle(f'Generated Samples - Epoch {epoch}', fontsize=14)
        plt.tight_layout()
        
        save_path = os.path.join(self.samples_dir, f'epoch_{epoch:03d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also log to TensorBoard
        self.writer.add_images('Generated_Images', 
                               torch.from_numpy(fake_images[:16]), 
                               global_step=epoch)
        
        print(f"  ✓ Samples saved: {save_path}")
    
    def log_metrics(self, epoch, step, g_loss, d_loss, d_real, d_fake):
        """Log metrics to TensorBoard and CSV"""
        global_step = epoch * 1000 + step
        
        # TensorBoard
        self.writer.add_scalar('Loss/Generator', g_loss, global_step)
        self.writer.add_scalar('Loss/Discriminator', d_loss, global_step)
        self.writer.add_scalar('Scores/D_real', d_real, global_step)
        self.writer.add_scalar('Scores/D_fake', d_fake, global_step)
        
        # CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, g_loss, d_loss, d_real, d_fake])
    
    def train(self, train_loader):
        """
        Main training loop
        """
        print("\n" + "=" * 70)
        print("GAN TRAINING STARTED")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Epochs: {self.num_epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Device: {self.device}")
        print(f"  Label smoothing: {self.label_smoothing}")
        print(f"  Gradient clipping: {self.gradient_clip}")
        print(f"  Save interval: every {self.save_interval} epochs")
        print(f"  Checkpoint interval: every {self.checkpoint_interval} epochs")
        print()
        
        total_steps = len(train_loader)
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_g_loss = 0
            epoch_d_loss = 0
            epoch_d_real = 0
            epoch_d_fake = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")
            
            for step, batch in enumerate(pbar):
                real_images = batch['image'].to(self.device)
                batch_size = real_images.size(0)
                
                # =====================
                # Train Discriminator
                # =====================
                d_loss, d_real, d_fake = self.gan.train_discriminator(real_images)
                
                # Gradient clipping for D
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.gan.D.parameters(), 
                        self.gradient_clip_value
                    )
                
                # =====================
                # Train Generator
                # =====================
                g_loss = self.gan.train_generator(batch_size)
                
                # Gradient clipping for G
                if self.gradient_clip:
                    torch.nn.utils.clip_grad_norm_(
                        self.gan.G.parameters(), 
                        self.gradient_clip_value
                    )
                
                # Accumulate losses
                epoch_g_loss += g_loss
                epoch_d_loss += d_loss
                epoch_d_real += d_real
                epoch_d_fake += d_fake
                
                # Update progress bar
                pbar.set_postfix({
                    'D_loss': f'{d_loss:.4f}',
                    'G_loss': f'{g_loss:.4f}',
                    'D(x)': f'{d_real:.3f}',
                    'D(G(z))': f'{d_fake:.3f}'
                })
                
                # Log metrics
                self.log_metrics(epoch, step, g_loss, d_loss, d_real, d_fake)
            
            # Calculate epoch averages
            avg_g_loss = epoch_g_loss / total_steps
            avg_d_loss = epoch_d_loss / total_steps
            avg_d_real = epoch_d_real / total_steps
            avg_d_fake = epoch_d_fake / total_steps
            
            # Store history
            self.history['epoch'].append(epoch)
            self.history['g_loss'].append(avg_g_loss)
            self.history['d_loss'].append(avg_d_loss)
            self.history['d_real'].append(avg_d_real)
            self.history['d_fake'].append(avg_d_fake)
            
            # Log epoch summary to TensorBoard
            self.writer.add_scalar('Epoch/G_loss', avg_g_loss, epoch)
            self.writer.add_scalar('Epoch/D_loss', avg_d_loss, epoch)
            self.writer.add_scalar('Epoch/D_real', avg_d_real, epoch)
            self.writer.add_scalar('Epoch/D_fake', avg_d_fake, epoch)
            
            print(f"\n  Epoch {epoch} Summary:")
            print(f"    G_loss: {avg_g_loss:.4f}")
            print(f"    D_loss: {avg_d_loss:.4f}")
            print(f"    D(real): {avg_d_real:.4f}")
            print(f"    D(fake): {avg_d_fake:.4f}")
            
            # Save samples
            if epoch % self.save_interval == 0 or epoch == 1:
                self.save_samples(epoch)
            
            # Save checkpoint
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch)
        
        # Save final models
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        
        self.save_checkpoint(self.num_epochs, is_final=True)
        self.save_samples(self.num_epochs)
        self.save_training_history()
        
        # Close TensorBoard writer
        self.writer.close()
        
        print(f"\n✓ Final models saved to: {self.checkpoint_dir}/")
        print(f"✓ Sample images saved to: {self.samples_dir}/")
        print(f"✓ Training logs saved to: {self.logs_dir}/")
        print(f"\nTo view TensorBoard logs, run:")
        print(f"  tensorboard --logdir={self.tensorboard_dir}")
    
    def save_training_history(self):
        """Save training history as CSV and plot"""
        # Save as CSV
        history_csv = os.path.join(self.logs_dir, 'training_history.csv')
        with open(history_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'g_loss', 'd_loss', 'd_real', 'd_fake'])
            for i in range(len(self.history['epoch'])):
                writer.writerow([
                    self.history['epoch'][i],
                    self.history['g_loss'][i],
                    self.history['d_loss'][i],
                    self.history['d_real'][i],
                    self.history['d_fake'][i],
                ])
        print(f"  ✓ Training history saved: {history_csv}")
        
        # Plot loss curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history['epoch'], self.history['g_loss'], label='Generator')
        axes[0].plot(self.history['epoch'], self.history['d_loss'], label='Discriminator')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training Losses')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Scores plot
        axes[1].plot(self.history['epoch'], self.history['d_real'], label='D(real)')
        axes[1].plot(self.history['epoch'], self.history['d_fake'], label='D(fake)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Discriminator Scores')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.logs_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Training curves saved: {plot_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Module 3: GAN Training Pipeline")
    print("=" * 70 + "\n")
    
    # Load configuration
    config = load_training_config()
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load training data
    print("\nLoading training data...")
    train_loader, train_dataset = get_preprocessed_dataloader(
        preprocessed_file='preprocessed_data/train_data.pt',
        batch_size=config.get('training', {}).get('batch_size', 64),
        shuffle=True
    )
    
    # Initialize trainer
    trainer = GANTrainer(config, device=device)
    
    # Start training
    trainer.train(train_loader)


if __name__ == "__main__":
    main()
