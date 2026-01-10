"""
Module 6: Monitoring & Update Pipeline
========================================
- Runtime monitoring (latency, request tracking, failure logging)
- Model versioning (G_v1, G_v2, G_v3)
- Memorization detection (privacy check)
- Periodic evaluation reports
"""

import os
import json
import time
import shutil
import hashlib
import logging
from datetime import datetime
from functools import wraps
from typing import Optional, Dict, List, Tuple
import numpy as np
import torch
from PIL import Image

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# RUNTIME MONITORING
# =============================================================================

class InferenceMonitor:
    """
    Monitor GAN inference performance in production.
    Tracks: latency, request frequency, failures.
    """
    
    def __init__(self, log_dir='logs/monitoring'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics_file = os.path.join(log_dir, 'inference_metrics.json')
        self.metrics = self._load_metrics()
        
        logger.info("✓ InferenceMonitor initialized")
    
    def _load_metrics(self) -> Dict:
        """Load existing metrics or create new"""
        if os.path.exists(self.metrics_file):
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'total_images_generated': 0,
            'latency_history': [],
            'daily_stats': {},
            'errors': []
        }
    
    def _save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def log_request(self, num_images: int, latency_ms: float, success: bool, 
                    error: Optional[str] = None):
        """Log a generation request"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        self.metrics['total_requests'] += 1
        
        if success:
            self.metrics['successful_requests'] += 1
            self.metrics['total_images_generated'] += num_images
            self.metrics['latency_history'].append({
                'timestamp': datetime.now().isoformat(),
                'num_images': num_images,
                'latency_ms': latency_ms
            })
            # Keep only last 1000 latency records
            self.metrics['latency_history'] = self.metrics['latency_history'][-1000:]
        else:
            self.metrics['failed_requests'] += 1
            self.metrics['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error
            })
            # Keep only last 100 errors
            self.metrics['errors'] = self.metrics['errors'][-100:]
        
        # Update daily stats
        if today not in self.metrics['daily_stats']:
            self.metrics['daily_stats'][today] = {
                'requests': 0,
                'images': 0,
                'avg_latency_ms': 0
            }
        
        self.metrics['daily_stats'][today]['requests'] += 1
        if success:
            self.metrics['daily_stats'][today]['images'] += num_images
            # Update running average
            n = self.metrics['daily_stats'][today]['requests']
            old_avg = self.metrics['daily_stats'][today]['avg_latency_ms']
            self.metrics['daily_stats'][today]['avg_latency_ms'] = \
                old_avg + (latency_ms - old_avg) / n
        
        self._save_metrics()
        
        logger.info(f"Request logged: {num_images} images, {latency_ms:.2f}ms, success={success}")
    
    def get_stats(self) -> Dict:
        """Get current monitoring statistics"""
        latencies = [r['latency_ms'] for r in self.metrics['latency_history']]
        
        return {
            'total_requests': self.metrics['total_requests'],
            'successful_requests': self.metrics['successful_requests'],
            'failed_requests': self.metrics['failed_requests'],
            'success_rate': self.metrics['successful_requests'] / max(1, self.metrics['total_requests']),
            'total_images_generated': self.metrics['total_images_generated'],
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'p50_latency_ms': np.percentile(latencies, 50) if latencies else 0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
            'p99_latency_ms': np.percentile(latencies, 99) if latencies else 0,
        }
    
    def print_report(self):
        """Print monitoring report"""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("INFERENCE MONITORING REPORT")
        print("=" * 60)
        print(f"\nTotal Requests:     {stats['total_requests']}")
        print(f"Successful:         {stats['successful_requests']}")
        print(f"Failed:             {stats['failed_requests']}")
        print(f"Success Rate:       {stats['success_rate']:.2%}")
        print(f"\nTotal Images:       {stats['total_images_generated']}")
        print(f"\nLatency (ms):")
        print(f"  Average:          {stats['avg_latency_ms']:.2f}")
        print(f"  P50:              {stats['p50_latency_ms']:.2f}")
        print(f"  P95:              {stats['p95_latency_ms']:.2f}")
        print(f"  P99:              {stats['p99_latency_ms']:.2f}")
        print("=" * 60)


def monitor_inference(monitor: InferenceMonitor):
    """Decorator to monitor inference function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                latency_ms = (time.time() - start_time) * 1000
                
                # Try to get num_images from result or kwargs
                num_images = kwargs.get('num_images', 1)
                if hasattr(result, '__len__'):
                    num_images = len(result)
                
                monitor.log_request(num_images, latency_ms, success=True)
                return result
            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                monitor.log_request(0, latency_ms, success=False, error=str(e))
                raise
        return wrapper
    return decorator


# =============================================================================
# MODEL VERSIONING
# =============================================================================

class ModelVersionManager:
    """
    Manage GAN model versions for production.
    Supports: versioning, comparison, rollback.
    """
    
    def __init__(self, models_dir='model_versions'):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.versions_file = os.path.join(models_dir, 'versions.json')
        self.versions = self._load_versions()
        
        logger.info(f"✓ ModelVersionManager initialized ({len(self.versions)} versions)")
    
    def _load_versions(self) -> Dict:
        """Load version history"""
        if os.path.exists(self.versions_file):
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {'versions': [], 'current': None}
    
    def _save_versions(self):
        """Save version history"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
    
    def save_version(self, generator_path: str, discriminator_path: str,
                     metrics: Dict, notes: str = "") -> str:
        """
        Save a new model version.
        
        Args:
            generator_path: Path to Generator checkpoint
            discriminator_path: Path to Discriminator checkpoint
            metrics: Evaluation metrics for this version
            notes: Optional notes about this version
        
        Returns:
            Version string (e.g., 'v3')
        """
        version_num = len(self.versions['versions']) + 1
        version_str = f"v{version_num}"
        version_dir = os.path.join(self.models_dir, version_str)
        os.makedirs(version_dir, exist_ok=True)
        
        # Copy model files
        shutil.copy(generator_path, os.path.join(version_dir, 'G.pt'))
        shutil.copy(discriminator_path, os.path.join(version_dir, 'D.pt'))
        
        # Create version metadata
        version_info = {
            'version': version_str,
            'created_at': datetime.now().isoformat(),
            'metrics': metrics,
            'notes': notes,
            'generator_hash': self._file_hash(generator_path),
            'discriminator_hash': self._file_hash(discriminator_path)
        }
        
        self.versions['versions'].append(version_info)
        self.versions['current'] = version_str
        self._save_versions()
        
        logger.info(f"✓ Saved model version: {version_str}")
        return version_str
    
    def _file_hash(self, filepath: str) -> str:
        """Calculate file hash for integrity"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    
    def get_version(self, version: str) -> Tuple[str, str]:
        """Get paths to a specific version's models"""
        version_dir = os.path.join(self.models_dir, version)
        return (
            os.path.join(version_dir, 'G.pt'),
            os.path.join(version_dir, 'D.pt')
        )
    
    def list_versions(self) -> List[Dict]:
        """List all versions with metadata"""
        return self.versions['versions']
    
    def compare_versions(self, v1: str, v2: str) -> Dict:
        """Compare two versions by their metrics"""
        v1_info = next((v for v in self.versions['versions'] if v['version'] == v1), None)
        v2_info = next((v for v in self.versions['versions'] if v['version'] == v2), None)
        
        if not v1_info or not v2_info:
            return {'error': 'Version not found'}
        
        comparison = {
            'v1': v1,
            'v2': v2,
            'metrics_diff': {}
        }
        
        for key in v1_info['metrics']:
            if key in v2_info['metrics']:
                diff = v2_info['metrics'][key] - v1_info['metrics'][key]
                comparison['metrics_diff'][key] = {
                    'v1': v1_info['metrics'][key],
                    'v2': v2_info['metrics'][key],
                    'diff': diff,
                    'improved': diff < 0 if 'loss' in key.lower() or 'fid' in key.lower() else diff > 0
                }
        
        return comparison
    
    def rollback(self, version: str) -> bool:
        """Rollback to a previous version"""
        if not any(v['version'] == version for v in self.versions['versions']):
            logger.error(f"Version {version} not found")
            return False
        
        g_path, d_path = self.get_version(version)
        shutil.copy(g_path, 'checkpoints/G_final.pt')
        shutil.copy(d_path, 'checkpoints/D_final.pt')
        
        self.versions['current'] = version
        self._save_versions()
        
        logger.info(f"✓ Rolled back to version: {version}")
        return True
    
    def print_versions(self):
        """Print all versions"""
        print("\n" + "=" * 60)
        print("MODEL VERSION HISTORY")
        print("=" * 60)
        
        for v in self.versions['versions']:
            current = " (CURRENT)" if v['version'] == self.versions['current'] else ""
            print(f"\n{v['version']}{current}")
            print(f"  Created: {v['created_at']}")
            if v['notes']:
                print(f"  Notes: {v['notes']}")
            print(f"  Metrics:")
            for k, val in v['metrics'].items():
                print(f"    {k}: {val:.4f}")
        
        print("=" * 60)


# =============================================================================
# MEMORIZATION DETECTION (Privacy Check)
# =============================================================================

class MemorizationDetector:
    """
    Detect if GAN is memorizing training samples.
    Uses nearest neighbor comparison in feature space.
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        
        # Use pretrained ResNet for feature extraction
        import torchvision.models as models
        self.feature_extractor = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor.fc = torch.nn.Identity()
        self.feature_extractor = self.feature_extractor.to(device)
        self.feature_extractor.eval()
        
        self.transform = torch.nn.Sequential(
            torch.nn.Upsample(size=(224, 224), mode='bilinear'),
        )
        
        logger.info("✓ MemorizationDetector initialized")
    
    def extract_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from images"""
        # Denormalize from [-1, 1] to [0, 1]
        images = images * 0.5 + 0.5
        
        # Resize for ResNet
        images = self.transform(images)
        
        # Normalize for ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        images = (images - mean) / std
        
        with torch.no_grad():
            features = self.feature_extractor(images.to(self.device))
        
        return features.cpu().numpy()
    
    def check_memorization(self, generated_images: torch.Tensor, 
                           real_images: torch.Tensor,
                           threshold: float = 0.95) -> Dict:
        """
        Check if generated images are too similar to training data.
        
        Args:
            generated_images: Tensor of generated images
            real_images: Tensor of real training images
            threshold: Cosine similarity threshold (above = potential memorization)
        
        Returns:
            Dict with memorization analysis results
        """
        logger.info("Checking for memorization...")
        
        # Extract features
        gen_features = self.extract_features(generated_images)
        real_features = self.extract_features(real_images)
        
        # Normalize features
        gen_features = gen_features / np.linalg.norm(gen_features, axis=1, keepdims=True)
        real_features = real_features / np.linalg.norm(real_features, axis=1, keepdims=True)
        
        # Compute cosine similarity matrix
        similarity_matrix = gen_features @ real_features.T
        
        # Find max similarity for each generated image
        max_similarities = np.max(similarity_matrix, axis=1)
        nearest_neighbors = np.argmax(similarity_matrix, axis=1)
        
        # Identify potential memorization
        memorized_mask = max_similarities > threshold
        num_memorized = np.sum(memorized_mask)
        
        results = {
            'num_generated': len(generated_images),
            'num_real': len(real_images),
            'threshold': threshold,
            'max_similarity_mean': float(np.mean(max_similarities)),
            'max_similarity_std': float(np.std(max_similarities)),
            'max_similarity_max': float(np.max(max_similarities)),
            'num_potential_memorized': int(num_memorized),
            'memorization_rate': float(num_memorized / len(generated_images)),
            'memorized_indices': np.where(memorized_mask)[0].tolist(),
            'nearest_neighbors': nearest_neighbors.tolist(),
            'is_safe': num_memorized == 0
        }
        
        if results['is_safe']:
            logger.info("✓ No memorization detected - model is safe")
        else:
            logger.warning(f"⚠️ Potential memorization: {num_memorized} images above threshold")
        
        return results
    
    def print_report(self, results: Dict):
        """Print memorization check report"""
        print("\n" + "=" * 60)
        print("MEMORIZATION CHECK REPORT")
        print("=" * 60)
        print(f"\nGenerated images analyzed: {results['num_generated']}")
        print(f"Real images compared:      {results['num_real']}")
        print(f"Similarity threshold:      {results['threshold']}")
        print(f"\nMax Similarity Stats:")
        print(f"  Mean:                    {results['max_similarity_mean']:.4f}")
        print(f"  Std:                     {results['max_similarity_std']:.4f}")
        print(f"  Max:                     {results['max_similarity_max']:.4f}")
        print(f"\nMemorization Analysis:")
        print(f"  Potential memorized:     {results['num_potential_memorized']}")
        print(f"  Memorization rate:       {results['memorization_rate']:.2%}")
        
        if results['is_safe']:
            print(f"\n✓ STATUS: SAFE - No memorization detected")
        else:
            print(f"\n⚠️ STATUS: WARNING - Potential memorization found")
            print(f"  Affected indices: {results['memorized_indices'][:10]}...")
        
        print("=" * 60)


# =============================================================================
# PERIODIC EVALUATION REPORT
# =============================================================================

def generate_evaluation_report(output_dir='reports'):
    """Generate a comprehensive evaluation report"""
    from evaluation import run_full_evaluation
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Run evaluation
    metrics, figures = run_full_evaluation(
        checkpoint_path='checkpoints/gan_final.pt',
        num_samples=64
    )
    
    # Generate report
    report_path = os.path.join(output_dir, f'evaluation_report_{timestamp}.md')
    
    with open(report_path, 'w') as f:
        f.write("# GAN Evaluation Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Metrics Summary\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Realism Score | {metrics['realism']['realism_ratio']:.4f} |\n")
        f.write(f"| Diversity Score | {metrics['diversity']['cluster_diversity']:.4f} |\n")
        f.write(f"| FID Score | {metrics['fid']['fid_score']:.4f} |\n")
        f.write(f"| Mode Collapse | {'Yes' if metrics['mode_collapse']['mode_collapse_detected'] else 'No'} |\n")
        
        f.write("\n## Generated Figures\n\n")
        for name, path in figures.items():
            f.write(f"- {name}: `{path}`\n")
    
    logger.info(f"✓ Evaluation report saved: {report_path}")
    return report_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Module 6: Monitoring & Update Pipeline")
    print("=" * 60 + "\n")
    
    # Initialize components
    monitor = InferenceMonitor()
    version_manager = ModelVersionManager()
    memorization_detector = MemorizationDetector()
    
    # Print current stats
    monitor.print_report()
    version_manager.print_versions()
    
    # Example: Check memorization
    print("\nRunning memorization check...")
    
    from data_loader import get_preprocessed_dataloader
    from inference import GANInference
    
    # Load data
    train_loader, _ = get_preprocessed_dataloader(
        preprocessed_file='preprocessed_data/train_data.pt',
        batch_size=64
    )
    real_batch = next(iter(train_loader))['image']
    
    # Generate fake images
    engine = GANInference(checkpoint_path='checkpoints/G_final.pt')
    generated = torch.from_numpy(engine.generate(64)).permute(0, 3, 1, 2).float() / 255.0
    generated = generated * 2 - 1  # Convert to [-1, 1]
    
    # Check memorization
    results = memorization_detector.check_memorization(generated, real_batch)
    memorization_detector.print_report(results)
    
    print("\n✓ Monitoring pipeline ready!")


if __name__ == "__main__":
    main()
