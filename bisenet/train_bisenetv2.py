#!/usr/bin/env python3
"""
BiSeNet V2 Training Script - SCOPE Smart Building System
CUDA-optimized training for spill segmentation with mixed precision support
Optimized for NVIDIA RTX 4070 (8GB VRAM)
"""

import os
import sys
import time
import yaml
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
try:
    # Use new PyTorch 2.0+ autocast syntax
    from torch.amp import autocast
    AUTOCAST_AVAILABLE = True
except ImportError:
    # Fallback to older syntax
    from torch.cuda.amp import autocast
    AUTOCAST_AVAILABLE = False

# Try to import optional dependencies with graceful handling
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from bisenet.bisenetv2_model import create_bisenetv2
from bisenet.dataset import create_dataloaders
from bisenet.losses import create_loss_function
from bisenet.metrics import SegmentationMetrics


class BiSeNetV2Trainer:
    """
    BiSeNet V2 Trainer with CUDA optimization and mixed precision support
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Setup logging
        self._setup_logging()
        
        # Setup device and mixed precision
        self.device = self._setup_device()
        self.use_mixed_precision = self._should_use_mixed_precision()
        self.scaler = self._setup_grad_scaler()
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.metrics = None
        self.writer = None
        
        # Training state
        self.current_epoch = 0
        self.best_metric = -float('inf') if self.config['validation']['best_mode'] == 'max' else float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Initialize all components
        self._initialize_model()
        self._initialize_dataloaders()
        self._initialize_optimizer()
        self._initialize_criterion()
        self._initialize_metrics()
        self._initialize_logging()
        
        self.logger.info("BiSeNet V2 Trainer initialized successfully")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Mixed Precision: {self.use_mixed_precision}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        device_name = self.config['training'].get('device', 'cuda')
        
        if device_name == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            
            # CUDA optimizations
            cuda_config = self.config.get('environment', {}).get('cuda', {})
            torch.backends.cudnn.benchmark = cuda_config.get('benchmark', True)
            torch.backends.cuda.allow_tf32 = cuda_config.get('allow_tf32', True)
            
        elif device_name == 'mps' and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        
        return device
    
    def _should_use_mixed_precision(self) -> bool:
        """Determine if mixed precision should be used based on device and config"""
        config_mixed_precision = self.config['training'].get('mixed_precision', True)
        
        # Only use mixed precision if explicitly enabled and device supports it
        if not config_mixed_precision:
            return False
        
        # Check device compatibility
        if self.device.type == 'cuda' and torch.cuda.is_available():
            return True
        elif self.device.type == 'mps' and torch.backends.mps.is_available():
            return True
        else:
            # CPU doesn't support mixed precision
            if config_mixed_precision:
                self.logger.warning("Mixed precision requested but not supported on CPU. Disabling.")
            return False
    
    def _setup_grad_scaler(self):
        """Setup gradient scaler for mixed precision training"""
        if not self.use_mixed_precision:
            return None
        
        try:
            # Use new PyTorch 2.0+ syntax
            if hasattr(torch.amp, 'GradScaler'):
                scaler = torch.amp.GradScaler(self.device.type)
                self.logger.info(f"Using new GradScaler for device: {self.device.type}")
                return scaler
            else:
                # Fallback to older syntax for older PyTorch versions
                from torch.cuda.amp import GradScaler
                scaler = GradScaler()
                self.logger.info("Using legacy GradScaler")
                return scaler
                    
        except Exception as e:
            self.logger.warning(f"Failed to initialize GradScaler: {e}. Disabling mixed precision.")
            self.use_mixed_precision = False
            return None
    
    def _initialize_model(self):
        """Initialize BiSeNet V2 model"""
        model_config = self.config['model']
        
        self.model = create_bisenetv2(
            num_classes=model_config['num_classes'],
            aux_mode=model_config.get('aux_mode', 'train'),
            pretrained=model_config.get('pretrained', False)
        )
        
        # Move to device
        self.model.to(self.device)
        
        # Model compilation (PyTorch 2.0+)
        if (hasattr(torch, 'compile') and 
            self.config.get('optimization', {}).get('compile_model', True) and 
            self.device.type == 'cuda'):
            try:
                compile_mode = self.config.get('optimization', {}).get('compile_mode', 'reduce-overhead')
                self.model = torch.compile(self.model, mode=compile_mode)
                self.logger.info(f"Model compiled with mode: {compile_mode}")
            except Exception as e:
                self.logger.warning(f"Model compilation failed: {e}")
                self.logger.warning("Falling back to eager execution mode")
                # Ensure we're not using a partially compiled model
                self.model = create_bisenetv2(
                    num_classes=model_config['num_classes'],
                    aux_mode=model_config.get('aux_mode', 'train'),
                    pretrained=model_config.get('pretrained', False)
                ).to(self.device)
        
        # Log model info
        model_info = self.model.get_model_info()
        self.logger.info(f"Model: {model_info['model_name']}")
        self.logger.info(f"Parameters: {model_info['total_parameters']:,}")
    
    def _initialize_dataloaders(self):
        """Initialize data loaders"""
        dataset_config = self.config['dataset']
        training_config = self.config['training']
        
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=dataset_config['data_dir'],
            batch_size=training_config['batch_size'],
            num_workers=training_config['num_workers'],
            image_size=tuple(dataset_config['image_size']),
            binary_segmentation=dataset_config.get('binary_segmentation', True)
        )
        
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    def _initialize_optimizer(self):
        """Initialize optimizer and scheduler"""
        opt_config = self.config['training']['optimizer']
        
        if opt_config['name'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                weight_decay=opt_config['weight_decay'],
                betas=opt_config.get('betas', [0.9, 0.999]),
                eps=opt_config.get('eps', 1e-8)
            )
        elif opt_config['name'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config['lr'],
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['name']}")
        
        # Initialize scheduler
        sched_config = self.config['training']['scheduler']
        
        if sched_config['name'] == 'CosineAnnealingLR':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config['T_max'],
                eta_min=sched_config.get('eta_min', 0)
            )
        elif sched_config['name'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config['step_size'],
                gamma=sched_config['gamma']
            )
        elif sched_config['name'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config['mode'],
                factor=sched_config['factor'],
                patience=sched_config['patience'],
                min_lr=sched_config.get('min_lr', 0)
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Optimizer: {opt_config['name']}")
        self.logger.info(f"Scheduler: {sched_config['name'] if self.scheduler else 'None'}")
    
    def _initialize_criterion(self):
        """Initialize loss function"""
        loss_config = self.config['loss']
        
        # Calculate class weights if not provided
        class_weights = loss_config.get('class_weights')
        if class_weights is None:
            self.logger.info("Calculating class weights from training data...")
            class_weights = self.train_loader.dataset.get_class_weights()
            class_weights = class_weights.to(self.device)
        
        self.criterion = create_loss_function(
            loss_type='bisenet',
            class_weights=class_weights,
            main_loss_weight=loss_config.get('main_loss_weight', 1.0),
            aux_loss_weight=loss_config.get('aux_loss_weight', 0.4),
            ce_weight=loss_config['main_loss'].get('ce_weight', 1.0),
            dice_weight=loss_config['main_loss'].get('dice_weight', 1.0),
            focal_weight=loss_config['main_loss'].get('focal_weight', 0.0)
        )
        
        self.logger.info(f"Loss function: BiSeNet V2 Loss")
    
    def _initialize_metrics(self):
        """Initialize evaluation metrics"""
        self.metrics = SegmentationMetrics(
            num_classes=self.config['model']['num_classes'],
            ignore_index=-100
        )
    
    def _initialize_logging(self):
        """Initialize experiment logging"""
        self.writer = None
        
        # Initialize TensorBoard if available and configured
        if self.config['logging'].get('use_tensorboard', True) and TENSORBOARD_AVAILABLE:
            try:
                log_dir = Path(self.config['logging']['log_dir']) / 'tensorboard'
                log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
                self.logger.info("TensorBoard logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TensorBoard: {e}")
                self.writer = None
        elif self.config['logging'].get('use_tensorboard', True) and not TENSORBOARD_AVAILABLE:
            self.logger.warning("TensorBoard requested but not available. Install with: pip install tensorboard")
        
        # Initialize Weights & Biases if configured
        if self.config['logging'].get('use_wandb', False) and WANDB_AVAILABLE:
            try:
                wandb_config = self.config['logging']['wandb']
                wandb.init(
                    project=wandb_config['project'],
                    entity=wandb_config.get('entity'),
                    tags=wandb_config.get('tags', []),
                    config=self.config
                )
                self.logger.info("Weights & Biases logging enabled")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
        elif self.config['logging'].get('use_wandb', False) and not WANDB_AVAILABLE:
            self.logger.warning("Weights & Biases requested but not available. Install with: pip install wandb")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {'total': [], 'main': [], 'aux': []}
        
        # Progress tracking
        log_every = self.config['logging'].get('log_every', 10)
        empty_cache_every = self.config.get('optimization', {}).get('empty_cache_every', 100)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            images = batch['image'].to(self.device, non_blocking=True)
            masks = batch['mask'].to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.use_mixed_precision:
                # Use new PyTorch 2.0+ autocast syntax
                with autocast(device_type=self.device.type):
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, masks)
                    total_loss = loss_dict['total_loss']
                
                # Backward pass with gradient scaling
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                max_grad_norm = self.config.get('optimization', {}).get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
            else:
                outputs = self.model(images)
                loss_dict = self.criterion(outputs, masks)
                total_loss = loss_dict['total_loss']
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                max_grad_norm = self.config.get('optimization', {}).get('max_grad_norm', 1.0)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                
                # Optimizer step
                self.optimizer.step()
            
            # Track losses
            epoch_losses['total'].append(total_loss.item())
            epoch_losses['main'].append(loss_dict.get('main_loss', total_loss).item())
            epoch_losses['aux'].append(loss_dict.get('aux_loss', 0.0).item())
            
            # Logging
            if batch_idx % log_every == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {total_loss.item():.4f}, "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )
            
            # Clear cache periodically
            if batch_idx % empty_cache_every == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Calculate average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        val_losses = []
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in self.val_loader:
                images = batch['image'].to(self.device, non_blocking=True)
                masks = batch['mask'].to(self.device, non_blocking=True)
                
                # Forward pass
                if self.use_mixed_precision:
                    with autocast(device_type=self.device.type):
                        outputs = self.model(images)
                        loss_dict = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss_dict = self.criterion(outputs, masks)
                
                val_losses.append(loss_dict['total_loss'].item())
                
                # Update metrics
                if isinstance(outputs, tuple):
                    preds = outputs[0]  # Main prediction
                else:
                    preds = outputs
                
                self.metrics.update(preds, masks)
        
        # Calculate metrics
        avg_loss = np.mean(val_losses)
        metric_results = self.metrics.compute()
        
        results = {'loss': avg_loss}
        results.update(metric_results)
        
        return results
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None
        }
        
        # Save regular checkpoint
        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved: {best_path}")
        
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint.get('best_metric', self.best_metric)
        
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch: {self.current_epoch}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting BiSeNet V2 training...")
        
        num_epochs = self.config['training']['epochs']
        validate_every = self.config['training'].get('validate_every', 5)
        save_every = self.config['training'].get('save_every', 10)
        
        best_metric_name = self.config['validation']['best_metric']
        best_mode = self.config['validation']['best_mode']
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_losses = self.train_epoch()
            
            # Log training losses
            self.logger.info(
                f"Epoch {epoch} Training - "
                f"Total Loss: {train_losses['total']:.4f}, "
                f"Main Loss: {train_losses['main']:.4f}, "
                f"Aux Loss: {train_losses['aux']:.4f}"
            )
            
            # Validation
            if epoch % validate_every == 0:
                val_results = self.validate_epoch()
                
                self.logger.info(
                    f"Epoch {epoch} Validation - "
                    f"Loss: {val_results['loss']:.4f}, "
                    f"IoU: {val_results.get('mean_iou', 0):.4f}, "
                    f"Dice: {val_results.get('mean_dice', 0):.4f}"
                )
                
                # Check for best model
                current_metric = val_results.get(best_metric_name, val_results['loss'])
                is_best = False
                
                if best_mode == 'max':
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        is_best = True
                else:  # min
                    if current_metric < self.best_metric:
                        self.best_metric = current_metric
                        is_best = True
                
                # Tensorboard logging
                if self.writer:
                    self.writer.add_scalar('Loss/Train', train_losses['total'], epoch)
                    self.writer.add_scalar('Loss/Val', val_results['loss'], epoch)
                    
                    for metric_name, value in val_results.items():
                        if metric_name != 'loss':
                            self.writer.add_scalar(f'Metrics/{metric_name}', value, epoch)
                    
                    self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)
                
                # Save checkpoint
                if epoch % save_every == 0 or is_best:
                    self.save_checkpoint(is_best=is_best)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_results.get('loss', train_losses['total']))
                else:
                    self.scheduler.step()
            
            # Log epoch time
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        self.logger.info("Training completed!")
        
        # Final model save
        self.save_checkpoint()
        
        if self.writer:
            self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='BiSeNet V2 Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint for resuming')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = BiSeNetV2Trainer(args.config)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()