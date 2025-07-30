#!/usr/bin/env python3
"""
Loss Functions for BiSeNet V2 Spill Segmentation
SCOPE Smart Building System

Implements combined loss functions for optimal spill boundary detection:
- CrossEntropyLoss for class predictions
- DiceLoss for boundary preservation
- FocalLoss for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import logging


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks
    Especially effective for boundary preservation and handling class imbalance
    """
    
    def __init__(self, smooth: float = 1e-5, ignore_index: int = -100):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice Loss
        
        Args:
            pred: Prediction tensor (B, C, H, W)
            target: Target tensor (B, H, W)
            
        Returns:
            Dice loss value
        """
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Create one-hot encoding for target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Handle ignore index
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        # Calculate Dice coefficient for each class
        dice_scores = []
        for c in range(num_classes):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target_one_hot[:, c:c+1, :, :]
            
            intersection = (pred_c * target_c).sum(dim=(2, 3))
            union = pred_c.sum(dim=(2, 3)) + target_c.sum(dim=(2, 3))
            
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice)
        
        # Average dice score across classes and batch
        dice_score = torch.stack(dice_scores, dim=1).mean()
        
        # Return 1 - dice for loss (minimize)
        return 1 - dice_score


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Focuses learning on hard examples
    """
    
    def __init__(self, alpha: Optional[Union[float, torch.Tensor]] = None, 
                 gamma: float = 2.0, ignore_index: int = -100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal Loss
        
        Args:
            pred: Prediction tensor (B, C, H, W)
            target: Target tensor (B, H, W)
            
        Returns:
            Focal loss value
        """
        # Calculate cross entropy
        ce_loss = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        
        # Calculate probabilities
        pt = torch.exp(-ce_loss)
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            if isinstance(self.alpha, torch.Tensor):
                alpha_t = self.alpha[target]
            else:
                alpha_t = self.alpha
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for BiSeNet V2 training
    Combines CrossEntropy, Dice, and optionally Focal losses
    """
    
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        focal_alpha: Optional[float] = None,
        focal_gamma: float = 2.0
    ):
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        # CrossEntropy Loss
        self.ce_loss = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )
        
        # Dice Loss
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        
        # Focal Loss (optional)
        if focal_weight > 0:
            self.focal_loss = FocalLoss(
                alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index
            )
        else:
            self.focal_loss = None
        
        self.logger = logging.getLogger(__name__)
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """
        Calculate combined loss
        
        Args:
            pred: Prediction tensor (B, C, H, W)
            target: Target tensor (B, H, W)
            
        Returns:
            Dictionary with total loss and individual loss components
        """
        losses = {}
        
        # CrossEntropy Loss
        if self.ce_weight > 0:
            losses['ce_loss'] = self.ce_loss(pred, target)
        else:
            losses['ce_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Dice Loss
        if self.dice_weight > 0:
            losses['dice_loss'] = self.dice_loss(pred, target)
        else:
            losses['dice_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Focal Loss (optional)
        if self.focal_weight > 0 and self.focal_loss is not None:
            losses['focal_loss'] = self.focal_loss(pred, target)
        else:
            losses['focal_loss'] = torch.tensor(0.0, device=pred.device)
        
        # Combined loss
        total_loss = (
            self.ce_weight * losses['ce_loss'] +
            self.dice_weight * losses['dice_loss'] +
            self.focal_weight * losses['focal_loss']
        )
        
        losses['total_loss'] = total_loss
        
        return losses


class BiSeNetV2Loss(nn.Module):
    """
    BiSeNet V2 specific loss function
    Handles main prediction and auxiliary predictions during training
    """
    
    def __init__(
        self,
        main_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.4,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        ignore_index: int = -100
    ):
        super(BiSeNetV2Loss, self).__init__()
        
        self.main_loss_weight = main_loss_weight
        self.aux_loss_weight = aux_loss_weight
        
        # Main loss function
        self.main_loss_fn = CombinedLoss(
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            focal_weight=focal_weight,
            class_weights=class_weights,
            ignore_index=ignore_index
        )
        
        # Auxiliary loss function (simpler - only CrossEntropy)
        self.aux_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights, ignore_index=ignore_index
        )
        
        self.logger = logging.getLogger(__name__)
    
    def forward(
        self, 
        predictions: Union[torch.Tensor, tuple], 
        target: torch.Tensor
    ) -> dict:
        """
        Calculate BiSeNet V2 loss
        
        Args:
            predictions: Main prediction or (main_pred, aux_preds) tuple
            target: Target tensor (B, H, W)
            
        Returns:
            Dictionary with total loss and loss components
        """
        if isinstance(predictions, tuple):
            main_pred, aux_preds = predictions
            
            # Main loss
            main_losses = self.main_loss_fn(main_pred, target)
            
            # Auxiliary losses
            aux_loss_total = torch.tensor(0.0, device=main_pred.device)
            for aux_pred in aux_preds:
                aux_loss_total += self.aux_loss_fn(aux_pred, target)
            
            aux_loss_total = aux_loss_total / len(aux_preds)
            
            # Combined loss
            total_loss = (
                self.main_loss_weight * main_losses['total_loss'] +
                self.aux_loss_weight * aux_loss_total
            )
            
            losses = {
                'total_loss': total_loss,
                'main_loss': main_losses['total_loss'],
                'main_ce_loss': main_losses['ce_loss'],
                'main_dice_loss': main_losses['dice_loss'],
                'aux_loss': aux_loss_total
            }
            
        else:
            # Only main prediction (evaluation mode)
            main_losses = self.main_loss_fn(predictions, target)
            losses = main_losses
        
        return losses


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss
    Alternative to Dice loss for boundary preservation
    """
    
    def __init__(self, smooth: float = 1e-5, ignore_index: int = -100):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU Loss
        
        Args:
            pred: Prediction tensor (B, C, H, W)
            target: Target tensor (B, H, W)
            
        Returns:
            IoU loss value
        """
        # Apply softmax to predictions
        pred = F.softmax(pred, dim=1)
        
        # Create one-hot encoding for target
        num_classes = pred.shape[1]
        target_one_hot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Handle ignore index
        if self.ignore_index >= 0:
            mask = (target != self.ignore_index).float().unsqueeze(1)
            pred = pred * mask
            target_one_hot = target_one_hot * mask
        
        # Calculate IoU for each class
        iou_scores = []
        for c in range(num_classes):
            pred_c = pred[:, c:c+1, :, :]
            target_c = target_one_hot[:, c:c+1, :, :]
            
            intersection = (pred_c * target_c).sum(dim=(2, 3))
            union = pred_c.sum(dim=(2, 3)) + target_c.sum(dim=(2, 3)) - intersection
            
            iou = (intersection + self.smooth) / (union + self.smooth)
            iou_scores.append(iou)
        
        # Average IoU score across classes and batch
        iou_score = torch.stack(iou_scores, dim=1).mean()
        
        # Return 1 - IoU for loss (minimize)
        return 1 - iou_score


def create_loss_function(
    loss_type: str = 'combined',
    class_weights: Optional[torch.Tensor] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create loss functions
    
    Args:
        loss_type: Type of loss ('ce', 'dice', 'focal', 'combined', 'bisenet')
        class_weights: Class weights for handling imbalance
        **kwargs: Additional arguments for loss functions
        
    Returns:
        Loss function module
    """
    if loss_type == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights, **kwargs)
    
    elif loss_type == 'dice':
        return DiceLoss(**kwargs)
    
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    
    elif loss_type == 'iou':
        return IoULoss(**kwargs)
    
    elif loss_type == 'combined':
        return CombinedLoss(class_weights=class_weights, **kwargs)
    
    elif loss_type == 'bisenet':
        return BiSeNetV2Loss(class_weights=class_weights, **kwargs)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing loss functions on {device}")
    
    # Create dummy data
    batch_size, num_classes, height, width = 2, 2, 64, 64
    pred = torch.randn(batch_size, num_classes, height, width).to(device)
    target = torch.randint(0, num_classes, (batch_size, height, width)).to(device)
    
    # Test individual losses
    print("\nTesting individual losses:")
    
    # CrossEntropy
    ce_loss = nn.CrossEntropyLoss()
    ce_val = ce_loss(pred, target)
    print(f"CrossEntropy Loss: {ce_val.item():.4f}")
    
    # Dice Loss
    dice_loss = DiceLoss()
    dice_val = dice_loss(pred, target)
    print(f"Dice Loss: {dice_val.item():.4f}")
    
    # Focal Loss
    focal_loss = FocalLoss(gamma=2.0)
    focal_val = focal_loss(pred, target)
    print(f"Focal Loss: {focal_val.item():.4f}")
    
    # IoU Loss
    iou_loss = IoULoss()
    iou_val = iou_loss(pred, target)
    print(f"IoU Loss: {iou_val.item():.4f}")
    
    # Combined Loss
    print("\nTesting combined loss:")
    combined_loss = CombinedLoss(ce_weight=1.0, dice_weight=1.0)
    combined_val = combined_loss(pred, target)
    print(f"Combined Loss Components:")
    for key, value in combined_val.items():
        print(f"  {key}: {value.item():.4f}")
    
    # BiSeNet V2 Loss with auxiliary predictions
    print("\nTesting BiSeNet V2 loss:")
    bisenet_loss = BiSeNetV2Loss()
    
    # Simulate auxiliary predictions
    aux_preds = [
        torch.randn(batch_size, num_classes, height, width).to(device),
        torch.randn(batch_size, num_classes, height, width).to(device),
        torch.randn(batch_size, num_classes, height, width).to(device)
    ]
    
    bisenet_val = bisenet_loss((pred, aux_preds), target)
    print(f"BiSeNet V2 Loss Components:")
    for key, value in bisenet_val.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\n✅ Loss function tests completed successfully!")