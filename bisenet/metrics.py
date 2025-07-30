#!/usr/bin/env python3
"""
Segmentation Metrics for BiSeNet V2 Training
SCOPE Smart Building System

Implements comprehensive evaluation metrics for spill segmentation:
- IoU (Intersection over Union)
- Dice Coefficient
- Pixel Accuracy
- Precision, Recall, F1-Score
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator
    """
    
    def __init__(self, num_classes: int, ignore_index: int = -100):
        """
        Initialize metrics calculator
        
        Args:
            num_classes: Number of segmentation classes
            ignore_index: Index to ignore in calculations
        """
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        
        # Confusion matrix for accumulating results
        self.confusion_matrix = None
        self.reset()
        
        self.logger = logging.getLogger(__name__)
    
    def reset(self):
        """Reset all metrics"""
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.int64
        )
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update metrics with new predictions and targets
        
        Args:
            predictions: Model predictions (B, C, H, W)
            targets: Ground truth targets (B, H, W)
        """
        # Ensure confusion matrix is on the same device as input tensors
        device = predictions.device
        if self.confusion_matrix.device != device:
            self.confusion_matrix = self.confusion_matrix.to(device)
        
        # Convert predictions to class indices
        if predictions.dim() == 4:  # (B, C, H, W)
            pred_classes = torch.argmax(predictions, dim=1)
        else:  # Already (B, H, W)
            pred_classes = predictions
        
        # Flatten tensors
        pred_flat = pred_classes.view(-1)
        target_flat = targets.view(-1)
        
        # Create mask for valid pixels (ignore_index excluded)
        if self.ignore_index >= 0:
            valid_mask = target_flat != self.ignore_index
            pred_flat = pred_flat[valid_mask]
            target_flat = target_flat[valid_mask]
        
        # Update confusion matrix
        indices = self.num_classes * target_flat + pred_flat
        bincount_result = torch.bincount(
            indices, minlength=self.num_classes**2
        ).reshape(self.num_classes, self.num_classes)
        
        self.confusion_matrix += bincount_result
    
    def compute_iou(self) -> Dict[str, float]:
        """
        Compute Intersection over Union (IoU) metrics
        
        Returns:
            Dictionary with IoU metrics
        """
        # IoU = TP / (TP + FP + FN)
        intersection = torch.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(dim=0) + 
            self.confusion_matrix.sum(dim=1) - 
            intersection
        )
        
        # Avoid division by zero
        iou = intersection.float() / (union.float() + 1e-6)
        
        results = {}
        for i, iou_val in enumerate(iou):
            results[f'iou_class_{i}'] = iou_val.item()
        
        results['mean_iou'] = iou.mean().item()
        
        return results
    
    def compute_dice(self) -> Dict[str, float]:
        """
        Compute Dice Coefficient metrics
        
        Returns:
            Dictionary with Dice metrics
        """
        # Dice = 2 * TP / (2 * TP + FP + FN)
        intersection = torch.diag(self.confusion_matrix)
        denominator = (
            2 * intersection + 
            self.confusion_matrix.sum(dim=0) + 
            self.confusion_matrix.sum(dim=1) - 
            2 * intersection
        )
        
        # Avoid division by zero
        dice = (2 * intersection.float()) / (denominator.float() + 1e-6)
        
        results = {}
        for i, dice_val in enumerate(dice):
            results[f'dice_class_{i}'] = dice_val.item()
        
        results['mean_dice'] = dice.mean().item()
        
        return results
    
    def compute_accuracy(self) -> Dict[str, float]:
        """
        Compute accuracy metrics
        
        Returns:
            Dictionary with accuracy metrics
        """
        # Overall accuracy
        correct = torch.diag(self.confusion_matrix).sum()
        total = self.confusion_matrix.sum()
        overall_accuracy = (correct.float() / (total.float() + 1e-6)).item()
        
        # Per-class accuracy
        class_correct = torch.diag(self.confusion_matrix)
        class_total = self.confusion_matrix.sum(dim=1)
        class_accuracy = class_correct.float() / (class_total.float() + 1e-6)
        
        results = {'overall_accuracy': overall_accuracy}
        for i, acc in enumerate(class_accuracy):
            results[f'accuracy_class_{i}'] = acc.item()
        
        results['mean_class_accuracy'] = class_accuracy.mean().item()
        
        return results
    
    def compute_precision_recall_f1(self) -> Dict[str, float]:
        """
        Compute precision, recall, and F1-score metrics
        
        Returns:
            Dictionary with precision, recall, F1 metrics
        """
        # Precision = TP / (TP + FP)
        tp = torch.diag(self.confusion_matrix)
        fp = self.confusion_matrix.sum(dim=0) - tp
        fn = self.confusion_matrix.sum(dim=1) - tp
        
        precision = tp.float() / (tp.float() + fp.float() + 1e-6)
        recall = tp.float() / (tp.float() + fn.float() + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        results = {}
        
        # Per-class metrics
        for i in range(self.num_classes):
            results[f'precision_class_{i}'] = precision[i].item()
            results[f'recall_class_{i}'] = recall[i].item()
            results[f'f1_class_{i}'] = f1[i].item()
        
        # Average metrics
        results['mean_precision'] = precision.mean().item()
        results['mean_recall'] = recall.mean().item()
        results['mean_f1'] = f1.mean().item()
        
        return results
    
    def compute_confusion_matrix_metrics(self) -> Dict[str, int]:
        """
        Compute confusion matrix based metrics
        
        Returns:
            Dictionary with TP, FP, TN, FN for each class
        """
        results = {}
        
        for i in range(self.num_classes):
            tp = self.confusion_matrix[i, i].item()
            fp = (self.confusion_matrix[:, i].sum() - tp).item()
            fn = (self.confusion_matrix[i, :].sum() - tp).item()
            tn = (self.confusion_matrix.sum() - tp - fp - fn).item()
            
            results[f'tp_class_{i}'] = tp
            results[f'fp_class_{i}'] = fp
            results[f'tn_class_{i}'] = tn
            results[f'fn_class_{i}'] = fn
        
        return results
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics
        
        Returns:
            Dictionary with all computed metrics
        """
        results = {}
        
        # IoU metrics
        results.update(self.compute_iou())
        
        # Dice metrics
        results.update(self.compute_dice())
        
        # Accuracy metrics
        results.update(self.compute_accuracy())
        
        # Precision, Recall, F1 metrics
        results.update(self.compute_precision_recall_f1())
        
        # Confusion matrix metrics
        cm_metrics = self.compute_confusion_matrix_metrics()
        results.update(cm_metrics)
        
        return results
    
    def get_confusion_matrix(self) -> torch.Tensor:
        """
        Get the current confusion matrix
        
        Returns:
            Confusion matrix tensor
        """
        return self.confusion_matrix.clone()
    
    def print_confusion_matrix(self):
        """Print confusion matrix in a readable format"""
        cm = self.confusion_matrix.numpy()
        
        print("\nConfusion Matrix:")
        print("True\\Pred", end="")
        for i in range(self.num_classes):
            print(f"\tClass_{i}", end="")
        print()
        
        for i in range(self.num_classes):
            print(f"Class_{i}", end="")
            for j in range(self.num_classes):
                print(f"\t{cm[i, j]}", end="")
            print()
        print()


class SpillDetectionMetrics:
    """
    Specialized metrics for spill detection (binary segmentation)
    """
    
    def __init__(self):
        """Initialize spill detection metrics"""
        self.base_metrics = SegmentationMetrics(num_classes=2, ignore_index=-100)
    
    def reset(self):
        """Reset all metrics"""
        self.base_metrics.reset()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions"""
        self.base_metrics.update(predictions, targets)
    
    def compute_spill_metrics(self) -> Dict[str, float]:
        """
        Compute spill-specific metrics
        
        Returns:
            Dictionary with spill detection metrics
        """
        all_metrics = self.base_metrics.compute()
        
        # Extract spill class (class 1) metrics
        spill_metrics = {
            'spill_iou': all_metrics['iou_class_1'],
            'spill_dice': all_metrics['dice_class_1'],
            'spill_precision': all_metrics['precision_class_1'],
            'spill_recall': all_metrics['recall_class_1'],
            'spill_f1': all_metrics['f1_class_1'],
            'background_iou': all_metrics['iou_class_0'],
            'overall_accuracy': all_metrics['overall_accuracy'],
            'mean_iou': all_metrics['mean_iou'],
            'mean_dice': all_metrics['mean_dice']
        }
        
        # Calculate spill-specific derived metrics
        tp = all_metrics['tp_class_1']
        fp = all_metrics['fp_class_1']
        fn = all_metrics['fn_class_1']
        tn = all_metrics['tn_class_1']
        
        # Sensitivity (True Positive Rate)
        sensitivity = tp / (tp + fn + 1e-6)
        spill_metrics['sensitivity'] = sensitivity
        
        # Specificity (True Negative Rate)
        specificity = tn / (tn + fp + 1e-6)
        spill_metrics['specificity'] = specificity
        
        # Balanced accuracy
        balanced_accuracy = (sensitivity + specificity) / 2
        spill_metrics['balanced_accuracy'] = balanced_accuracy
        
        return spill_metrics
    
    def compute(self) -> Dict[str, float]:
        """Compute all spill detection metrics"""
        return self.compute_spill_metrics()


def test_metrics():
    """Test metrics computation"""
    print("Testing segmentation metrics...")
    
    # Create dummy data
    batch_size, num_classes, height, width = 2, 2, 32, 32
    
    # Random predictions and targets
    predictions = torch.randn(batch_size, num_classes, height, width)
    targets = torch.randint(0, num_classes, (batch_size, height, width))
    
    # Test general metrics
    print("\nTesting general segmentation metrics:")
    metrics = SegmentationMetrics(num_classes=2)
    metrics.update(predictions, targets)
    results = metrics.compute()
    
    print("General Metrics:")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Test spill-specific metrics
    print("\nTesting spill detection metrics:")
    spill_metrics = SpillDetectionMetrics()
    spill_metrics.update(predictions, targets)
    spill_results = spill_metrics.compute()
    
    print("Spill Metrics:")
    for key, value in spill_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Print confusion matrix
    metrics.print_confusion_matrix()
    
    print("✅ Metrics test completed successfully!")


if __name__ == "__main__":
    test_metrics()