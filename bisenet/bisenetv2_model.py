#!/usr/bin/env python3
"""
BiSeNet V2 - Bilateral Segmentation Network for Real-time Semantic Segmentation
SCOPE Smart Building System - Optimized for spill detection

Based on: BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation
https://arxiv.org/abs/2004.02147
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import logging


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, dilation: int = 1, bias: bool = False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class DetailBranch(nn.Module):
    """
    Detail Branch for capturing low-level spatial details
    Uses standard convolutions to preserve spatial information
    """
    
    def __init__(self):
        super(DetailBranch, self).__init__()
        
        # Stage 1: 1/2 resolution
        self.stage1 = nn.Sequential(
            ConvBNReLU(3, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        )
        
        # Stage 2: 1/4 resolution
        self.stage2 = nn.Sequential(
            ConvBNReLU(64, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, stride=1, padding=1)
        )
        
        # Stage 3: 1/8 resolution
        self.stage3 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class StemBlock(nn.Module):
    """Stem block for semantic branch"""
    
    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super(StemBlock, self).__init__()
        
        self.conv_branch = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(out_channels, out_channels//2, kernel_size=1, stride=1, padding=0),
            ConvBNReLU(out_channels//2, out_channels, kernel_size=3, stride=2, padding=1)
        )
        
        self.pool_branch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv_branch(x)
        pool_out = self.pool_branch(x)
        return torch.cat([conv_out, pool_out], dim=1)


class GatherExpansionLayer(nn.Module):
    """Gather and Expansion Layer"""
    
    def __init__(self, in_channels: int, out_channels: int, expansion_ratio: int = 6, stride: int = 1):
        super(GatherExpansionLayer, self).__init__()
        
        self.stride = stride
        expanded_channels = in_channels * expansion_ratio
        
        # 3x3 depthwise convolution
        self.conv_dw = ConvBNReLU(
            in_channels, expanded_channels, kernel_size=3, stride=stride, 
            padding=1, bias=False
        )
        
        # 1x1 pointwise convolution
        self.conv_pw = nn.Sequential(
            nn.Conv2d(expanded_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        x = self.conv_dw(x)
        x = self.conv_pw(x)
        return F.relu(x + shortcut, inplace=True)


class ContextEmbeddingBlock(nn.Module):
    """Context Embedding Block for capturing global context"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ContextEmbeddingBlock, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_gap = ConvBNReLU(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = self.gap(x)
        gap = self.conv_gap(gap)
        gap = F.interpolate(gap, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        x = x + gap
        x = self.conv_last(x)
        return x


class SemanticBranch(nn.Module):
    """
    Semantic Branch for capturing high-level semantic information
    Uses efficient operations for speed
    """
    
    def __init__(self):
        super(SemanticBranch, self).__init__()
        
        # Stem block
        self.stem = StemBlock(3, 16)
        
        # Stage 3: 1/8 resolution
        self.stage3 = nn.Sequential(
            GatherExpansionLayer(32, 32, expansion_ratio=3, stride=2),  # 1/8
            GatherExpansionLayer(32, 32, expansion_ratio=3, stride=1),
        )
        
        # Stage 4: 1/16 resolution  
        self.stage4 = nn.Sequential(
            GatherExpansionLayer(32, 64, expansion_ratio=6, stride=2),  # 1/16
            GatherExpansionLayer(64, 64, expansion_ratio=6, stride=1),
        )
        
        # Stage 5: 1/32 resolution
        self.stage5 = nn.Sequential(
            GatherExpansionLayer(64, 128, expansion_ratio=6, stride=2),  # 1/32
            GatherExpansionLayer(128, 128, expansion_ratio=6, stride=1),
            GatherExpansionLayer(128, 128, expansion_ratio=6, stride=1),
            GatherExpansionLayer(128, 128, expansion_ratio=6, stride=1),
        )
        
        # Context Embedding
        self.context_embedding = ContextEmbeddingBlock(128, 128)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)           # 1/4
        x = self.stage3(x)         # 1/8  
        x = self.stage4(x)         # 1/16
        x = self.stage5(x)         # 1/32
        x = self.context_embedding(x)
        return x


class BilateralGuidedAggregation(nn.Module):
    """
    Bilateral Guided Aggregation Layer
    Fuses detail and semantic features effectively
    """
    
    def __init__(self, detail_channels: int = 128, semantic_channels: int = 128, out_channels: int = 128):
        super(BilateralGuidedAggregation, self).__init__()
        
        # Detail branch processing
        self.detail_dwconv = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, kernel_size=3, stride=1, 
                     padding=1, groups=detail_channels, bias=False),
            nn.BatchNorm2d(detail_channels)
        )
        self.detail_conv = nn.Sequential(
            nn.Conv2d(detail_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Semantic branch processing
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(semantic_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.semantic_dwconv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, 
                     padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Fusion
        self.conv_fuse = ConvBNReLU(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, detail_feat: torch.Tensor, semantic_feat: torch.Tensor) -> torch.Tensor:
        # Upsample semantic features to match detail resolution
        semantic_feat = F.interpolate(
            semantic_feat, size=detail_feat.shape[2:], 
            mode='bilinear', align_corners=False
        )
        
        # Process detail features
        detail_feat = self.detail_dwconv(detail_feat)
        detail_feat = self.detail_conv(detail_feat)
        
        # Process semantic features
        semantic_feat = self.semantic_conv(semantic_feat)
        semantic_feat = self.semantic_dwconv(semantic_feat)
        
        # Guided aggregation
        semantic_att = torch.sigmoid(semantic_feat)
        detail_feat = detail_feat * semantic_att
        
        # Fusion
        fused_feat = detail_feat + semantic_feat
        fused_feat = self.conv_fuse(fused_feat)
        
        return fused_feat


class SegmentationHead(nn.Module):
    """Segmentation head for final predictions"""
    
    def __init__(self, in_channels: int, mid_channels: int, num_classes: int, dropout_rate: float = 0.1):
        super(SegmentationHead, self).__init__()
        
        self.conv = ConvBNReLU(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout2d(dropout_rate)
        self.conv_out = nn.Conv2d(mid_channels, num_classes, kernel_size=1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dropout(x)
        x = self.conv_out(x)
        return x


class BiSeNetV2(nn.Module):
    """
    BiSeNet V2 - Bilateral Segmentation Network
    Optimized for real-time semantic segmentation with CUDA acceleration
    """
    
    def __init__(self, num_classes: int = 2, aux_mode: str = 'train'):
        super(BiSeNetV2, self).__init__()
        
        self.num_classes = num_classes
        self.aux_mode = aux_mode
        
        # Two-branch architecture
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        
        # Bilateral guided aggregation
        self.aggregation = BilateralGuidedAggregation(
            detail_channels=128, 
            semantic_channels=128, 
            out_channels=128
        )
        
        # Segmentation heads
        self.seg_head = SegmentationHead(128, 1024, num_classes, dropout_rate=0.1)
        
        # Auxiliary heads for training
        if aux_mode == 'train':
            self.aux_head1 = SegmentationHead(32, 256, num_classes, dropout_rate=0.1)
            self.aux_head2 = SegmentationHead(64, 256, num_classes, dropout_rate=0.1)
            self.aux_head3 = SegmentationHead(128, 256, num_classes, dropout_rate=0.1)
        
        self._initialize_weights()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"BiSeNet V2 initialized with {num_classes} classes")
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor (B, 3, H, W)
            
        Returns:
            If aux_mode == 'eval': Main prediction (B, num_classes, H, W)
            If aux_mode == 'train': Main prediction + auxiliary predictions
        """
        input_size = x.shape[2:]
        
        # Extract features from both branches
        detail_feat = self.detail_branch(x)        # 1/8 resolution
        semantic_feat = self.semantic_branch(x)    # 1/32 resolution
        
        # Aggregate features
        fused_feat = self.aggregation(detail_feat, semantic_feat)  # 1/8 resolution
        
        # Main segmentation head
        main_pred = self.seg_head(fused_feat)
        
        # Upsample to original resolution
        main_pred = F.interpolate(
            main_pred, size=input_size, mode='bilinear', align_corners=False
        )
        
        if self.aux_mode == 'eval' or not self.training:
            return main_pred
        
        # Auxiliary predictions for training
        aux_preds = []
        
        # Get intermediate features from semantic branch
        x_stem = self.semantic_branch.stem(x)
        x_s3 = self.semantic_branch.stage3(x_stem)
        x_s4 = self.semantic_branch.stage4(x_s3)
        
        # Auxiliary head predictions
        aux1 = self.aux_head1(x_s3)
        aux1 = F.interpolate(aux1, size=input_size, mode='bilinear', align_corners=False)
        
        aux2 = self.aux_head2(x_s4)
        aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=False)
        
        aux3 = self.aux_head3(semantic_feat)
        aux3 = F.interpolate(aux3, size=input_size, mode='bilinear', align_corners=False)
        
        aux_preds = [aux1, aux2, aux3]
        
        return main_pred, aux_preds
    
    def get_model_info(self) -> dict:
        """Get model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BiSeNet V2',
            'num_classes': self.num_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'aux_mode': self.aux_mode
        }


def create_bisenetv2(num_classes: int = 2, aux_mode: str = 'train', pretrained: bool = False) -> BiSeNetV2:
    """
    Create BiSeNet V2 model
    
    Args:
        num_classes: Number of segmentation classes
        aux_mode: 'train' for training with aux losses, 'eval' for inference
        pretrained: Load pretrained weights (not implemented yet)
        
    Returns:
        BiSeNet V2 model
    """
    model = BiSeNetV2(num_classes=num_classes, aux_mode=aux_mode)
    
    if pretrained:
        # TODO: Implement pretrained weights loading
        logging.warning("Pretrained weights not implemented yet")
    
    return model


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing BiSeNet V2 on {device}")
    
    # Create model
    model = create_bisenetv2(num_classes=2, aux_mode='train')
    model = model.to(device)
    
    # Test input
    x = torch.randn(2, 3, 512, 512).to(device)
    
    # Forward pass
    print("Testing forward pass...")
    if model.training:
        main_pred, aux_preds = model(x)
        print(f"Main prediction shape: {main_pred.shape}")
        print(f"Number of auxiliary predictions: {len(aux_preds)}")
        for i, aux_pred in enumerate(aux_preds):
            print(f"Aux prediction {i+1} shape: {aux_pred.shape}")
    else:
        pred = model(x)
        print(f"Prediction shape: {pred.shape}")
    
    # Model info
    info = model.get_model_info()
    print("\nModel Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    print("\n✅ BiSeNet V2 model test completed successfully!")