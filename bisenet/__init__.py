"""
BiSeNet V2 Implementation for SCOPE Smart Building System
Real-time bilateral segmentation for spill detection
"""

from .bisenetv2_model import BiSeNetV2
from .bisenetv2_spill_detector import BiSeNetV2SpillDetector

__all__ = ['BiSeNetV2', 'BiSeNetV2SpillDetector']