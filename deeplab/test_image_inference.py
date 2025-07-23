#!/usr/bin/env python3
"""
Test COCO Spill Detection Model on a Single Image
Usage: python test_image_inference.py --image path/to/image.jpg --model models/coco_spill_detector.pt
"""

import torch
import cv2
import numpy as np
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

class ImageSpillTester:
    def __init__(self, model_path="models/coco_spill_detector.pt"):
        self.device = self._get_device()
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.load_model()
        
    def _get_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def load_model(self):
        """Load the trained spill detection model"""
        try:
            print(f"Loading model from {self.model_path}")
            
            # Load processor
            self.processor = SegformerImageProcessor.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512"
            )
            
            # Load model
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/segformer-b2-finetuned-ade-512-512",
                num_labels=2,
                ignore_mismatched_sizes=True
            )
            
            # Load trained weights if available
            if Path(self.model_path).exists():
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        # Training checkpoint format
                        state_dict = checkpoint['model_state_dict']
                        print(f"✅ Loaded trained weights from checkpoint at {self.model_path}")
                        if 'epoch' in checkpoint:
                            print(f"   Checkpoint from epoch {checkpoint['epoch']}")
                    else:
                        # Direct state dict format
                        state_dict = checkpoint
                        print(f"✅ Loaded trained weights from {self.model_path}")
                else:
                    # Old format - direct state dict
                    state_dict = checkpoint
                    print(f"✅ Loaded trained weights from {self.model_path}")
                
                # Check if state dict has extra "segformer." prefix (from wrapped models)
                sample_key = next(iter(state_dict.keys()))
                if sample_key.startswith('segformer.segformer.') or sample_key.startswith('segformer.encoder.'):
                    print("   Detected wrapped model format - removing extra 'segformer.' prefix")
                    # Remove the extra "segformer." prefix for wrapped models
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key.startswith('segformer.'):
                            new_key = key[10:]  # Remove "segformer." prefix
                            new_state_dict[new_key] = value
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict
                
                self.model.load_state_dict(state_dict)
            else:
                print(f"⚠️  Model file not found at {self.model_path}, using base model")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded on {self.device}")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise e
    
    def preprocess_image(self, image_path):
        """Preprocess image for model input"""
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size
        
        # Process with SegFormer processor
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs, image, original_size
    
    def predict_spill(self, image_path):
        """Detect spills in the image"""
        inputs, original_image, original_size = self.preprocess_image(image_path)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Get predictions
            predictions = torch.argmax(logits, dim=1)
            
            # Resize to original image size
            w, h = original_size
            predictions = torch.nn.functional.interpolate(
                predictions.unsqueeze(1).float(),
                size=(h, w),
                mode='nearest'
            ).squeeze()
            
            return predictions.cpu().numpy(), original_image
    
    def create_visualization(self, image_path, output_path=None):
        """Create visualization with spill detection overlay"""
        mask, original_image = self.predict_spill(image_path)
        
        # Convert PIL to numpy array
        image_np = np.array(original_image)
        
        # Create overlay
        overlay = image_np.copy()
        spill_mask = (mask == 1)
        
        # Color spills red
        if len(overlay.shape) == 3:
            overlay[spill_mask] = [255, 0, 0]  # Red color for spills
        
        # Calculate spill statistics
        total_pixels = mask.size
        spill_pixels = np.sum(spill_mask)
        spill_percentage = (spill_pixels / total_pixels) * 100
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction mask
        axes[1].imshow(mask, cmap='viridis')
        axes[1].set_title('Spill Detection Mask')
        axes[1].axis('off')
        
        # Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay (Red = Spill)\n{spill_percentage:.2f}% spill coverage')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save result
        if output_path is None:
            output_path = f"spill_detection_result_{Path(image_path).stem}.png"
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Results saved to: {output_path}")
        print(f"   Image size: {image_np.shape[:2]}")
        print(f"   Spill pixels: {spill_pixels:,}")
        print(f"   Total pixels: {total_pixels:,}")
        print(f"   Spill coverage: {spill_percentage:.2f}%")
        
        return mask, spill_percentage > 1.0  # Consider >1% as spill detected

def main():
    parser = argparse.ArgumentParser(description="Test spill detection on a single image")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--model", default="models/coco_spill_detector.pt", 
                       help="Path to trained model")
    parser.add_argument("--output", help="Output path for result image")
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"❌ Image file not found: {args.image}")
        return
    
    # Check if model exists
    if not Path(args.model).exists():
        print(f"⚠️  Model file not found at {args.model}")
        print("Available models:")
        models_dir = Path("models")
        if models_dir.exists():
            for model_file in models_dir.glob("*.pt"):
                print(f"  - {model_file}")
        else:
            print("  - No models directory found")
        print("Will use base model without trained weights")
    
    try:
        # Create tester and run inference
        tester = ImageSpillTester(args.model)
        mask, spill_detected = tester.create_visualization(args.image, args.output)
        
        # Print final result
        if spill_detected:
            print("\n🚨 SPILL DETECTED!")
        else:
            print("\n✅ No significant spills detected")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()