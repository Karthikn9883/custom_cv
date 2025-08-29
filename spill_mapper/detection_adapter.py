# spill_mapper/detection_adapter.py
import numpy as np, cv2, torch, torchvision
from pathlib import Path

class Detector:
    """
    DeepLabV3 (.pth) segmentation → 'spill' mask.
    - Preproc: BGR -> RGB, normalize (ImageNet), stretch to 640x640 (to match your training).
    - Inference: FP16 on CUDA (if available), cudnn.benchmark enabled.
    - Postproc: argmax -> class map -> connected components -> bbox/centroid.
    """
    def __init__(self, weights:str, spill_class:int=1, infer_size=(640,640)):
        assert weights and weights.endswith(".pth"), "Provide DeepLab .pth weights"
        self.spill_class = int(spill_class)
        self.input_w, self.input_h = infer_size  # (640,640)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True

        # Load checkpoint
        ckpt = torch.load(weights, map_location="cpu")
        sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        # Infer num classes
        num_classes = None
        for k,v in sd.items():
            if "classifier.4.weight" in k and v.ndim==4:
                num_classes = int(v.shape[0]); break
        if num_classes is None: num_classes = 2

        # Build model & load
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, num_classes=num_classes)
        clean = {}
        for k,v in sd.items():
            k = k.replace("module.","").replace("model.","")
            clean[k] = v
        self.model.load_state_dict(clean, strict=False)
        self.model.to(self.device).eval()
        # FP16 if CUDA
        self.use_half = self.device.type == "cuda"
        if self.use_half:
            self.model.half()

        # ImageNet norm (typical for torchvision)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _preprocess(self, bgr):
        # stretch to 640x640 (matches training exactly)
        img = cv2.resize(bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        ten = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
        if self.use_half: ten = ten.half()
        return ten.to(self.device), img.shape[:2]  # (H,W) of 640x640

    def _postprocess(self, logits, orig_shape, src_wh):
        # logits: [1, C, H, W], already at 640x640 since we fed fixed size
        seg = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.uint8)  # [H,W]
        mask = (seg == self.spill_class).astype(np.uint8) * 255

        # Find components
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        results = []
        H640, W640 = orig_shape
        for i in range(1, n):  # skip background
            x,y,w,h,area = stats[i]
            if area < 200:   # tune min area
                continue
            cx, cy = cents[i]  # in 640x640 space
            # Scale back to source frame pixel space
            src_w, src_h = src_wh
            sx = src_w / float(W640)
            sy = src_h / float(H640)
            x1,y1,x2,y2 = x*sx, y*sy, (x+w)*sx, (y+h)*sy
            u,v = cx*sx, cy*sy
            # confidence proxy: foreground ratio in box
            roi = mask[y:y+h, x:x+w]
            conf = float(np.clip(roi.mean()/255.0, 0.0, 1.0))
            results.append({
                "bbox":[float(x1),float(y1),float(x2),float(y2)],
                "center":[float(u),float(v)],
                "conf":conf,
                "label":"spill"
            })
        return results

    def infer(self, bgr):
        src_h, src_w = bgr.shape[:2]
        ten, (H, W) = self._preprocess(bgr)
        with torch.no_grad():
            out = self.model(ten)["out"]   # [1,C,640,640]
        return self._postprocess(out, (H,W), (src_w, src_h))
