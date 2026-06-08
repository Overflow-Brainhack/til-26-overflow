"""Manages the CV model."""

from turbojpeg import TurboJPEG, TJPF_RGB
import torchvision
import numpy as np
from rfdetr import RFDETRLarge
from typing import Any
import torch

class RFDETRManager:
    def __init__(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.model = RFDETRLarge(num_classes=18, resolution=1280, pretrain_weights='models/extra_data_checkpoint26_best_ema.pth')

        self.model.optimize_for_inference(compile=True, dtype=torch.float16)
        
        # Initialize TurboJPEG once during startup to avoid overhead per request
        self.jpeg = TurboJPEG()

        self._warmup()
        
    def _warmup(self):
        """Warm up the GPU, nvJPEG, and compiled model graphs to stabilize inference latency."""
        print("[RFDETRManager] Starting pipeline warm-up...")
        
        with torch.inference_mode():
            # 1. Warm up the torchvision nvJPEG decoder with a minimal valid JPEG payload
            # This creates a dummy 1x1 black JPEG byte stream
            dummy_jpeg = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00`\x00`\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\x27"2#\x1c\x1c7H079\x31\x34\x3c\x3cBase\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9'
            try:
                for _ in range(3):
                    _ = self.torchvision_decode(dummy_jpeg)
            except Exception as e:
                print(f"[RFDETRManager] Note: Torchvision hardware decode warm-up skipped or unsupported: {e}")

            # 2. Warm up the model's compiled forward pass
            # We use a batch-size of 1 and match your target 1024 resolution configuration
            dummy_tensor = torch.rand(3, 1024, 1024, device='cuda', dtype=torch.float32)   
            
            # The first few passes trigger compilation. 15-20 passes lock in CUDA graphs.
            for i in range(20):
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # Call the internal prediction method directly to avoid list-comprehension overhead during warm-up
                    _ = self.model.predict(dummy_tensor, threshold=0.4)
                
            # Synchronize CUDA to guarantee all kernels are fully baked before proceeding
            torch.cuda.synchronize()
            
        print("[RFDETRManager] Pipeline warm-up complete. Ready for production requests.")
        
    # 2s/it --> 10.5 min
    def turbojpeg_decode(self, image: bytes):
        # TurboJPEG directly decodes bytes to a NumPy array in memory.
        # We explicitly pass TJPF_RGB to ensure it decodes to RGB format (matching PIL's behavior).
        im = self.jpeg.decode(image, pixel_format=TJPF_RGB)
        tensor = torch.from_numpy(im).to('cuda', non_blocking=True)
        tensor = tensor.permute(2, 0, 1).float()
        tensor.mul_(1.0/255.0)
        return tensor

    # 1.9s/it --> 10 min
    # but gives slightly lower score and speed?
    def torchvision_decode(self, image: bytes):
        # 1. Zero-copy conversion of raw bytes to a 1D uint8 PyTorch tensor
        img_1d = torch.frombuffer(image, dtype=torch.uint8)
            
        # 2. Decode directly to a (C, H, W) PyTorch tensor.
        # Passing device='cuda' attempts hardware decoding via nvJPEG.
        # (If your torchvision setup doesn't support nvJPEG, omit device='cuda' 
        # and append .to('cuda') to the end of this line).
        tensor = torchvision.io.decode_jpeg(img_1d, device='cuda')
        tensor = tensor.float() / 255.0
        return tensor
        
    def _preprocess(self, image: bytes) -> bytes:
        """Strip adversarial perturbations with minimal accuracy impact."""
        im = Image.open(BytesIO(image)).convert("RGB")
        # Gaussian blur radius=1: kills sub-pixel adversarial noise, within YOLO blur augment range
        im = im.filter(ImageFilter.GaussianBlur(radius=1))
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def infer(self, image: bytes) -> list[dict[str, Any]]:
        # tuning threshold for 1024_aug_best_total
        # 0.1 -> 0.703
        # 0.3 -> 0.745
        # 0.4 -> 0.749 (top score)
        # 0.5 -> 0.746 
        with torch.inference_mode():
            tensor = self.turbojpeg_decode(image)
            #tensor = self.torchvision_decode(image)
                                    
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                detections = self.model.predict(tensor, threshold=0.4)

        # Extract indices directly without referencing descriptive metadata dictionaries
        preds = [
            {
                "category_id": int(det[3]), 
                "bbox": [float(det[0][0]), float(det[0][1]), float(det[0][2] - det[0][0]), float(det[0][3] - det[0][1])]
            }
            for det in detections
        ]
            
        return preds

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions.
        """
        # image = self._preprocess(image)
        return self.infer(image)
