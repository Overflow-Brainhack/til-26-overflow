from abc import ABC, abstractmethod
import random
import torch
import torch.nn.functional as F
import torchvision.io as tvio
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


class Manager(ABC):
    defense = False

    def __init__(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"

    @staticmethod
    def _median_filter(img: torch.Tensor, size: int = 3) -> torch.Tensor:
        p = size // 2
        padded = F.pad(img.float().unsqueeze(0), (p, p, p, p), mode="reflect")
        unfolded = padded.unfold(2, size, 1).unfold(
            3, size, 1
        )  # (1, C, H, W, size, size)
        C, H, W = img.shape
        return (
            unfolded.contiguous()
            .view(1, C, H, W, size * size)
            .median(dim=-1)
            .values.squeeze(0)
            .to(torch.uint8)
        )

    def _preprocess(self, image: bytes) -> bytes:
        """Strip adversarial perturbations: resize jitter, median filter, bit-depth reduction, JPEG recompression."""
        buf = torch.frombuffer(image, dtype=torch.uint8)
        img = tvio.decode_image(buf, mode=tvio.ImageReadMode.RGB)  # (3, H, W) uint8
        _, H, W = img.shape

        scale = random.uniform(0.9, 1.0)
        img = TF.resize(
            img,
            [int(H * scale), int(W * scale)],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        img = TF.resize(
            img, [H, W], interpolation=InterpolationMode.BILINEAR, antialias=True
        )

        img = self._median_filter(img)

        return tvio.encode_jpeg(img, quality=80).numpy().tobytes()

    @abstractmethod
    def infer(self, image: bytes) -> list[dict[str, int | list[float]]]: ...

    def cv(self, image: bytes) -> list[dict[str, int | list[float]]]:
        if self.defense:
            return self.infer(self._preprocess(image))

        return self.infer(image)
