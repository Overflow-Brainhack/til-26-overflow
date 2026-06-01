"""Image helpers shared across scoring and the noise phase."""

import base64
from io import BytesIO

from PIL import Image


def thumbnail_b64(img: Image.Image, max_dim: int = 200, quality: int = 80) -> str:
    """Resize to fit within max_dim (longest edge), encode as base64 JPEG."""
    w, h = img.size
    scale = min(1.0, max_dim / max(w, h, 1))
    thumb = img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.Resampling.LANCZOS,
    )
    buf = BytesIO()
    thumb.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
