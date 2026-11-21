from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

try:  # optional: iPhone .heic support
    import pillow_heif

    pillow_heif.register_heif_opener()
except Exception:  # pragma: no cover
    pass

Image.MAX_IMAGE_PIXELS = None  # big phone/camera photos are fine


class ImageLoadError(Exception):
    """The file is not a readable image."""


@dataclass
class LoadedImage:
    bgr: np.ndarray                 # processed image, BGR uint8 (what the detector sees)
    scale: float                    # processed_size / original_size (1.0 if not downscaled)
    original_size: tuple[int, int]  # (width, height) after EXIF rotation, before downscale


def open_pil(source: bytes | bytearray | str | Path) -> Image.Image:
    """Open an image, apply EXIF orientation, return RGB. Shared by loader and annotator
    so box coordinates always line up with the picture the user sees."""
    try:
        if isinstance(source, (bytes, bytearray)):
            img = Image.open(io.BytesIO(source))
        else:
            img = Image.open(source)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(f"Cannot read image: {exc}") from exc


def load_image(source: bytes | bytearray | str | Path, max_side: int = 3200) -> LoadedImage:
    img = open_pil(source)
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    rgb = np.asarray(img)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return LoadedImage(bgr=bgr, scale=scale, original_size=(w, h))
