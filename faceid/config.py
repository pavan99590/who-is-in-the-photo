from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "models"
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


@dataclass(frozen=True)
class DetectConfig:
    """Everything that affects which faces are found and how they are embedded."""

    model_pack: str = "buffalo_l"   # SCRFD-10G detector + ArcFace R50 (w600k)
    det_size: int = 1280            # detector input size for GROUP photos (small faces need this)
    gallery_det_size: int = 640     # reference portraits have one big face: cheaper is fine
    det_thresh: float = 0.5
    max_side: int = 3200            # group photos larger than this are downscaled first
    gallery_max_side: int = 1600
    min_face_px: int = 20           # ignore detections smaller than this (too small to embed)
    tiling: bool = False            # extra pass over overlapping tiles for very large photos
    tile_overlap: float = 0.25
    nms_iou: float = 0.4

    @property
    def cache_tag(self) -> str:
        return f"{self.model_pack}-{self.gallery_det_size}-{self.gallery_max_side}-{self.det_thresh}-sel3"  # sel3 = central-face rule + thumbnails


@dataclass(frozen=True)
class MatchConfig:
    """Open-set decision rule. Defaults are starting points: calibrate with scripts/evaluate.py."""

    threshold: float = 0.40   # min cosine similarity to accept a name
    min_margin: float = 0.0   # assigned score must beat every other person by at least this
