from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import DEFAULT_MODEL_ROOT, DetectConfig


class ModelsNotFoundError(RuntimeError):
    """Model files are missing. They are fetched once at install time, never at runtime."""


@dataclass
class DetectedFace:
    bbox: np.ndarray            # float32 [x1, y1, x2, y2] in the coordinates of the image given
    det_score: float
    embedding: np.ndarray       # L2-normalised float32 (512,)
    kps: np.ndarray | None = None

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def nms(faces: list[DetectedFace], iou_thresh: float) -> list[DetectedFace]:
    """Keep the highest-scoring face among overlapping detections."""
    kept: list[DetectedFace] = []
    for f in sorted(faces, key=lambda f: f.det_score, reverse=True):
        if all(iou(f.bbox, k.bbox) < iou_thresh for k in kept):
            kept.append(f)
    return kept


def tile_boxes(w: int, h: int, tile: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Overlapping square-ish windows covering a w x h image."""
    stride = max(1, int(tile * (1 - overlap)))

    def starts(total: int) -> list[int]:
        if total <= tile:
            return [0]
        s = list(range(0, total - tile + 1, stride))
        if s[-1] + tile < total:
            s.append(total - tile)
        return s

    return [(x, y, min(x + tile, w), min(y + tile, h)) for y in starts(h) for x in starts(w)]


class FaceEngine:
    """Detector + embedder. Lazy-loaded, CPU only, and refuses to touch the network."""

    def __init__(self, cfg: DetectConfig = DetectConfig(), model_root: str | Path = DEFAULT_MODEL_ROOT):
        self.cfg = cfg
        self.model_root = Path(model_root)
        self._app = None
        self._det_size: int | None = None

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self._app is not None:
            return
        pack_dir = self.model_root / "models" / self.cfg.model_pack
        if not pack_dir.is_dir() or not any(pack_dir.glob("*.onnx")):
            raise ModelsNotFoundError(
                f"Model files not found in {pack_dir}. Run once (needs internet, install time only):\n"
                f"    python scripts/download_models.py"
            )
        from insightface.app import FaceAnalysis  # imported here so tests run without models

        app = FaceAnalysis(
            name=self.cfg.model_pack,
            root=str(self.model_root),
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        app.prepare(ctx_id=-1, det_size=(self.cfg.det_size, self.cfg.det_size), det_thresh=self.cfg.det_thresh)
        self._app = app
        self._det_size = self.cfg.det_size

    def _set_det_size(self, size: int) -> None:
        if size != self._det_size:
            self._app.det_model.prepare(ctx_id=-1, input_size=(size, size), det_thresh=self.cfg.det_thresh)
            self._det_size = size

    @property
    def cache_tag(self) -> str:
        return self.cfg.cache_tag

    # ---------------------------------------------------------------- detection
    def _detect_once(self, bgr: np.ndarray) -> list[DetectedFace]:
        out = []
        for f in self._app.get(bgr):
            face = DetectedFace(
                bbox=np.asarray(f.bbox, dtype=np.float32),
                det_score=float(f.det_score),
                embedding=np.asarray(f.normed_embedding, dtype=np.float32),
                kps=None if getattr(f, "kps", None) is None else np.asarray(f.kps, dtype=np.float32),
            )
            if min(face.width, face.height) >= self.cfg.min_face_px:
                out.append(face)
        return out

    def detect(self, bgr: np.ndarray, det_size: int | None = None, tiling: bool | None = None) -> list[DetectedFace]:
        """Find all faces and embed them. Returned boxes are in `bgr` pixel coordinates."""
        self._load()
        size = det_size or self.cfg.det_size
        use_tiling = self.cfg.tiling if tiling is None else tiling
        self._set_det_size(size)

        faces = self._detect_once(bgr)
        h, w = bgr.shape[:2]
        if use_tiling and max(h, w) > size * 1.25:
            for x0, y0, x1, y1 in tile_boxes(w, h, size, self.cfg.tile_overlap):
                for f in self._detect_once(np.ascontiguousarray(bgr[y0:y1, x0:x1])):
                    b = f.bbox
                    # A face cut by an inner tile edge gives a poor embedding: skip it.
                    cut = (
                        (b[0] < 3 and x0 > 0) or (b[1] < 3 and y0 > 0)
                        or (b[2] > (x1 - x0) - 3 and x1 < w) or (b[3] > (y1 - y0) - 3 and y1 < h)
                    )
                    if cut:
                        continue
                    f.bbox = b + np.array([x0, y0, x0, y0], dtype=np.float32)
                    if f.kps is not None:
                        f.kps = f.kps + np.array([x0, y0], dtype=np.float32)
                    faces.append(f)
            faces = nms(faces, self.cfg.nms_iou)
        return faces
