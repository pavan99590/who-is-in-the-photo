from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from .config import DetectConfig, MatchConfig
from .engine import DetectedFace, FaceEngine
from .gallery import Gallery
from .imageio import LoadedImage, load_image
from .matcher import match_faces


@dataclass
class FaceResult:
    index: int                    # 1-based, left to right
    bbox: list[float]             # [x1, y1, x2, y2] in ORIGINAL image pixels
    det_score: float
    status: str                   # known | unknown
    name: str | None
    score: float
    margin: float
    candidate: str | None
    candidate_score: float
    low_quality: bool             # tiny or low-confidence face: treat the label with care

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IdentifyResult:
    width: int
    height: int
    faces: list[FaceResult] = field(default_factory=list)
    elapsed_s: float = 0.0
    threshold: float = 0.0

    @property
    def n_known(self) -> int:
        return sum(f.status == "known" for f in self.faces)

    @property
    def n_unknown(self) -> int:
        return len(self.faces) - self.n_known

    def to_dict(self) -> dict:
        return {
            "width": self.width, "height": self.height, "elapsed_s": round(self.elapsed_s, 3),
            "threshold": self.threshold, "n_faces": len(self.faces),
            "n_known": self.n_known, "n_unknown": self.n_unknown,
            "faces": [f.to_dict() for f in self.faces],
        }


class Recognizer:
    def __init__(self, engine: FaceEngine | None = None, match_cfg: MatchConfig = MatchConfig()):
        self.engine = engine or FaceEngine(DetectConfig())
        self.match_cfg = match_cfg

    def detect(self, source) -> tuple[LoadedImage, list[DetectedFace]]:
        """Stage 1 (slow): find and embed faces, left to right. Cache this when tuning thresholds."""
        img = load_image(source, max_side=self.engine.cfg.max_side)
        faces = self.engine.detect(img.bgr)
        faces.sort(key=lambda f: (f.bbox[0] + f.bbox[2]) / 2)
        return img, faces

    def decide(self, img: LoadedImage, faces: list[DetectedFace], gallery: Gallery,
               match_cfg: MatchConfig | None = None) -> IdentifyResult:
        """Stage 2 (instant): turn embeddings into names."""
        cfg = match_cfg or self.match_cfg
        w, h = img.original_size
        matches = match_faces(
            [f.embedding for f in faces] if faces else [],
            gallery.embeddings, gallery.names, cfg,
        ) if faces else []
        out = []
        for i, (f, m) in enumerate(zip(faces, matches), start=1):
            x1, y1, x2, y2 = (f.bbox / img.scale).tolist()
            out.append(FaceResult(
                index=i,
                bbox=[max(0.0, x1), max(0.0, y1), min(float(w), x2), min(float(h), y2)],
                det_score=f.det_score, status=m.status, name=m.name,
                score=round(m.score, 4), margin=round(m.margin, 4),
                candidate=m.candidate, candidate_score=round(m.candidate_score, 4),
                low_quality=min(f.width, f.height) < 40 or f.det_score < 0.6,
            ))
        return IdentifyResult(width=w, height=h, faces=out, threshold=cfg.threshold)

    def identify(self, source, gallery: Gallery, match_cfg: MatchConfig | None = None) -> IdentifyResult:
        t0 = time.perf_counter()
        img, faces = self.detect(source)
        result = self.decide(img, faces, gallery, match_cfg)
        result.elapsed_s = time.perf_counter() - t0
        return result
