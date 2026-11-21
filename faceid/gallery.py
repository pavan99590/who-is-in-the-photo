from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import io
import math

import numpy as np
from PIL import Image, ImageOps

from .config import DEFAULT_CACHE_DIR, IMAGE_EXTS
from .engine import FaceEngine
from .imageio import ImageLoadError, load_image


def name_from_filename(filename: str) -> str:
    """'Priya_Sharma.jpg' -> 'Priya Sharma'."""
    stem = Path(filename).stem
    name = re.sub(r"\s+", " ", stem.replace("_", " ")).strip()
    return name or stem


def is_candidate_file(filename: str) -> bool:
    base = Path(filename).name
    return (not base.startswith(".")) and Path(base).suffix.lower() in IMAGE_EXTS


@dataclass
class EnrollRecord:
    filename: str
    name: str
    status: str                       # ok | multiple_faces | no_face | unreadable
    n_faces: int = 0
    det_score: float | None = None
    embedding: np.ndarray | None = None
    message: str = ""
    thumb: bytes | None = None        # small JPEG of the face that was used, so the UI can show it

    @property
    def usable(self) -> bool:
        return self.embedding is not None


@dataclass
class Gallery:
    names: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    embeddings: np.ndarray = field(default_factory=lambda: np.zeros((0, 512), dtype=np.float32))
    records: list[EnrollRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.names)

    @property
    def warnings(self) -> list[str]:
        return [f"{r.filename}: {r.message}" for r in self.records if r.message]


def pick_reference_face(faces, width: int, height: int):
    """Choose which face in a reference photo is 'the person'.

    Plain 'largest face' failed on a real case: a photo with 3 people where a half-cropped
    neighbour at the edge was larger than the centred subject. Portraits are centred, so
    we weight face area by closeness to the image centre (a gaussian on the distance,
    as a fraction of the image diagonal).
    """
    diag = math.hypot(width, height)

    def score(f) -> float:
        cx, cy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
        d = math.hypot(cx - width / 2, cy - height / 2) / diag
        return f.area * math.exp(-((d / 0.2) ** 2))

    return max(faces, key=score)


def make_thumb(bgr: np.ndarray, face, size: int = 96) -> bytes | None:
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = face.bbox
    side = max(x2 - x1, y2 - y1) * 1.6
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    box = (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)), int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))
    if box[2] - box[0] < 4 or box[3] - box[1] < 4:
        return None
    crop = Image.fromarray(bgr[:, :, ::-1]).crop(box)
    buf = io.BytesIO()
    ImageOps.fit(crop, (size, size)).save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _cache_key(data: bytes, tag: str) -> str:
    return hashlib.sha1(data + tag.encode()).hexdigest()


def enroll_items(
    items: Iterable[tuple[str, bytes]],
    engine: FaceEngine,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    progress: Callable[[int, int, str], None] | None = None,
) -> Gallery:
    """Build a gallery from (filename, file bytes) pairs.

    Works for a folder on disk and for files uploaded through the browser alike.
    One embedding per file, cached by content hash so re-opening a folder is instant.
    """
    files = sorted((i for i in items if is_candidate_file(i[0])), key=lambda i: Path(i[0]).name.lower())
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    records: list[EnrollRecord] = []
    used: dict[str, int] = {}
    for idx, (filename, data) in enumerate(files):
        if progress:
            progress(idx, len(files), filename)
        base_name = name_from_filename(Path(filename).name)
        used[base_name] = used.get(base_name, 0) + 1
        name = base_name if used[base_name] == 1 else f"{base_name} ({used[base_name]})"
        dup_msg = "" if used[base_name] == 1 else f"duplicate name, listed as '{name}'. "
        rec = EnrollRecord(filename=filename, name=name, status="ok", message=dup_msg.strip())

        key = _cache_key(data, engine.cache_tag)
        cache_file = cache / f"{key}.npz" if cache else None
        if cache_file is not None and cache_file.exists():
            with np.load(cache_file) as z:
                rec.embedding = z["emb"].astype(np.float32)
                rec.det_score = float(z["det_score"])
                rec.n_faces = int(z["n_faces"])
                rec.thumb = z["thumb"].tobytes() if "thumb" in z.files and z["thumb"].size else None
        else:
            try:
                img = load_image(data, max_side=engine.cfg.gallery_max_side)
            except ImageLoadError as exc:
                rec.status, rec.message = "unreadable", (dup_msg + f"not a readable image ({exc})").strip()
                records.append(rec)
                continue
            faces = engine.detect(img.bgr, det_size=engine.cfg.gallery_det_size, tiling=False)
            rec.n_faces = len(faces)
            if not faces:
                rec.status, rec.message = "no_face", (dup_msg + "no face found, this person will not be recognised").strip()
                records.append(rec)
                continue
            best = pick_reference_face(faces, img.bgr.shape[1], img.bgr.shape[0])
            rec.embedding, rec.det_score = best.embedding, best.det_score
            rec.thumb = make_thumb(img.bgr, best)
            if cache_file is not None:
                np.savez(cache_file, emb=best.embedding, det_score=best.det_score, n_faces=len(faces),
                         thumb=np.frombuffer(rec.thumb or b"", dtype=np.uint8))

        if rec.n_faces > 1:
            rec.status = "multiple_faces"
            rec.message = (dup_msg + f"{rec.n_faces} faces in this photo, used the one nearest the centre").strip()
        records.append(rec)

    usable = [r for r in records if r.usable]
    emb = np.stack([r.embedding for r in usable]).astype(np.float32) if usable else np.zeros((0, 512), np.float32)
    if progress:
        progress(len(files), len(files), "")
    return Gallery(names=[r.name for r in usable], files=[r.filename for r in usable], embeddings=emb, records=records)


def enroll_folder(
    folder: str | Path,
    engine: FaceEngine,
    cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
    progress: Callable[[int, int, str], None] | None = None,
) -> Gallery:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a folder: {folder}")
    items = [(p.name, p.read_bytes()) for p in folder.iterdir() if p.is_file() and is_candidate_file(p.name)]
    return enroll_items(items, engine, cache_dir=cache_dir, progress=progress)