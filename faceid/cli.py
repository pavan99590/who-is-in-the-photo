from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .annotate import annotate
from .config import DetectConfig, MatchConfig
from .engine import FaceEngine, ModelsNotFoundError
from .gallery import enroll_folder
from .pipeline import Recognizer


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="faceid", description="Offline: who is in this group photo?")
    p.add_argument("--gallery", required=True, help="folder with one photo per person, named 'First Last.jpg'")
    p.add_argument("--photo", help="group photo to identify (omit to only check the gallery)")
    p.add_argument("--out", help="write the annotated photo here")
    p.add_argument("--json", action="store_true", help="print machine-readable results")
    p.add_argument("--threshold", type=float, default=MatchConfig().threshold)
    p.add_argument("--min-margin", type=float, default=MatchConfig().min_margin)
    p.add_argument("--det-size", type=int, default=DetectConfig().det_size)
    p.add_argument("--tiling", action="store_true", help="extra tiled pass, helps on very large photos")
    p.add_argument("--models", help="model root folder (default: ./models)")
    a = p.parse_args(argv)

    cfg = DetectConfig(det_size=a.det_size, tiling=a.tiling)
    engine = FaceEngine(cfg, model_root=a.models) if a.models else FaceEngine(cfg)
    rec = Recognizer(engine, MatchConfig(threshold=a.threshold, min_margin=a.min_margin))

    try:
        gallery = enroll_folder(a.gallery, engine, progress=lambda i, n, f: print(f"\renrolling {i}/{n} {f[:40]:40}", end="", file=sys.stderr))
        print(file=sys.stderr)
        for w in gallery.warnings:
            print(f"warning: {w}", file=sys.stderr)
        print(f"{len(gallery)} people enrolled from {len(gallery.records)} files", file=sys.stderr)
        if not a.photo:
            return 0
        result = rec.identify(a.photo, gallery)
    except ModelsNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"{len(result.faces)} faces: {result.n_known} known, {result.n_unknown} unknown ({result.elapsed_s:.1f}s)")
        for f in result.faces:
            who = f.name if f.status == "known" else f"Unknown (closest: {f.candidate} {f.candidate_score:.2f})"
            flag = "  [low quality]" if f.low_quality else ""
            print(f"  {f.index:>2}. {who:<48} score={f.score:.2f} margin={f.margin:+.2f}{flag}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        annotate(a.photo, result).save(a.out, quality=92)
        print(f"annotated photo -> {a.out}", file=sys.stderr)
    return 0
