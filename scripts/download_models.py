"""Download the face models ONCE, at install time. After this the app never touches the internet.

    python scripts/download_models.py            # buffalo_l (~280 MB download)
    python scripts/download_models.py --prune    # also delete the unused landmark/gender-age models
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from faceid.config import DEFAULT_MODEL_ROOT, DetectConfig  # noqa: E402

UNUSED = ("1k3d68.onnx", "2d106det.onnx", "genderage.onnx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default=DetectConfig().model_pack)
    ap.add_argument("--root", default=str(DEFAULT_MODEL_ROOT))
    ap.add_argument("--prune", action="store_true", help="remove models this app does not use")
    a = ap.parse_args()

    from insightface.app import FaceAnalysis

    FaceAnalysis(name=a.pack, root=a.root, providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    pack_dir = Path(a.root) / "models" / a.pack
    if a.prune:
        for n in UNUSED:
            (pack_dir / n).unlink(missing_ok=True)
        zip_file = pack_dir.with_suffix(".zip")
        zip_file.unlink(missing_ok=True)
    size = sum(p.stat().st_size for p in pack_dir.glob("*.onnx")) / 1e6
    print(f"Models ready in {pack_dir} ({size:.0f} MB). The app now runs fully offline.")


if __name__ == "__main__":
    main()
