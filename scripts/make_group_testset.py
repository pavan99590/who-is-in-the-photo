"""Build quick test "group photos" (collages) plus labels.json from the LFW test set.

    python scripts/make_group_testset.py --src testset_lfw --out testset_collage
    python scripts/evaluate.py testset_collage --out outputs/eval_collage

Reads  <src>/gallery/*.jpg, <src>/probes/known|unknown/*.jpg, <src>/probes.json (from make_lfw_testset.py).
Writes <out>/gallery/        copy of the gallery
       <out>/groups/*.jpg    collages: known people (a DIFFERENT photo than the gallery one) + strangers
       <out>/labels.json     {"photo.jpg": {"present": [...names...], "n_unknown": N}}   <- what evaluate.py reads
Collages are synthetic. They test the pipeline and the unknown logic, not real group-photo conditions.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

# (photo name, fraction of the faces that are gallery people). The rest are strangers.
RECIPES = [("easy_all_known", 1.0), ("mixed_a", 0.6), ("mixed_b", 0.3), ("mixed_c", 0.8), ("none_known", 0.0)]


def face_tile(path: Path, rng: random.Random, base: int) -> Image.Image:
    """Centre crop (LFW faces are centred; this drops most background people), random size."""
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side, cx, cy = int(min(w, h) * 0.6), w // 2, h // 2
    crop = img.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))
    size = int(base * rng.uniform(0.8, 1.2))
    return crop.resize((size, size), Image.LANCZOS)


def collage(paths: list[Path], rng: random.Random, base: int = 150, cols: int = 5) -> Image.Image:
    cell = int(base * 1.35)
    rows = max(1, -(-len(paths) // cols))
    canvas = Image.new("RGB", (cols * cell, rows * cell), tuple(rng.randint(70, 140) for _ in range(3)))
    for i, p in enumerate(paths):
        t = face_tile(p, rng, base)
        r, c = divmod(i, cols)
        canvas.paste(t, (c * cell + rng.randint(0, cell - t.width), r * cell + rng.randint(0, cell - t.height)))
    return canvas


def no_face_photo(rng: random.Random) -> Image.Image:
    """A scenery-like image with no faces: sky gradient, ground, a few shapes."""
    w, h = 1000, 600
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        d.line([(0, y), (w, y)], fill=(90 + y // 6, 140 + y // 8, 220 - y // 5))
    d.rectangle([0, int(h * 0.7), w, h], fill=(60, 120, 60))
    for _ in range(6):
        x, y = rng.randint(0, w - 120), rng.randint(int(h * 0.5), int(h * 0.8))
        d.rectangle([x, y, x + rng.randint(40, 120), y + rng.randint(40, 120)], fill=tuple(rng.randint(80, 200) for _ in range(3)))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="testset_lfw")
    ap.add_argument("--out", default="testset_collage")
    ap.add_argument("--faces", type=int, default=10, help="faces per photo")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    if src.resolve() == out.resolve():
        raise SystemExit("--out must be a different folder from --src")
    probes = json.loads((src / "probes.json").read_text())
    known = [(rel, name) for rel, name in probes.items() if name]
    strangers = [rel for rel, name in probes.items() if not name]

    rng = random.Random(a.seed)
    shutil.rmtree(out, ignore_errors=True)
    (out / "groups").mkdir(parents=True)
    shutil.copytree(src / "gallery", out / "gallery")

    labels = {}
    print(f"{'photo':<18}{'faces':>6}{'in gallery':>12}{'strangers':>11}")
    for name, frac in RECIPES:
        k = min(round(frac * a.faces), len(known))
        u = min(a.faces - k, len(strangers))
        chosen_k, chosen_u = rng.sample(known, k), rng.sample(strangers, u)
        items = [(src / "probes" / rel) for rel, _ in chosen_k] + [(src / "probes" / rel) for rel in chosen_u]
        rng.shuffle(items)
        collage(items, rng).save(out / "groups" / f"{name}.jpg", quality=92)
        labels[f"{name}.jpg"] = {"present": [n for _, n in chosen_k], "n_unknown": u}
        print(f"{name + '.jpg':<18}{k + u:>6}{k:>12}{u:>11}")

    no_face_photo(rng).save(out / "groups" / "noface.jpg", quality=92)
    labels["noface.jpg"] = {"present": [], "n_unknown": 0}
    print(f"{'noface.jpg':<18}{0:>6}{0:>12}{0:>11}")

    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"\nwritten to {out}/ (seed={a.seed}). Next:\n  python scripts/evaluate.py {out} --out outputs/eval_collage")


if __name__ == "__main__":
    main()