"""Measure accuracy and pick the unknown-threshold from DATA instead of guessing.

Test-set layout:
    testset/
      gallery/            one photo per person, 'First Last.jpg'
      groups/             group photos
      labels.json         {"group1.jpg": {"present": ["Priya Sharma", ...], "n_unknown": 3}, ...}
                          'present' = people in the photo who ARE in the gallery.
                          'n_unknown' (optional) = people in the photo who are NOT in the gallery.

    python scripts/evaluate.py testset --out outputs/eval
Detection (slow) runs once per photo; the threshold/margin sweep then reuses the cached embeddings.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from faceid import DetectConfig, FaceEngine, MatchConfig, Recognizer, enroll_folder  # noqa: E402
from faceid.matcher import match_faces  # noqa: E402


def score_threshold(cached, gallery, labels, cfg):
    tp = fp = fn = 0            # at person level: named correctly / named wrongly / missed
    unknown_total = unknown_ok = 0
    for fname, (img, faces) in cached.items():
        truth = set(labels[fname]["present"])
        matches = match_faces([f.embedding for f in faces], gallery.embeddings, gallery.names, cfg) if faces else []
        pred = [m.name for m in matches if m.status == "known"]
        tp += len(set(pred) & truth)
        fp += sum(1 for n in pred if n not in truth)        # a stranger given a name, or the wrong name
        fn += len(truth - set(pred))
        if "n_unknown" in labels[fname]:
            n_unk = labels[fname]["n_unknown"]
            unknown_total += n_unk
            wrongly_named = sum(1 for n in pred if n not in truth)
            unknown_ok += max(0, n_unk - wrongly_named)
    prec = tp / (tp + fp) if tp + fp else 1.0
    rec = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    unk_rej = unknown_ok / unknown_total if unknown_total else float("nan")
    return dict(threshold=cfg.threshold, min_margin=cfg.min_margin, tp=tp, fp=fp, fn=fn,
                precision=round(prec, 4), recall=round(rec, 4), f1=round(f1, 4),
                unknown_rejection=round(unk_rej, 4) if unknown_total else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--tiling", action="store_true")
    ap.add_argument("--det-size", type=int, default=DetectConfig().det_size)
    a = ap.parse_args()

    root = Path(a.testset)
    labels = json.loads((root / "labels.json").read_text())
    engine = FaceEngine(DetectConfig(det_size=a.det_size, tiling=a.tiling))
    rec = Recognizer(engine)
    gallery = enroll_folder(root / "gallery", engine)
    print(f"gallery: {len(gallery)} people; warnings: {len(gallery.warnings)}")

    cached, expected_faces, found_faces = {}, 0, 0
    for fname in labels:
        img, faces = rec.detect(root / "groups" / fname)
        cached[fname] = (img, faces)
        if "n_unknown" in labels[fname]:
            expected_faces += len(labels[fname]["present"]) + labels[fname]["n_unknown"]
            found_faces += len(faces)
        print(f"  {fname}: {len(faces)} faces detected")
    if expected_faces:
        print(f"detection: {found_faces} found vs {expected_faces} expected")

    rows = []
    for margin in (0.0, 0.02, 0.05, 0.08):
        for thr in np.arange(0.20, 0.71, 0.02):
            rows.append(score_threshold(cached, gallery, labels, MatchConfig(float(round(thr, 2)), margin)))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Prefer zero false accepts (naming a stranger is the worst error here), then best recall.
    key = lambda r: (r["fp"], -r["recall"], -r["f1"])  # noqa: E731
    top = min(key(r) for r in rows)
    plateau = sorted((r for r in rows if key(r) == top and r["min_margin"] == 0.0), key=lambda r: r["threshold"])
    # Many thresholds tie on a small test set: take the MIDDLE of the plateau, not its edge,
    # because the middle is the most forgiving on unseen data.
    best = plateau[len(plateau) // 2] if plateau else sorted(rows, key=key)[0]
    print("\nbest (fewest false names, then recall):", best)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        m0 = [r for r in rows if r["min_margin"] == 0.0]
        plt.figure(figsize=(7, 4))
        for key in ("precision", "recall", "f1"):
            plt.plot([r["threshold"] for r in m0], [r[key] for r in m0], label=key)
        plt.axvline(best["threshold"], ls="--", c="grey")
        plt.xlabel("similarity threshold"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(out / "threshold_sweep.png", dpi=150)
        print(f"plot -> {out / 'threshold_sweep.png'}")
    except ImportError:
        print("(install matplotlib to also get a plot)")
    print(f"table -> {out / 'sweep.csv'}")


if __name__ == "__main__":
    main()
