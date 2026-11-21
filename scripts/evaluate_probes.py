"""Identity-level evaluation: pick the unknown-threshold from data.

    python scripts/evaluate_probes.py testset_lfw --out outputs/eval_probes

For every probe photo (one face each) we find the closest gallery person, then sweep the
similarity threshold and report, per threshold:
  TAR     known probes named correctly                 (higher is better)
  misid   known probes named as the WRONG person       (lower is better)
  FAR     unknown probes that wrongly got a name       (lower is better)
Recommended threshold = the lowest one where wrong names (misid + FAR) stay under --max-false-rate.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from faceid import DetectConfig, FaceEngine, enroll_folder, enroll_items  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("testset")
    ap.add_argument("--out", default="outputs/eval_probes")
    ap.add_argument("--max-false-rate", type=float, default=0.01)
    ap.add_argument("--safety", type=float, default=0.03, help="added to the lowest passing threshold")
    ap.add_argument("--show", type=float, default=0.40,
                    help="list known probes scoring below this (or matched wrongly) and strangers scoring above 0.75x this")
    a = ap.parse_args()

    root = Path(a.testset)
    labels = json.loads((root / "probes.json").read_text())
    engine = FaceEngine(DetectConfig())
    gallery = enroll_folder(root / "gallery", engine)
    print(f"gallery: {len(gallery)} people ({len(gallery.warnings)} warnings)")
    for w in gallery.warnings:
        print("  warning:", w)

    items = [(rel, (root / "probes" / rel).read_bytes()) for rel in labels]
    probes = enroll_items(items, engine)           # same single-face embedding path as the gallery
    emb = {r.filename: r.embedding for r in probes.records if r.usable}
    missed = [rel for rel in labels if rel not in emb]
    print(f"probes: {len(emb)} embedded, {len(missed)} with no detectable face (counted as failures)")

    G = gallery.embeddings
    rows_data = []   # (truth_name or None, best_name, best_score)
    for rel, truth in labels.items():
        if rel not in emb:
            continue
        s = G @ emb[rel]
        j = int(np.argmax(s))
        rows_data.append((truth, gallery.names[j], float(s[j])))
        if truth is not None and (gallery.names[j] != truth or s[j] < a.show):
            print(f"  weak known probe: {rel} -> best match '{gallery.names[j]}' score {s[j]:.2f}")
        if truth is None and s[j] >= 0.75 * a.show:
            print(f"  close stranger:  {rel} -> looks like '{gallery.names[j]}' score {s[j]:.2f}")
    # Probes where no face was found count against us: they stay in the denominators.
    n_known = sum(1 for t in labels.values() if t is not None)
    n_unknown = sum(1 for t in labels.values() if t is None)

    rows = []
    for thr in np.arange(0.10, 0.81, 0.01):
        correct = sum(1 for t, b, s in rows_data if t is not None and s >= thr and b == t)
        misid = sum(1 for t, b, s in rows_data if t is not None and s >= thr and b != t)
        fa = sum(1 for t, b, s in rows_data if t is None and s >= thr)
        rows.append(dict(
            threshold=round(float(thr), 2),
            TAR=round(correct / n_known, 4) if n_known else "",
            misid=round(misid / n_known, 4) if n_known else "",
            FAR=round(fa / n_unknown, 4) if n_unknown else "",
            false_name_rate=round((misid + fa) / max(1, len(labels)), 4),
        ))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "probe_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["false_name_rate"] <= a.max_false_rate]
    best = ok[0] if ok else rows[-1]
    print(f"\nknown probes: {n_known} | unknown probes: {n_unknown}")
    print(f"lowest threshold with false-name rate <= {a.max_false_rate:.0%}: {best}")
    print(f"suggested --threshold: {best['threshold'] + a.safety:.2f}  (that + {a.safety} safety margin for unseen photos)")
    if n_known and n_unknown:
        same = [s for t, b, s in rows_data if t is not None and b == t]
        diff = [s for t, b, s in rows_data if t is None]
        print(f"same-person scores   median={np.median(same):.2f}  5th pct={np.percentile(same, 5):.2f}")
        print(f"stranger scores      median={np.median(diff):.2f}  95th pct={np.percentile(diff, 95):.2f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        for key in ("TAR", "misid", "FAR"):
            plt.plot([r["threshold"] for r in rows], [r[key] for r in rows], label=key)
        plt.axvline(best["threshold"], ls="--", c="grey")
        plt.xlabel("similarity threshold"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(out / "probe_sweep.png", dpi=150)
        print(f"plot -> {out / 'probe_sweep.png'}")
    except ImportError:
        print("(pip install matplotlib to also get a plot)")
    print(f"table -> {out / 'probe_sweep.csv'}")


if __name__ == "__main__":
    main()