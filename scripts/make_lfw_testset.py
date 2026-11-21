"""Build a reproducible identity test set from LFW (Labeled Faces in the Wild).

    python scripts/make_lfw_testset.py --download --out testset_lfw      # needs internet, dev time only
    python scripts/make_lfw_testset.py --lfw path/to/lfw_funneled --out testset_lfw

LFW layout expected: <root>/<Person_Name>/<Person_Name_0001.jpg>

Output:
    testset_lfw/gallery/            one photo per person, 'Person Name.jpg'   (the "folder of 50")
    testset_lfw/probes/known/       a DIFFERENT photo of a gallery person
    testset_lfw/probes/unknown/     photos of people who are NOT in the gallery
    testset_lfw/probes.json         {"known/xyz.jpg": "Person Name", "unknown/abc.jpg": null}
Same --seed gives the same set, so results in your docs can be reproduced.
Do not commit the images (size + dataset licence): the script and the seed are the evidence.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png"}


def find_lfw_root(download: bool, lfw: str | None) -> Path:
    if lfw:
        return Path(lfw)
    if not download:
        raise SystemExit("Give --lfw <folder> or --download")
    from sklearn.datasets import fetch_lfw_people  # downloads once into ~/scikit_learn_data

    fetch_lfw_people(min_faces_per_person=2, download_if_missing=True)
    root = Path.home() / "scikit_learn_data" / "lfw_home" / "lfw_funneled"
    if not root.is_dir():
        raise SystemExit(f"Download finished but {root} not found. Locate 'lfw_funneled' and pass it with --lfw.")
    return root


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lfw")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="testset_lfw")
    ap.add_argument("--people", type=int, default=50, help="gallery size")
    ap.add_argument("--unknown", type=int, default=50, help="how many strangers to use as unknown probes")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    root = find_lfw_root(a.download, a.lfw)
    people = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        imgs = sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        if imgs:
            people[d.name] = imgs
    multi = [n for n, imgs in people.items() if len(imgs) >= 2]
    if len(multi) < a.people:
        raise SystemExit(f"Only {len(multi)} people with 2+ photos, need {a.people}")

    rng = random.Random(a.seed)
    gallery_people = rng.sample(multi, a.people)
    others = [n for n in people if n not in set(gallery_people)]
    unknown_people = rng.sample(others, min(a.unknown, len(others)))

    out = Path(a.out)
    shutil.rmtree(out, ignore_errors=True)
    for sub in ("gallery", "probes/known", "probes/unknown"):
        (out / sub).mkdir(parents=True)

    labels = {}
    for n in gallery_people:
        imgs = people[n]
        ref, probe = rng.sample(imgs, 2)
        pretty = n.replace("_", " ")
        shutil.copy(ref, out / "gallery" / f"{pretty}.jpg")
        fname = f"{n}__{probe.name}"
        shutil.copy(probe, out / "probes" / "known" / fname)
        labels[f"known/{fname}"] = pretty
    for n in unknown_people:
        img = rng.choice(people[n])
        fname = f"{n}__{img.name}"
        shutil.copy(img, out / "probes" / "unknown" / fname)
        labels[f"unknown/{fname}"] = None

    (out / "probes.json").write_text(json.dumps(labels, indent=1))
    print(f"gallery: {len(gallery_people)} people | known probes: {len(gallery_people)} | unknown probes: {len(unknown_people)}")
    print(f"written to {out}/  (seed={a.seed})")


if __name__ == "__main__":
    main()