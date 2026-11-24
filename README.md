# Who Is in the Photo? Offline face identification

Open a folder with one photo per person, upload a group photo, and see who is who.
Faces of people who are not in the folder are marked **Unknown**. Everything runs on your own
laptop, on CPU, with **no internet at runtime** and no online APIs.

- Names appear as boxes on the photo and as a list next to it.
- The page loads no external fonts, scripts or images, and the server only listens on `127.0.0.1`.
- Task A of the Industry.AI take-home. Related docs: [AI_COLLAB.md](AI_COLLAB.md),
  [RESEARCH.md](RESEARCH.md), [ARCHITECTURE.md](ARCHITECTURE.md), test evidence in [docs/evidence/](docs/evidence/).
- [Full AI collaboration chat (Claude share)](https://claude.ai/share/4868ca07-8f57-473b-a47a-29b41d5b8825).

## Setup

Needs Python 3.10 to 3.12. Tested on Windows (Python 3.11) and Linux (Python 3.12). No GPU needed.
Internet is needed **once**, for `pip` and the model download. After that the app is fully offline.

```bash
git clone https://github.com/pavan99590/who-is-in-the-photo.git
cd who-is-in-the-photo

python -m venv .venv
.venv\Scripts\activate            # Windows (PowerShell). Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py --prune     # about 190 MB into ./models
python -m pytest -q                           # optional check: 18 tests pass, no models needed
```

If `pip install insightface` fails on Windows with "Microsoft Visual C++ 14.0 or greater is required",
install "Build Tools for Visual Studio" (C++ workload) and run the install again.

## Run

```bash
python run.py
```

This starts the app and opens `http://127.0.0.1:8000`. Options: `--port 8001` if the port is taken,
`--no-browser` to skip opening the browser. Stop it with Ctrl+C.
The chip at the top right shows **Ready · works offline** once the models are loaded.

1. **Open the folder of people.** Click *Choose folder*, pick the folder, and wait for "N people ready".
   Reference photos that need a look (several faces, no face, unreadable) are listed with the face that was used.
2. **Add the group photo.** Drop it on the page or click the box. Results appear in a few seconds.

Use a Chromium-based browser (Chrome or Edge). The folder picker uses the browser's directory upload.
All photos stay on this computer. Face fingerprints and small thumbnails are cached in `.cache/`
so reopening a folder is instant. Delete that folder to clear them.

## Gallery format (the folder of people)

- **One photo per person.** The file name is the person's name: `Priya Sharma.jpg` or `Priya_Sharma.jpg`
  both show as "Priya Sharma".
- Best reference photo: one clear, front-facing face, the person alone.
- Formats: jpg, jpeg, png, webp, bmp, tif/tiff, heic/heif. Hidden files and non-images are ignored.
- Two files with the same name: the second is listed as "Name (2)".
- Several faces in a reference photo: the face nearest the centre (weighted by size) is used, with a
  warning and a thumbnail so you can check it.
- No face found, or an unreadable file: that person is listed with a warning and cannot be recognised.
- Windows hides file extensions, so a file can end up as `name.jpg.jpg`, which shows as "name.jpg".
  Turn on *View > Show > File name extensions* in File Explorer to check.

Group photos can be any common image. Phone photos are rotated using their EXIF data, and photos larger
than 3200 px on the long side are scaled down for detection.

## Reading the output

- **Green box + name:** recognised. **Red box + "Unknown":** not matched to anyone in the folder.
- **Number on each face** (left to right) matches the row in the results list. Hover a face or a row to highlight both.
- **Dashed box and "?":** small or unclear face. Treat that label with care.
- **Crowded photos** (more than 14 faces, or small faces): the photo shows numbers only, and names appear on
  hover. Tick *Show names on the photo* to show them anyway.
- **Results list:** the name or "Unknown", and a **similarity** score (cosine similarity of face embeddings,
  higher is more similar). For Unknown faces, "Closest in folder: X (0.31)" is only information, **not a name**.
- **Summary:** how many faces were recognised, how many Unknown, faces found and time taken.
- **In the folder but not found in this photo:** gallery people who matched no face.
- **How strict:** the slider (0.20 to 0.70, default 0.40) re-labels instantly without re-detecting.
  Higher means fewer wrong names and more Unknown. Lower means more names and more risk of a wrong one.
- **Download marked photo** saves the photo with boxes and names.

**How a name is given.** Each face is compared with every person in the folder. A face gets a name only if the
best match scores at least the threshold (0.40) and beats every other person, and each person is assigned to at
most one face per photo. Everything else is Unknown. In the LFW calibration test, same-person scores had a median
of 0.70 (5th percentile 0.56) and strangers a median of 0.13 (95th percentile 0.19).

## Command line (optional)

```bash
python -m faceid --gallery path/to/people --photo group.jpg --out outputs/annotated.jpg
```
Other options: `--json`, `--threshold 0.4`, `--tiling` (extra pass for very large photos).

## Results so far

Details and limits are in [docs/evidence/group_test_results.md](docs/evidence/group_test_results.md).

- LFW identity test (50 gallery, 50 known and 50 unknown probes, seed 42): 49 of 50 known people named
  correctly, 0 wrong names, no stranger named at 0.40. The one miss looks like a mislabelled dataset pair.
- Synthetic group collages: 26 of 27 present people named, 0 wrong names, no stranger named, 0 faces found in a no-face photo.
- One real 35-face group photo: all 35 faces found, the one person present in the gallery was named
  (similarity 0.80), the other 34 faces stayed Unknown.

These are counts on small test sets, not an accuracy estimate.

Rebuild the test data (not in the repo, for privacy and dataset licence reasons):
```bash
python scripts/make_lfw_testset.py --download --out testset_lfw
python scripts/evaluate_probes.py testset_lfw
python scripts/make_group_testset.py --src testset_lfw --out testset_collage
python scripts/evaluate.py testset_collage
```

## Known limitations

- **Threshold tuned on LFW**, which is mostly public figures, many of them Western. Recognition accuracy and
  false-match rates can differ across groups of people, so 0.40 may not be right for other people.
  The real-photo evidence is a single photo with a single present person.
- **Reference photo quality matters.** Strong head turns, sunglasses, masks, heavy blur or a large age gap
  can produce a miss (shown as Unknown). A wrong face choice in a multi-face reference photo is possible.
- **Lookalikes** (siblings, twins) can be confused, and an unknown person who closely resembles someone in the
  folder may be named if the score reaches the threshold.
- **A person appearing twice** in one photo gets the name on one face only. The other shows Unknown.
- **Small faces:** faces under 20 px are ignored and under 40 px are flagged. Crowds are slower:
  about 3 s for 10 faces and 6 to 9 s for 35 faces on a laptop CPU.
- **Single local user:** the app keeps one gallery in memory and is not meant to be served to several people.
- An empty folder gives no feedback in some browsers, because the browser sends no files.
- Face recognition is probabilistic. Use it as an aid and check the result, not as the only basis for
  decisions about people.
- The InsightFace pretrained models (`buffalo_l`) are released for non-commercial research use. Check the licence
  before any other use.

## Troubleshooting

- **Chip says "Setup needed" or an error mentions model files:** run `python scripts/download_models.py --prune`
  once with internet, then restart.
- **Port in use:** `python run.py --port 8001`.
- **First folder load is slow:** the first run computes a fingerprint for every photo. Later runs use the cache.
- **Page looks old after an update:** press Ctrl+F5.

## Project layout

```
run.py                 start the app
faceid/                the program: engine (detect + embed), gallery, matcher, pipeline, annotate,
                       server (FastAPI), static/index.html (the page), cli
scripts/               download_models, make_lfw_testset, make_group_testset, evaluate_probes, evaluate
tests/                 18 automatic tests (matching, enrolment, caching, web API)
docs/evidence/         threshold plot and CSV, group test results
models/                downloaded models (not in Git)
```