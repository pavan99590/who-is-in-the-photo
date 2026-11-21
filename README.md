# Who Is in the Photo? (offline face identification)

Work in progress. Core pipeline is done; web UI, full docs and test data come next.

## Setup (internet needed ONCE, at install time)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_models.py --prune              # ~190 MB, cached in ./models
```
After this the app runs with no internet. The engine refuses to download anything at runtime.

## Try the pipeline from the command line
```bash
python -m faceid --gallery path/to/people_folder --photo group.jpg --out outputs/annotated.jpg
```
Gallery files are named after the person: `Priya Sharma.jpg` (underscores become spaces).

## How it decides
1. SCRFD detector finds every face in the group photo (1280 px detector input for small faces).
2. ArcFace (w600k) turns each face into a 512-d embedding; gallery embeddings are cached by file hash.
3. Cosine similarity between every face and every person, then Hungarian assignment so one person
   can only be claimed by one face.
4. A face gets a name only if its similarity clears the threshold (and margin); otherwise "Unknown".
   Calibrate the threshold on your own data: `python scripts/evaluate.py testset/`.

## Tests
```bash
python -m pytest -q        # no model files needed
```
