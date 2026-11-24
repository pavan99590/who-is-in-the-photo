# Architecture

## Runtime diagram

The overview below was generated with Archify from the repository's current runtime code. It shows the local user journey, recognition core, local model/cache files, and marked-photo export.

![Who Is in the Photo local runtime architecture](./.archify/architecture-who-is-in-the-photo-20261124-203419/architecture.svg)

Open the [interactive Archify diagram](./.archify/architecture-who-is-in-the-photo-20261124-203419/architecture.html) or review its [editable Archify source](./.archify/architecture-who-is-in-the-photo-20261124-203419/candidate.json).

## Project folders

```text
who-is-in-the-photo/
├── run.py                    # Starts the local FastAPI server and browser
├── requirements.txt          # Python runtime and test dependencies
├── README.md                 # Install, run, usage, limitations, and results
├── AI_COLLAB.md              # AI collaboration decisions and chat evidence
├── RESEARCH.md               # Required research doc; absent from this checkout
├── ARCHITECTURE.md           # This document and generated diagram
├── faceid/                   # App code
│   ├── static/index.html     # Single-page browser UI
│   ├── server.py             # FastAPI routes and in-memory app state
│   ├── pipeline.py           # Detection and matching stages
│   ├── engine.py             # InsightFace face detection and embeddings
│   ├── gallery.py            # Reference-photo enrolment and embedding cache
│   ├── matcher.py            # Cosine similarity and one-to-one assignment
│   ├── imageio.py            # Image decoding, EXIF orientation, and resizing
│   ├── annotate.py           # Draws boxes and labels for image downloads
│   ├── config.py             # Model, cache, image, and threshold defaults
│   └── cli.py                # Optional command-line interface
├── scripts/                  # Model download and evaluation/test-data tools
├── tests/                    # Gallery, matcher, and web API tests
├── docs/evidence/             # Evaluation results and plots
├── models/                   # Downloaded ONNX files; not required in Git
├── .cache/embeddings/         # Generated local reference-embedding cache
├── testset_*/                 # Local evaluation fixtures, not runtime inputs
├── outputs/                  # Generated evaluation outputs and annotations
├── .agents/skills/archify/    # Diagram-authoring skill; not part of app runtime
├── skills-lock.json          # Pins the project-local Archify skill
└── .archify/                  # Editable diagram source, HTML, and SVG export
```

## Request flow

1. The user selects a folder in the browser. The UI sends its image files to `POST /api/gallery`.
2. FastAPI starts gallery enrollment in a background thread. The gallery derives names from filenames, detects a reference face, and caches embeddings locally.
3. The user uploads a group photo to `POST /api/identify`. The recognition pipeline detects faces and creates embeddings.
4. The matcher compares group-photo embeddings with the gallery, assigns each gallery identity to at most one face, and labels a face `Unknown` if the configured threshold is not met.
5. FastAPI returns the result data and the EXIF-corrected photo. The UI draws boxes and a results list; the user can download a JPEG with labels.
6. Moving the threshold slider calls `POST /api/rematch`. It reuses the last detection results and changes the labels without running face detection again.

## Runtime and state

- `run.py` binds the server to `127.0.0.1`; browser/API traffic stays on the same laptop.
- The InsightFace models are loaded from `models/` and use the CPU execution provider. Model download is a setup-time action, not an inference-time network call.
- The active gallery and latest group-photo result live in the FastAPI process memory. This is intentionally a single-local-user workflow, not a multi-user hosted service.
- Reference embeddings are cached as local `.npz` files under `.cache/embeddings/`. Delete `.cache/` to clear that cache.
- The browser UI is a local static HTML file served by FastAPI; the app does not need a database or hosted frontend.

## Trade-offs

- **Single-process local app:** This keeps installation and runtime simple and avoids hosting or remote photo transfer. The trade-off is one active gallery/latest photo and no multi-user isolation.
- **One reference image per person:** This makes enrollment easy and matches the required folder format. It gives less coverage of pose, lighting, age, or appearance changes than a multi-image gallery.
- **Thresholded open-set matching:** Conservative `Unknown` results reduce unsupported names, while a strict threshold can miss real gallery members. The slider lets the user inspect this trade-off, but the decision remains probabilistic.
- **One-to-one assignment:** A person can be named on at most one face per photo. This avoids duplicate assignments but means repeated appearances of the same person are not all named.
- **Local model and cache files:** Runtime remains offline and avoids a database, at the cost of a one-time model download, local disk use, and model-license obligations.
- **Small evaluation sets:** LFW, synthetic collages, and the limited real-photo test are useful checks, not a guarantee of accuracy on the unseen demo photo or all groups of people.
