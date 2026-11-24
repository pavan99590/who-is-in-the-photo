# AI Collaboration

## Scope and evidence

This is a retrospective draft based on the implemented code, project documentation, test evidence, and the AI conversation available in this VS Code session. The earlier conversations from the original implementation were not available to the assistant.

The "AI suggestion" entries below are retrospective alternatives, not quotations from those missing conversations. The "I chose" wording is a draft inferred from the implementation; edit it if it does not reflect your actual reasoning. The excerpts are literal excerpts from the conversation available here. Do not present this document as the complete project chat archive until you add the earlier chat exports or share links.

## Key decisions

### Decision 1: A browser workflow instead of a command-line-only app

- **AI suggestion (retrospective):** A CLI would be the smallest interface to build.
- **I chose:** A local web page with a folder picker, photo drop zone, marked preview, and results list.
- **Why:** The intended user is office staff, and the assessment specifically scores usability and UI.
- **Evidence:** The UI has separate gallery and group-photo steps in [faceid/static/index.html](./faceid/static/index.html); the CLI remains an optional alternative in [faceid/cli.py](./faceid/cli.py).

### Decision 2: Run recognition locally on CPU

- **AI suggestion (retrospective):** A hosted recognition API could avoid packaging models.
- **I chose:** Local InsightFace ONNX models with the CPU execution provider, and a server bound to `127.0.0.1`.
- **Why:** Runtime internet and online APIs are explicitly prohibited; keeping photos local also fits the privacy-sensitive use case.
- **Evidence:** [faceid/engine.py](./faceid/engine.py) selects `CPUExecutionProvider`; [run.py](./run.py) binds Uvicorn to `127.0.0.1`. The models are downloaded separately during setup.

### Decision 3: Return `Unknown` instead of always naming the closest person

- **AI suggestion (retrospective):** Always return the nearest gallery identity.
- **I chose:** Only accept a match that clears a configurable similarity threshold; otherwise label it `Unknown`.
- **Why:** An incorrect name is worse than an inconclusive result when some group-photo attendees are not in the gallery.
- **Evidence:** The threshold and default are in [faceid/config.py](./faceid/config.py); the open-set decision is in [faceid/matcher.py](./faceid/matcher.py). The UI exposes the strictness slider in [faceid/static/index.html](./faceid/static/index.html).

### Decision 4: Assign each gallery identity to at most one face

- **AI suggestion (retrospective):** Match every face independently to its nearest gallery embedding.
- **I chose:** Use a one-to-one Hungarian assignment for each group photo.
- **Why:** Independent nearest-neighbor matches could assign the same person to several faces in one image.
- **Evidence:** [faceid/matcher.py](./faceid/matcher.py) uses `linear_sum_assignment`; the limitation for duplicate appearances is documented in [README.md](./README.md).

### Decision 5: Separate slow detection from threshold rematching

- **AI suggestion (retrospective):** Re-run the full recognition pipeline whenever the threshold changes.
- **I chose:** Detect and embed once, then re-apply the matching decision to the saved faces when the slider moves.
- **Why:** This keeps threshold feedback immediate and avoids repeating the expensive CPU detection step.
- **Evidence:** Detection and decision are separate methods in [faceid/pipeline.py](./faceid/pipeline.py); `/api/rematch` reuses the saved image, faces, and gallery in [faceid/server.py](./faceid/server.py).

### Decision 6: Cache gallery embeddings in local files

- **AI suggestion (retrospective):** Recompute every reference embedding each time, or add a database.
- **I chose:** Store content-hash-keyed `.npz` cache entries under `.cache/embeddings/`.
- **Why:** Reopening the same folder should be fast without adding a database service or any network dependency.
- **Evidence:** Cache lookup and writes are in [faceid/gallery.py](./faceid/gallery.py); the cache path is in [faceid/config.py](./faceid/config.py). The cache can be deleted to clear the saved fingerprints.

### Decision 7: Keep the gallery format simple

- **AI suggestion (retrospective):** Support multiple reference photos for each person.
- **I chose:** One image per person, with the filename supplying the name; when a reference image has several faces, select the face nearest the image centre and show a warning.
- **Why:** This follows the assessment's folder format and makes setup understandable without a separate identity-management screen.
- **Evidence:** Naming, reference selection, and warnings are implemented in [faceid/gallery.py](./faceid/gallery.py) and explained in [README.md](./README.md).

### Decision 8: Report evaluation counts with their limitations

- **AI suggestion (retrospective):** Present a single accuracy number as a general performance claim.
- **I chose:** Include LFW, synthetic collage, and one real-photo result, while stating that these small tests are not an accuracy estimate.
- **Why:** A public benchmark and a small real-photo check provide useful evidence, but do not establish performance on every office group or demographic.
- **Evidence:** The test counts and caveats are in [README.md](./README.md) and [docs/evidence/group_test_results.md](./docs/evidence/group_test_results.md).

## Chat excerpts

### Excerpt 1: Reviewing the project structure

> **Me:** can you give project and folder architecture of this project
>
> **AI:** The application is primarily in-memory:
> - One gallery is retained in `AppState.gallery`.
> - The most recently uploaded photo and detected faces are retained in `AppState.last`.
> - Access is synchronized using a lock.
> - Gallery loading runs in a background thread.

### Excerpt 2: Restating the offline constraint and documenting honestly

> **Me:** "It must work fully offline on your own laptop. No internet at runtime. No online APIs or hosted services."
>
> **AI:** "I can help create both docs. I’ll check Archify’s instructions and the project’s existing files first; I’ll avoid inventing AI-chat excerpts or history that isn’t available in this conversation."

## Full AI chat history

The full project-development history from the very first implementation prompt is not available in this workspace. The excerpts above are from the current documentation session only, so they are not a substitute for the complete history requested by the assessment. Before submission, export the original AI conversations (or add share links) and append them here or include the exported files in the repository. Do not claim that this limited transcript is the full project history.
