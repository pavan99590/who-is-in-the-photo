"""Local web app: everything runs on this computer, nothing leaves it."""
from __future__ import annotations

import base64
import io
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from .annotate import annotate
from .config import DEFAULT_CACHE_DIR, DetectConfig, MatchConfig
from .engine import FaceEngine, ModelsNotFoundError
from .gallery import enroll_items, is_candidate_file
from .imageio import ImageLoadError, open_pil
from .pipeline import Recognizer

STATIC = Path(__file__).parent / "static"


class Rematch(BaseModel):
    threshold: float
    min_margin: float = 0.0


class AppState:
    def __init__(self, engine, default_threshold: float):
        self.engine = engine
        self.rec = Recognizer(engine)
        self.default_threshold = default_threshold
        self.lock = threading.Lock()
        self.models = {"state": "loading", "message": ""}
        self.gallery = None
        self.enroll = {"state": "idle", "done": 0, "total": 0, "current": "", "message": ""}
        self.last: dict | None = None
        self.photo_id = 0


def _thumb_url(b: bytes | None) -> str | None:
    return None if not b else "data:image/jpeg;base64," + base64.b64encode(b).decode()


def create_app(engine=None, cache_dir=DEFAULT_CACHE_DIR, default_threshold: float | None = None) -> FastAPI:
    engine = engine or FaceEngine(DetectConfig())
    st = AppState(engine, MatchConfig().threshold if default_threshold is None else default_threshold)

    def warm_up() -> None:
        try:
            if hasattr(engine, "_load"):
                engine._load()
            st.models["state"] = "ready"
        except ModelsNotFoundError as exc:
            st.models.update(state="error", message=str(exc))
        except Exception as exc:  # pragma: no cover
            st.models.update(state="error", message=f"Could not start the face models: {exc}")

    @asynccontextmanager
    async def lifespan(_app):
        threading.Thread(target=warm_up, daemon=True).start()
        yield

    app = FastAPI(title="Who is in the photo?", lifespan=lifespan)

    # ------------------------------------------------------------------ pages
    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/status")
    def status():
        return {"models": st.models, "default_threshold": st.default_threshold,
                "people": 0 if st.gallery is None else len(st.gallery)}

    # ---------------------------------------------------------------- gallery
    def _enroll_job(items):
        try:
            def progress(i, n, name):
                st.enroll.update(done=i, total=n, current=name)

            g = enroll_items(items, st.engine, cache_dir=cache_dir, progress=progress)
            with st.lock:
                st.gallery, st.last = g, None
            st.enroll.update(state="done", done=len(g.records), total=len(g.records), current="")
        except ModelsNotFoundError as exc:
            st.enroll.update(state="error", message=str(exc))
        except Exception as exc:  # pragma: no cover
            st.enroll.update(state="error", message=f"Could not read the folder: {exc}")

    @app.post("/api/gallery")
    def start_gallery(files: list[UploadFile] = File(...)):
        if st.models["state"] == "error":
            raise HTTPException(503, st.models["message"])
        if st.enroll["state"] == "running":
            raise HTTPException(409, "Still reading the previous folder.")
        items = []
        for f in files:
            name = Path(f.filename or "").name
            if is_candidate_file(name):
                items.append((name, f.file.read()))
        if not items:
            raise HTTPException(400, "No photos found in that folder.")
        st.enroll.update(state="running", done=0, total=len(items), current="", message="")
        threading.Thread(target=_enroll_job, args=(items,), daemon=True).start()
        return {"started": True, "total": len(items)}

    @app.get("/api/gallery")
    def gallery_status():
        g = st.gallery
        people = []
        if g is not None:
            people = [{"name": r.name, "file": r.filename, "status": r.status, "message": r.message,
                       "thumb": _thumb_url(r.thumb)} for r in g.records]
        return {**st.enroll, "n_people": 0 if g is None else len(g), "people": people}

    # --------------------------------------------------------------- identify
    def _payload(result, gallery) -> dict:
        named = {f.name for f in result.faces if f.status == "known"}
        out = result.to_dict()
        out["missing"] = [n for n in gallery.names if n not in named]
        out["gallery_size"] = len(gallery)
        out["photo_id"] = st.photo_id
        return out

    @app.post("/api/identify")
    def identify(file: UploadFile = File(...), threshold: float | None = Form(None)):
        if st.enroll["state"] == "running":
            raise HTTPException(409, "Still reading the folder, one moment.")
        g = st.gallery
        if g is None or len(g) == 0:
            raise HTTPException(409, "Open the folder with everyone's photos first.")
        data = file.file.read()
        cfg = MatchConfig(threshold=st.default_threshold if threshold is None else threshold)
        try:
            with st.lock:
                import time
                t0 = time.perf_counter()
                img, faces = st.rec.detect(data)
                result = st.rec.decide(img, faces, g, cfg)
                result.elapsed_s = time.perf_counter() - t0
                st.photo_id += 1
                st.last = {"data": data, "img": img, "faces": faces, "gallery": g, "result": result, "jpeg": None}
        except ImageLoadError:
            raise HTTPException(400, "That file is not a photo I can read. Try a JPG or PNG.")
        except ModelsNotFoundError as exc:
            raise HTTPException(503, str(exc))
        return _payload(result, g)

    @app.post("/api/rematch")
    def rematch(body: Rematch):
        with st.lock:
            if st.last is None:
                raise HTTPException(409, "Add a group photo first.")
            last = st.last
            cfg = MatchConfig(threshold=body.threshold, min_margin=body.min_margin)
            result = st.rec.decide(last["img"], last["faces"], last["gallery"], cfg)
            result.elapsed_s = last["result"].elapsed_s     # keep the time the detection took
            last["result"] = result
        return _payload(result, last["gallery"])

    @app.get("/api/photo")
    def photo():
        """The group photo, EXIF-rotated exactly as the detector saw it, so boxes line up."""
        if st.last is None:
            raise HTTPException(404, "No photo yet.")
        if st.last["jpeg"] is None:
            buf = io.BytesIO()
            open_pil(st.last["data"]).save(buf, "JPEG", quality=92)
            st.last["jpeg"] = buf.getvalue()
        return Response(st.last["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/annotated")
    def annotated():
        if st.last is None:
            raise HTTPException(404, "No photo yet.")
        buf = io.BytesIO()
        annotate(st.last["data"], st.last["result"]).save(buf, "JPEG", quality=92)
        return Response(buf.getvalue(), media_type="image/jpeg",
                        headers={"Content-Disposition": 'attachment; filename="who-is-in-the-photo.jpg"'})

    return app


app = create_app()
