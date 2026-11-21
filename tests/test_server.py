import io
import time

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from faceid.config import DetectConfig
from faceid.engine import DetectedFace
from faceid.server import create_app


def png(size=(300, 200), colour=(120, 130, 140)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "PNG")
    return buf.getvalue()


def emb(seed):
    e = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return e / np.linalg.norm(e)


def face(x1, y1, x2, y2, seed):
    return DetectedFace(np.array([x1, y1, x2, y2], np.float32), 0.9, emb(seed))


class StubEngine:
    cfg = DetectConfig()
    cache_tag = "stub-server"

    def __init__(self, calls):
        self.calls = list(calls)

    def detect(self, bgr, det_size=None, tiling=None):
        return self.calls.pop(0)


def wait_for_gallery(client, tries=50):
    for _ in range(tries):
        g = client.get("/api/gallery").json()
        if g["state"] in ("done", "error"):
            return g
        time.sleep(0.05)
    raise AssertionError("gallery never finished")


def make_client(tmp_path, calls):
    return TestClient(create_app(StubEngine(calls), cache_dir=tmp_path, default_threshold=0.4))


def test_full_flow_known_unknown_and_missing(tmp_path):
    # (images must differ, identical bytes share one cache entry)
    # enrol order is alphabetical: Asha, Ravi, then the group photo has Asha, a stranger, Ravi (left to right)
    calls = [[face(20, 20, 120, 120, 1)], [face(20, 20, 120, 120, 2)],
             [face(10, 40, 60, 100, 1), face(100, 40, 150, 100, 99), face(200, 40, 250, 100, 2)]]
    with make_client(tmp_path, calls) as c:
        files = [("files", ("Asha Rao.png", png(), "image/png")), ("files", ("Ravi_Kumar.png", png(colour=(10, 200, 90)), "image/png")),
                 ("files", ("notes.txt", b"x", "text/plain"))]
        assert c.post("/api/gallery", files=files).json()["total"] == 2
        g = wait_for_gallery(c)
        assert g["n_people"] == 2 and g["people"][0]["thumb"].startswith("data:image/jpeg")

        r = c.post("/api/identify", files={"file": ("group.png", png(), "image/png")}).json()
        assert [f["name"] for f in r["faces"]] == ["Asha Rao", None, "Ravi Kumar"]
        assert (r["n_known"], r["n_unknown"], r["missing"]) == (2, 1, [])
        assert c.get("/api/photo").status_code == 200
        assert c.get("/api/annotated").headers["content-type"] == "image/jpeg"

        r2 = c.post("/api/rematch", json={"threshold": 1.01}).json()    # impossible threshold: nobody passes
        assert r2["n_known"] == 0 and sorted(r2["missing"]) == ["Asha Rao", "Ravi Kumar"]


def test_identify_before_folder_gives_friendly_error(tmp_path):
    with make_client(tmp_path, []) as c:
        r = c.post("/api/identify", files={"file": ("g.png", png(), "image/png")})
        assert r.status_code == 409 and "folder" in r.json()["detail"].lower()


def test_empty_folder_and_bad_photo(tmp_path):
    with make_client(tmp_path, [[face(20, 20, 120, 120, 1)]]) as c:
        assert c.post("/api/gallery", files=[("files", ("a.txt", b"x", "text/plain"))]).status_code == 400
        c.post("/api/gallery", files=[("files", ("Asha Rao.png", png(), "image/png"))])
        wait_for_gallery(c)
        bad = c.post("/api/identify", files={"file": ("x.jpg", b"not an image", "image/jpeg")})
        assert bad.status_code == 400 and "not a photo" in bad.json()["detail"]


def test_group_photo_with_no_faces(tmp_path):
    with make_client(tmp_path, [[face(20, 20, 120, 120, 1)], []]) as c:
        c.post("/api/gallery", files=[("files", ("Asha Rao.png", png(), "image/png"))])
        wait_for_gallery(c)
        r = c.post("/api/identify", files={"file": ("g.png", png(), "image/png")}).json()
        assert r["n_faces"] == 0 and r["faces"] == []


def test_index_page_is_served_and_has_no_external_links(tmp_path):
    with make_client(tmp_path, []) as c:
        html = c.get("/").text
        assert "Who is in the photo?" in html
        assert "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "https://" not in html
