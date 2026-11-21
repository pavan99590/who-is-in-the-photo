import io

import numpy as np
from PIL import Image

from faceid.config import DetectConfig
from faceid.engine import DetectedFace, nms, tile_boxes
from faceid.gallery import enroll_items, name_from_filename


def png_bytes(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buf, "PNG")
    return buf.getvalue()


def face(x1, y1, x2, y2, score=0.9, seed=0):
    e = np.random.default_rng(seed).normal(size=512).astype(np.float32)
    return DetectedFace(np.array([x1, y1, x2, y2], np.float32), score, e / np.linalg.norm(e))


class StubEngine:
    """Stands in for the real detector so the tests need no model files."""
    cfg = DetectConfig()
    cache_tag = "stub"

    def __init__(self, faces_by_call):
        self.faces_by_call, self.calls = list(faces_by_call), 0

    def detect(self, bgr, det_size=None, tiling=None):
        out = self.faces_by_call[self.calls]
        self.calls += 1
        return out


def test_name_from_filename():
    assert name_from_filename("Priya_Sharma.jpg") == "Priya Sharma"
    assert name_from_filename("  Ravi   Kumar .PNG") == "Ravi Kumar"


def test_enroll_handles_bad_files_duplicates_and_hidden(tmp_path):
    items = [
        ("Priya Sharma.png", png_bytes()),
        ("Priya_Sharma.png", png_bytes((65, 64))),      # duplicate name
        ("Nobody.png", png_bytes((66, 64))),            # no face
        ("broken.jpg", b"not an image"),
        ("._Priya Sharma.png", png_bytes()),            # macOS junk: ignored
        ("notes.txt", b"hello"),                        # not an image: ignored
    ]
    # files are processed in name order: broken (no detect call), Nobody, Priya Sharma, Priya_Sharma
    eng = StubEngine([[], [face(0, 0, 50, 50, seed=1)], [face(0, 0, 50, 50, seed=2)]])
    g = enroll_items(items, eng, cache_dir=tmp_path)
    assert g.names == ["Priya Sharma", "Priya Sharma (2)"]
    by_file = {r.filename: r for r in g.records}
    assert by_file["Nobody.png"].status == "no_face"
    assert by_file["broken.jpg"].status == "unreadable"
    assert "._Priya Sharma.png" not in by_file and "notes.txt" not in by_file
    assert any("duplicate" in w for w in g.warnings)


def test_multiple_faces_uses_largest_and_warns(tmp_path):
    small, big = face(0, 0, 20, 20, seed=1), face(10, 10, 60, 60, seed=2)
    g = enroll_items([("Asha Rao.png", png_bytes())], StubEngine([[small, big]]), cache_dir=tmp_path)
    assert np.allclose(g.embeddings[0], big.embedding)
    assert g.records[0].status == "multiple_faces"


def test_reference_prefers_centred_face_over_bigger_edge_face(tmp_path):
    # Regression from LFW (Jason Kidd probe): the biggest face was a half-cropped neighbour.
    left = face(4, 6, 87, 129, seed=1)
    centre = face(75, 68, 167, 196, seed=2)
    edge = face(181, -7, 292, 133, seed=3)          # biggest, but at the border
    g = enroll_items([("Jason Kidd.png", png_bytes((250, 250)))], StubEngine([[left, centre, edge]]), cache_dir=tmp_path)
    assert np.allclose(g.embeddings[0], centre.embedding)


def test_cache_avoids_recomputing(tmp_path):
    items = [("Asha Rao.png", png_bytes())]
    eng = StubEngine([[face(0, 0, 50, 50, seed=3)]])
    first = enroll_items(items, eng, cache_dir=tmp_path)
    second = enroll_items(items, eng, cache_dir=tmp_path)   # stub would IndexError on a 2nd detect
    assert eng.calls == 1
    assert np.allclose(first.embeddings, second.embeddings)


def test_nms_and_tiles():
    a, b, c = face(0, 0, 100, 100, 0.9), face(5, 5, 100, 100, 0.8), face(300, 300, 400, 400, 0.7)
    assert len(nms([a, b, c], 0.4)) == 2
    boxes = tile_boxes(3000, 2000, 1280, 0.25)
    assert boxes[0][:2] == (0, 0)
    assert max(b[2] for b in boxes) == 3000 and max(b[3] for b in boxes) == 2000
    assert tile_boxes(800, 600, 1280, 0.25) == [(0, 0, 800, 600)]
