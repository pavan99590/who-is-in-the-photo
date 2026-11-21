import numpy as np

from faceid.config import MatchConfig
from faceid.matcher import match_faces


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def make_people(n=5, d=64, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([unit(rng.normal(size=d)) for _ in range(n)])


def noisy(v, amount, seed=1):
    rng = np.random.default_rng(seed)
    return unit(v + amount * rng.normal(size=v.shape))


NAMES = ["A", "B", "C", "D", "E"]


def test_exact_match_is_known():
    people = make_people()
    out = match_faces(people[[2]], people, NAMES)
    assert out[0].status == "known" and out[0].name == "C"


def test_stranger_is_unknown():
    people = make_people()
    stranger = unit(np.random.default_rng(99).normal(size=64))
    out = match_faces(stranger[None], people, NAMES, MatchConfig(threshold=0.5))
    assert out[0].status == "unknown" and out[0].name is None
    assert out[0].candidate in NAMES  # still reports the closest person for the UI


def test_one_person_cannot_be_claimed_twice():
    people = make_people()
    good = noisy(people[1], 0.02, seed=3)
    worse = noisy(people[1], 0.25, seed=4)
    out = match_faces(np.stack([good, worse]), people, NAMES, MatchConfig(threshold=0.3))
    names = [m.name for m in out if m.status == "known"]
    assert names.count("B") == 1
    assert out[0].name == "B"           # the better match wins the name


def test_threshold_controls_acceptance():
    people = make_people()
    face = noisy(people[0], 0.4, seed=5)
    strict = match_faces(face[None], people, NAMES, MatchConfig(threshold=0.99))
    loose = match_faces(face[None], people, NAMES, MatchConfig(threshold=0.1))
    assert strict[0].status == "unknown" and loose[0].status == "known"


def test_margin_rejects_ambiguous_face():
    a = unit([1, 0, 0.0])
    b = unit([1, 0.02, 0.0])             # near-twin of a
    face = unit([1, 0.01, 0.0])
    out = match_faces(face[None], np.stack([a, b]), ["A", "B"], MatchConfig(threshold=0.3, min_margin=0.05))
    assert out[0].status == "unknown"


def test_empty_gallery_and_no_faces():
    people = make_people()
    assert match_faces(np.zeros((0, 64)), people, NAMES) == []
    out = match_faces(people[:2], np.zeros((0, 64)), [])
    assert [m.status for m in out] == ["unknown", "unknown"]


def test_more_faces_than_people():
    people = make_people(2)
    faces = np.stack([people[0], people[1], unit(np.random.default_rng(7).normal(size=64))])
    out = match_faces(faces, people, ["A", "B"], MatchConfig(threshold=0.5))
    assert [m.status for m in out] == ["known", "known", "unknown"]
