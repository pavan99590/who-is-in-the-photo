from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import MatchConfig


@dataclass
class Match:
    face_index: int
    status: str                 # "known" | "unknown"
    name: str | None            # set only when known
    score: float                # cosine similarity to the assigned person (or best candidate)
    margin: float               # score minus the best *other* person's score
    candidate: str | None       # closest person in the gallery (also for unknown faces)
    candidate_score: float


def match_faces(
    face_embs: np.ndarray,
    gallery_embs: np.ndarray,
    names: list[str],
    cfg: MatchConfig = MatchConfig(),
) -> list[Match]:
    """Open-set identification.

    1. Cosine similarity of every face to every gallery person (embeddings are L2-normalised).
    2. Hungarian assignment so one person can be claimed by at most one face in a photo.
    3. A face is "known" only if its assigned person clears the threshold and the margin.
    Everything else is "unknown".
    """
    n_faces, n_people = len(face_embs), len(names)
    if n_faces == 0:
        return []
    if n_people == 0:
        return [Match(i, "unknown", None, 0.0, 0.0, None, 0.0) for i in range(n_faces)]

    sims = np.asarray(face_embs, dtype=np.float32) @ np.asarray(gallery_embs, dtype=np.float32).T
    rows, cols = linear_sum_assignment(-sims)
    assigned = dict(zip(rows.tolist(), cols.tolist()))

    results: list[Match] = []
    for i in range(n_faces):
        row = sims[i]
        best = int(np.argmax(row))
        j = assigned.get(i, best)
        others = np.delete(row, j)
        others_max = float(others.max()) if others.size else 0.0
        score = float(row[j])
        margin = score - others_max

        known = i in assigned and score >= cfg.threshold and margin >= cfg.min_margin
        results.append(
            Match(
                face_index=i,
                status="known" if known else "unknown",
                name=names[j] if known else None,
                score=score,
                margin=margin,
                candidate=names[best],
                candidate_score=float(row[best]),
            )
        )
    return results
