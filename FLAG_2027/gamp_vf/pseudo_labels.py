"""Stage E: confidence-aware pseudo-label mining between voice groups and face groups.

Inputs are *unlabelled* pools: voice clips and face images without identity and
without row alignment (row alignment would leak the association). Groups come
from ``clustering``; scores are raw cosines of group prototypes in the teacher's
shared space.

Every candidate carries: score (raw cosine), margin, reciprocal flag,
cross-language consistency, predicted-gender agreement, a calibrated confidence
(logistic model fitted on a *labelled calibration pool the teacher never
trained on*), a decision (high / ambiguous / unreliable) and a reason.
Ground truth is attached only for diagnostics, after decisions are made.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from .prototypes import identity_prototypes

MASKED = -2.0  # score given to gender-incompatible pairs (below any cosine)
FEATURES = ("score", "margin", "reciprocal", "language_consistency", "log_voice_size", "log_face_size")


@dataclass
class Groups:
    ids: np.ndarray            # group ids (>= 0)
    prototypes: np.ndarray     # (G, d)
    members: list[np.ndarray]  # row indices into the pool per group
    language_prototypes: list[dict[int, np.ndarray]]
    gender_prob: np.ndarray    # (G,) predicted P(male); nan if unavailable


def make_groups(embeddings, labels, rows, languages=None, balanced=False, gender_prob_rows=None) -> Groups:
    keep = labels >= 0
    if not keep.any():  # nothing clustered: an empty group set, never a forced assignment
        return Groups(np.array([], dtype=np.int64), np.zeros((0, embeddings.shape[1])), [], [], np.array([]))
    p = identity_prototypes(embeddings[keep], labels[keep], None if languages is None else languages[keep],
                            "language_balanced" if balanced else "uniform")
    members = [rows[keep][labels[keep] == g] for g in p.identities]
    if gender_prob_rows is None:
        gp = np.full(len(p.identities), np.nan)
    else:
        gp = np.array([gender_prob_rows[keep][labels[keep] == g].mean() for g in p.identities])
    return Groups(p.identities, p.prototypes, members, p.language_prototypes, gp)


def gender_compatible(voice: Groups, face: Groups, confidence: float = 0.25) -> np.ndarray:
    """False only where both predictions are confident and disagree (unknown -> compatible)."""
    v, f = voice.gender_prob[:, None], face.gender_prob[None, :]
    sure = (np.abs(v - 0.5) >= confidence) & (np.abs(f - 0.5) >= confidence)
    return ~(sure & ((v > 0.5) != (f > 0.5)))


def candidate_features(voice: Groups, face: Groups, scores: np.ndarray) -> dict[str, np.ndarray]:
    """Per voice group: its best face group and the features describing that choice."""
    if scores.size == 0:
        empty = np.zeros(scores.shape[0])
        return {"best_face": np.zeros(scores.shape[0], dtype=np.int64), **{k: empty for k in FEATURES}}
    best = scores.argmax(1)
    top = scores[np.arange(len(scores)), best]
    second = np.sort(scores, axis=1)[:, -2] if scores.shape[1] > 1 else np.full(len(scores), -1.0)
    reciprocal = scores.argmax(0)[best] == np.arange(len(scores))
    consistency = np.ones(len(scores))
    for g, per_lang in enumerate(voice.language_prototypes):
        known = [v for lang, v in per_lang.items() if lang >= 0]
        if len(known) >= 2:  # fraction of languages whose own prototype picks the same face group
            allowed = scores[g] > MASKED + 0.5  # respect the gender mask
            consistency[g] = np.mean([np.argmax(np.where(allowed, face.prototypes @ v, MASKED))
                                      == best[g] for v in known])
    return {"best_face": best, "score": top, "margin": top - second, "reciprocal": reciprocal.astype(float),
            "language_consistency": consistency,
            "log_voice_size": np.log([len(m) for m in voice.members]),
            "log_face_size": np.log([len(face.members[b]) for b in best])}


def feature_matrix(c: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([c[k] for k in FEATURES]).astype(np.float64)


def group_majority(groups: Groups, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Majority true identity and purity per group (diagnostics / calibration labels only)."""
    major, purity = [], []
    if not groups.members:
        return np.array([], dtype=truth.dtype), np.array([])
    for m in groups.members:
        values, counts = np.unique(truth[m], return_counts=True)
        major.append(values[counts.argmax()])
        purity.append(counts.max() / len(m))
    return np.asarray(major), np.asarray(purity)


class ConfidenceCalibrator:
    """Logistic model P(candidate correct | features), fitted on the calibration pool."""

    def __init__(self, c: float = 1.0) -> None:
        self.model = LogisticRegression(C=c, max_iter=1000)
        self.mean = self.std = None
        self.constant: float | None = None

    def fit(self, features: np.ndarray, correct: np.ndarray) -> "ConfidenceCalibrator":
        if len(correct) == 0:
            self.constant = 0.0  # no calibration evidence: nothing may be accepted
            return self
        self.mean, self.std = features.mean(0), features.std(0) + 1e-9
        if len(np.unique(correct)) < 2:
            self.constant = float(correct.mean())  # degenerate pool: constant confidence
        else:
            self.model.fit((features - self.mean) / self.std, correct.astype(int))
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.constant is not None:
            return np.full(len(features), self.constant)
        return self.model.predict_proba((features - self.mean) / self.std)[:, 1]


def precision_threshold(confidence: np.ndarray, correct: np.ndarray, target: float) -> float:
    """Smallest confidence threshold whose accepted set reaches ``target`` precision (inf if none)."""
    order = np.argsort(-confidence)
    hits = np.cumsum(correct[order])
    precision = hits / np.arange(1, len(order) + 1)
    ok = np.flatnonzero(precision >= target)
    if len(ok) == 0:
        return float("inf")
    # Largest accepted prefix with the target precision.
    return float(confidence[order][ok.max()])


def decide(confidence: np.ndarray, high: float, low: float) -> np.ndarray:
    return np.where(confidence >= high, "high", np.where(confidence >= low, "ambiguous", "unreliable"))


def pair_rows(voice_rows: np.ndarray, face_rows: np.ndarray, cap: int, rng: np.random.Generator):
    """Training rows for one accepted pair: random voice clip x face image pairing."""
    n = min(cap, max(len(voice_rows), len(face_rows)))
    v = rng.choice(voice_rows, n, replace=len(voice_rows) < n)
    f = rng.choice(face_rows, n, replace=len(face_rows) < n)
    return f, v


def conflict_groups(voice_protos: np.ndarray, face_protos: np.ndarray, voice_sim: float, face_sim: float) -> np.ndarray:
    """Union-find over accepted pairs whose voice or face prototypes are as close as same-person groups:
    such pseudo identities may be one person and must never be each other's negatives."""
    n = len(voice_protos)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    vs, fs = voice_protos @ voice_protos.T, face_protos @ face_protos.T
    for i in range(n):
        for j in range(i + 1, n):
            if vs[i, j] >= voice_sim or fs[i, j] >= face_sim:
                parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])
