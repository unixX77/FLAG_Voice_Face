"""Modality-agnostic unimodal verification and retrieval (Phase 1 voices, Phase 2 faces).

Trials are generated deterministically from metadata only (never from embeddings):
  * genuine pairs need different ``content`` keys (audio/image hash), so a
    recording is never paired with a copy of itself;
  * every genuine pair is tagged with its condition (same/different ``group``,
    e.g. language, and same/different ``session``, e.g. source video);
  * impostors are drawn per genuine anchor: same gender + same group, different
    gender + same group, same gender + different group, and ``similar`` (the
    anchor's most similar same-gender identity by a metadata attribute).
Scores are cosine similarities of the given embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


@dataclass
class Pool:
    identity: np.ndarray            # (N,) str
    gender: np.ndarray              # (N,) 0/1
    group: np.ndarray               # (N,) e.g. language id; -1 unknown
    session: np.ndarray             # (N,) e.g. video id; "" unknown
    content: np.ndarray             # (N,) hash/path; equal = same recording
    attribute: np.ndarray | None = None  # (N,) e.g. F0 for "similar" impostors
    quality: np.ndarray | None = None    # (N,) e.g. duration seconds

    def __post_init__(self):
        n = len(self.identity)
        for name in ("gender", "group", "session", "content"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} does not align with identity")


@dataclass
class Trials:
    a: np.ndarray
    b: np.ndarray
    label: np.ndarray        # 1 genuine, 0 impostor
    condition: np.ndarray    # str tag


GENUINE = ("same_group_same_session", "same_group_diff_session", "diff_group_same_session", "diff_group_diff_session")
IMPOSTOR = ("imp_same_gender_same_group", "imp_diff_gender_same_group", "imp_same_gender_diff_group", "imp_similar")


def build_trials(pool: Pool, per_identity: int = 40, seed: int = 0) -> Trials:
    """Deterministic trials. For each identity, up to ``per_identity`` genuine pairs per
    genuine condition; each genuine pair's first clip is the anchor of one impostor per
    impostor condition (when such an impostor exists)."""
    rng = np.random.default_rng(seed)
    ids = np.unique(pool.identity)  # sorted -> deterministic
    rows_of = {i: np.flatnonzero(pool.identity == i) for i in ids}
    gender_of = {i: int(pool.gender[rows_of[i][0]]) for i in ids}
    for i in ids:
        if len(np.unique(pool.gender[rows_of[i]])) != 1:
            raise ValueError(f"identity {i} has more than one gender")
    similar_of = {}
    if pool.attribute is not None:
        med = {i: np.nanmedian(pool.attribute[rows_of[i]]) for i in ids}
        for i in ids:
            cand = [j for j in ids if j != i and gender_of[j] == gender_of[i] and np.isfinite(med[j])]
            if cand and np.isfinite(med[i]):
                similar_of[i] = min(cand, key=lambda j: (abs(med[j] - med[i]), j))
    a, b, lab, cond = [], [], [], []

    def add(x, y, label, c):
        a.append(x), b.append(y), lab.append(label), cond.append(c)

    for i in ids:
        r = rows_of[i]
        # All candidate genuine pairs (upper triangle, different content), grouped by condition.
        x, y = np.triu_indices(len(r), 1)
        x, y = r[x], r[y]
        keep = pool.content[x] != pool.content[y]
        x, y = x[keep], y[keep]
        same_g = pool.group[x] == pool.group[y]
        same_s = (pool.session[x] == pool.session[y]) & (pool.session[x] != "")
        for c, m in zip(GENUINE, (same_g & same_s, same_g & ~same_s, ~same_g & same_s, ~same_g & ~same_s)):
            idx = np.flatnonzero(m)
            if len(idx) == 0:
                continue
            idx = rng.choice(idx, min(per_identity, len(idx)), replace=False)
            for k in np.sort(idx):
                anchor, other = x[k], y[k]
                add(anchor, other, 1, c)
                # Impostors for this anchor.
                g, grp = gender_of[i], pool.group[anchor]
                others = pool.identity != i
                options = {
                    "imp_same_gender_same_group": others & (pool.gender == g) & (pool.group == grp),
                    "imp_diff_gender_same_group": others & (pool.gender != g) & (pool.group == grp),
                    "imp_same_gender_diff_group": others & (pool.gender == g) & (pool.group != grp) & (pool.group >= 0),
                }
                if i in similar_of:
                    options["imp_similar"] = (pool.identity == similar_of[i]) & (pool.group == grp)
                for ic, mask in options.items():
                    cand = np.flatnonzero(mask)
                    if len(cand):
                        add(anchor, int(rng.choice(cand)), 0, ic)
    return Trials(np.asarray(a), np.asarray(b), np.asarray(lab), np.asarray(cond))


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def scores(emb: np.ndarray, trials: Trials) -> np.ndarray:
    e = unit(emb)
    return (e[trials.a] * e[trials.b]).sum(1)


def eer(labels: np.ndarray, s: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2)


def verification_metrics(labels: np.ndarray, s: np.ndarray) -> dict:
    fpr, tpr, _ = roc_curve(labels, s)
    def tpr_at(f):
        return float(np.interp(f, fpr, tpr))
    return {"eer": eer(labels, s), "auc": float(roc_auc_score(labels, s)), "tpr_at_fpr1": tpr_at(0.01),
            "tpr_at_fpr5": tpr_at(0.05), "n_genuine": int(labels.sum()), "n_impostor": int((1 - labels).sum()),
            "genuine_mean": float(s[labels == 1].mean()), "impostor_mean": float(s[labels == 0].mean())}


def condition_metrics(trials: Trials, s: np.ndarray, genuine: tuple[str, ...], impostor: tuple[str, ...]) -> dict | None:
    """Metrics for a chosen genuine-condition set vs impostor-condition set."""
    m = np.isin(trials.condition, genuine + impostor)
    if trials.label[m].sum() == 0 or (1 - trials.label[m]).sum() == 0:
        return None
    return verification_metrics(trials.label[m], s[m])


def retrieval(emb: np.ndarray, pool: Pool, query_mask: np.ndarray | None = None,
              gallery_mask: np.ndarray | None = None, exclude_same_session: bool = False) -> dict:
    """Leave-one-out speaker/face retrieval. Gallery excludes the query's own content
    (duplicates) and optionally its session. Returns R@1, R@5, MRR and neighbour biases."""
    e = unit(emb).astype(np.float32)
    n = len(e)
    q_idx = np.flatnonzero(np.ones(n, bool) if query_mask is None else query_mask)
    g_mask = np.ones(n, bool) if gallery_mask is None else gallery_mask
    r1 = r5 = 0
    rr, same_group_nn, same_group_base, nn_same_gender = [], [], [], []
    evaluated = 0
    for start in range(0, len(q_idx), 1024):
        q = q_idx[start:start + 1024]
        sim = e[q] @ e.T
        for row, qi in enumerate(q):
            valid = g_mask & (pool.content != pool.content[qi])
            if exclude_same_session and pool.session[qi] != "":
                valid &= pool.session != pool.session[qi]
            positives = valid & (pool.identity == pool.identity[qi])
            if not positives.any():
                continue
            evaluated += 1
            sv = np.where(valid, sim[row], -np.inf)
            best_pos = sv[positives].max()
            rank = int((sv > best_pos).sum()) + 1
            r1 += rank == 1
            r5 += rank <= 5
            rr.append(1.0 / rank)
            nn = int(np.argmax(sv))
            nn_same_gender.append(pool.gender[nn] == pool.gender[qi])
            if pool.group[qi] >= 0:
                same_group_nn.append(pool.group[nn] == pool.group[qi])
                same_group_base.append(float((valid & (pool.group == pool.group[qi])).sum() / valid.sum()))
    out = {"queries": evaluated, "recall_at_1": r1 / max(evaluated, 1), "recall_at_5": r5 / max(evaluated, 1),
           "mrr": float(np.mean(rr)) if rr else float("nan"),
           "nn_same_gender": float(np.mean(nn_same_gender)) if nn_same_gender else float("nan")}
    if same_group_nn:
        out["nn_same_group"] = float(np.mean(same_group_nn))
        out["nn_same_group_base_rate"] = float(np.mean(same_group_base))
    return out


def fisher_ratio(emb: np.ndarray, labels: np.ndarray) -> float:
    """trace(between-class scatter) / trace(within-class scatter) on unit vectors."""
    e = unit(emb)
    mu = e.mean(0)
    sb = sw = 0.0
    for c in np.unique(labels):
        x = e[labels == c]
        m = x.mean(0)
        sb += len(x) * ((m - mu) ** 2).sum()
        sw += ((x - m) ** 2).sum()
    return float(sb / sw)


def variance_explained(emb: np.ndarray, labels: np.ndarray) -> float:
    """Share of total variance explained by class means (R^2 of a one-way ANOVA, summed over dims)."""
    e = unit(emb)
    total = ((e - e.mean(0)) ** 2).sum()
    within = sum(((e[labels == c] - e[labels == c].mean(0)) ** 2).sum() for c in np.unique(labels))
    return float(1 - within / total)
