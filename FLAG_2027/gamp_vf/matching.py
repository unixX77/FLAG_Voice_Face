"""Stage F: assignment between voice groups (rows) and face groups (columns).

Training-time pseudo-label refinement only; never used to score test trials.

* ``local_selection`` (F1): independent per-row choice: reciprocal best + margin.
* ``hungarian_partial`` (F2): maximum-weight one-to-one matching with dummy
  nodes, so a row stays unmatched when every real option scores below
  ``min_score``. Rectangular matrices are supported.
* ``sinkhorn_partial`` (F3): log-domain entropic OT with a dustbin row/column
  (unmatched mass). Returned weights are **assignment weights**, not
  calibrated probabilities.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp


def local_selection(scores: np.ndarray, min_margin: float = 0.0) -> np.ndarray:
    """match[i] = column or -1: reciprocal top-1 with row margin >= min_margin."""
    match = np.full(scores.shape[0], -1)
    if scores.size == 0:
        return match
    best = scores.argmax(1)
    reciprocal = scores.argmax(0)[best] == np.arange(len(scores))
    if scores.shape[1] > 1:
        part = -np.partition(-scores, 1, axis=1)
        margin = part[:, 0] - part[:, 1]
    else:
        margin = np.full(len(scores), np.inf)
    ok = reciprocal & (margin >= min_margin)
    match[ok] = best[ok]
    return match


def hungarian_partial(scores: np.ndarray, min_score: float) -> np.ndarray:
    """One-to-one max-weight matching; pairs scoring below ``min_score`` are left unmatched."""
    n, m = scores.shape
    match = np.full(n, -1)
    if n == 0 or m == 0:
        return match
    # Augment with n row-dummies (columns) and m column-dummies (rows): choosing
    # a dummy is worth exactly min_score, so a real pair is only kept if better.
    big = np.full((n + m, m + n), -1e9)
    big[:n, :m] = scores
    big[:n, m:] = np.where(np.eye(n, dtype=bool), min_score, -1e9)
    big[n:, :m] = np.where(np.eye(m, dtype=bool), min_score, -1e9)
    big[n:, m:] = min_score  # dummy-dummy, free
    rows, cols = linear_sum_assignment(-big)
    for r, c in zip(rows, cols):
        if r < n and c < m and scores[r, c] >= min_score:
            match[r] = c
    return match


def sinkhorn_partial(scores: np.ndarray, temperature: float = 0.05, dustbin: float = 0.0,
                     iterations: int = 200) -> np.ndarray:
    """Log-domain Sinkhorn with dustbins. Returns (n+1, m+1) transport weights.

    Each real row/column carries unit mass; the dustbin row/column absorb
    unmatched mass (capacities m and n). ``dustbin`` is the score of being
    unmatched (same units as ``scores``).
    """
    n, m = scores.shape
    logits = np.full((n + 1, m + 1), dustbin / temperature)
    logits[:n, :m] = scores / temperature
    log_mu = np.r_[np.zeros(n), np.log(m)] if m else np.zeros(n + 1)
    log_nu = np.r_[np.zeros(m), np.log(n)] if n else np.zeros(m + 1)
    # Normalise total masses so both marginals sum to n + m.
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    for _ in range(iterations):
        u = log_mu - logsumexp(logits + v[None, :], axis=1)
        v = log_nu - logsumexp(logits + u[:, None], axis=0)
    return np.exp(logits + u[:, None] + v[None, :])


def sinkhorn_match(scores: np.ndarray, temperature: float = 0.05, dustbin: float = 0.0,
                   min_weight: float = 0.5, iterations: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Row argmax of the Sinkhorn plan; unmatched (-1) if the best real weight < min_weight
    or the dustbin wins. Returns (match, weight of the chosen entry)."""
    n, m = scores.shape
    match = np.full(n, -1)
    weight = np.zeros(n)
    if n == 0 or m == 0:
        return match, weight
    plan = sinkhorn_partial(scores, temperature, dustbin, iterations)
    real = plan[:n, :m]
    best = real.argmax(1)
    w = real[np.arange(n), best]
    ok = (w >= min_weight) & (w >= plan[:n, m])
    # Keep the assignment one-to-one: for a column claimed twice, keep the heavier row.
    for col in np.unique(best[ok]):
        rows = np.flatnonzero(ok & (best == col))
        ok[rows[rows != rows[np.argmax(w[rows])]]] = False
    match[ok] = best[ok]
    weight[ok] = w[ok]
    return match, weight
