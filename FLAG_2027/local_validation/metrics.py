"""Verification metrics and score-file validation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, roc_curve


def compute_eer(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (EER, ROC-AUC); larger scores must mean a stronger match."""
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be equally sized 1-D arrays")
    if len(labels) == 0 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores contain NaN or infinity")

    fpr, tpr, _ = roc_curve(labels, scores)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return float(eer), float(roc_auc_score(labels, scores))


def _read_two_column_file(path: Path, value_name: str) -> dict[str, str]:
    values: dict[str, str] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pair_id", value_name}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            pair_id = row["pair_id"]
            if pair_id in values:
                raise ValueError(f"duplicate pair_id {pair_id} in {path}")
            values[pair_id] = row[value_name]
    return values


def evaluate_score_file(
    ground_truth_path: Path, score_path: Path
) -> tuple[float, float]:
    """Join ground truth and predictions by pair ID, then compute EER/AUC."""
    truth = _read_two_column_file(ground_truth_path, "label")
    predictions = _read_two_column_file(score_path, "score")
    if set(truth) != set(predictions):
        missing = sorted(set(truth) - set(predictions))[:5]
        extra = sorted(set(predictions) - set(truth))[:5]
        raise ValueError(f"pair ID mismatch; missing={missing}, extra={extra}")
    pair_ids = sorted(truth)
    labels = np.asarray([int(truth[key]) for key in pair_ids])
    scores = np.asarray([float(predictions[key]) for key in pair_ids])
    return compute_eer(labels, scores)
