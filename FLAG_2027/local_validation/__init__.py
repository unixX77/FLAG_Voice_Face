"""Identity-disjoint local validation utilities for FLAG 2027."""

from .metrics import compute_eer, evaluate_score_file

__all__ = ["compute_eer", "evaluate_score_file"]
