"""Evaluate a local score CSV against a fold's private local ground truth."""

from __future__ import annotations

import argparse
from pathlib import Path

from local_validation.metrics import evaluate_score_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    args = parser.parse_args()
    eer, auc = evaluate_score_file(args.ground_truth, args.scores)
    print(f"EER: {100.0 * eer:.4f}%")
    print(f"AUC: {100.0 * auc:.4f}%")


if __name__ == "__main__":
    main()
