"""Summarize best-checkpoint metrics across completed validation folds."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


METRICS = ("standard_eer", "standard_auc", "gender_eer", "gender_auc", "mean_eer")


def load_best_metrics(checkpoint: Path) -> tuple[int, dict[str, float]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if "epoch" not in payload or "validation" not in payload:
        raise ValueError(f"Not a fold-training checkpoint: {checkpoint}")
    validation = payload["validation"]
    missing = [metric for metric in METRICS if metric not in validation]
    if missing:
        raise ValueError(f"Missing validation metrics in {checkpoint}: {missing}")
    return int(payload["epoch"]), {
        metric: float(validation[metric]) for metric in METRICS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Directory containing fold_0 through fold_4 run directories",
    )
    parser.add_argument(
        "--checkpoint-name", default="checkpoint_best.pth.tar"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float]] = []
    print("fold  epoch  standard_EER  gender_EER  mean_EER")
    for fold in range(5):
        checkpoint = args.run_root / f"fold_{fold}" / args.checkpoint_name
        epoch, metrics = load_best_metrics(checkpoint)
        rows.append(metrics)
        print(
            f"{fold:>4d}  {epoch:>5d}  {metrics['standard_eer']:>12.4f}  "
            f"{metrics['gender_eer']:>10.4f}  {metrics['mean_eer']:>8.4f}"
        )
    print("\nFive-fold aggregate (mean +/- population std)")
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows])
        print(f"{metric:>13s}: {values.mean():.4f} +/- {values.std(ddof=0):.4f}")


if __name__ == "__main__":
    main()
