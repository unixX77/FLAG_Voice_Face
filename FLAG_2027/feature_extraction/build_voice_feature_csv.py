#!/usr/bin/env python3
"""Combine layer-wise WavLM statistics and physiology into one voice CSV.

Example (layers 3 to 6 averaged, mean pooling, plus physiology)::

    python FLAG_2027/feature_extraction/build_voice_feature_csv.py \\
      --layers-npy features/voice_v2_wavlm_layers/train_English_layers.npy \\
      --layers 3-6 --pooling mean \\
      --physiology-npy features/voice_v3_physiology/train_English_physiology.npy \\
      --manifest train_set/train_set/train_English.txt \\
      --output-csv features/voice_v4_combined/train_English_voices.csv

A six-field training manifest appends its class index as the final column, as
in the original feature CSVs; a three-field evaluation manifest does not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_layers(text: str) -> list[int]:
    layers: list[int] = []
    for part in text.split(","):
        if "-" in part:
            start, stop = (int(value) for value in part.split("-"))
            layers.extend(range(start, stop + 1))
        else:
            layers.append(int(part))
    return layers


def manifest_labels(path: Path) -> np.ndarray | None:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if all(len(row) == 6 for row in rows):
        return np.asarray([int(row[5]) for row in rows], dtype=np.int64)
    if all(len(row) == 3 for row in rows):
        return None
    raise ValueError(f"Manifest must have 3 or 6 fields per row: {path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--layers-npy", type=Path)
    parser.add_argument(
        "--layers",
        default="3-6",
        help="Hidden-state indices to average, e.g. '6', '3-6', or '2,4,6'",
    )
    parser.add_argument("--pooling", choices=("mean", "meanstd"), default="mean")
    parser.add_argument("--physiology-npy", type=Path)
    parser.add_argument(
        "--extra-csv",
        type=Path,
        help="Another feature CSV to append (e.g. the x-vector features)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    labels = manifest_labels(args.manifest)
    rows = len(args.manifest.read_text().splitlines())
    blocks: list[np.ndarray] = []
    description: dict[str, object] = {}

    if args.layers_npy is not None:
        hidden = np.load(args.layers_npy, mmap_mode="r")
        layers = parse_layers(args.layers)
        if max(layers) >= hidden.shape[1] or min(layers) < 0:
            raise ValueError(f"Layers {layers} outside 0..{hidden.shape[1] - 1}")
        selected = np.asarray(hidden[:, layers], dtype=np.float32).mean(axis=1)
        width = hidden.shape[2] // 2
        blocks.append(selected if args.pooling == "meanstd" else selected[:, :width])
        description.update(layers=layers, pooling=args.pooling)

    if args.physiology_npy is not None:
        physiology = np.load(args.physiology_npy).astype(np.float32)
        # Unsupervised imputation of rare unvoiced recordings.
        medians = np.nanmedian(physiology, axis=0)
        missing = np.isnan(physiology)
        physiology[missing] = np.take(medians, np.nonzero(missing)[1])
        blocks.append(physiology)
        description["physiology_dims"] = physiology.shape[1]
        description["physiology_imputed_values"] = int(missing.sum())

    if args.extra_csv is not None:
        extra = pd.read_csv(args.extra_csv, header=None, dtype=np.float32).to_numpy()
        if labels is not None:
            if not np.array_equal(extra[:, -1].astype(np.int64), labels):
                raise ValueError("Extra CSV labels do not match the manifest")
            extra = extra[:, :-1]
        blocks.append(extra)
        description["extra_dims"] = extra.shape[1]

    if not blocks:
        raise ValueError("Provide at least one feature source")
    if any(len(block) != rows for block in blocks):
        raise ValueError(f"Every feature source must have {rows} rows")
    matrix = np.concatenate(blocks, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("Combined features contain non-finite values")
    output = matrix if labels is None else np.column_stack((matrix, labels))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_csv, output, delimiter=",", fmt="%.8g")

    description.update(
        manifest=str(args.manifest),
        layers_npy=str(args.layers_npy),
        physiology_npy=str(args.physiology_npy),
        extra_csv=str(args.extra_csv),
        rows=rows,
        feature_dim=matrix.shape[1],
    )
    report = args.output_csv.with_name(args.output_csv.stem + "_report.json")
    report.write_text(json.dumps(description, indent=2) + "\n")
    print(json.dumps(description, indent=2))


if __name__ == "__main__":
    main()
