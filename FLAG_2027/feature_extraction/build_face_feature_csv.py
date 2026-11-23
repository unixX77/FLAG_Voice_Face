#!/usr/bin/env python3
"""Build a face feature CSV from per-layer ViT features (see extract_face_features_layers.py).

Each ``--block`` adds one feature group and has the form ``NPY:LAYERS:TOKENS``:

- ``NPY``: a ``*_layers.npy`` file. With ``--flip-average`` the matching
  ``*_layers_flip.npy`` is averaged in (horizontal-flip test-time augmentation).
- ``LAYERS``: hidden-state indices to average, e.g. ``12``, ``8-12``, ``1-12``.
- ``TOKENS``: ``cls``, ``patch`` (mean patch token), ``both``, or ``farl_final``
  (the official 512-D FaRL embedding rebuilt from the last-layer CLS token;
  requires ``--farl-checkpoint`` and the ``clip`` package).

A six-field training manifest appends its class index as the final column.

Example::

    python FLAG_2027/feature_extraction/build_face_feature_csv.py \\
      --manifest train_set/train_set/train_English.txt --flip-average \\
      --block features/face_layers_dinov2/train_English_layers.npy:1-12:cls \\
      --output-csv features/face_v3_dinov2_l1_12_cls/train_English_faces.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from build_voice_feature_csv import manifest_labels, parse_layers


def load_layers(path: Path, flip_average: bool) -> np.ndarray:
    features = np.load(path, mmap_mode="r")
    if not flip_average:
        return features
    flipped = np.load(path.with_name(path.stem + "_flip.npy"), mmap_mode="r")
    return (np.asarray(features, dtype=np.float32) + np.asarray(flipped, dtype=np.float32)) / 2


def farl_final(cls_tokens: np.ndarray, checkpoint: Path) -> np.ndarray:
    import clip
    import torch

    from extract_face_features_farl import load_farl_model

    model, _, _, _ = load_farl_model(clip, checkpoint, torch.device("cpu"))
    with torch.no_grad():
        projected = model.visual.ln_post(torch.from_numpy(cls_tokens)) @ model.visual.proj
    return torch.nn.functional.normalize(projected, dim=1).numpy()


def build_block(spec: str, flip_average: bool, farl_checkpoint: Path | None) -> np.ndarray:
    path_text, layer_text, tokens = spec.rsplit(":", 2)
    features = load_layers(Path(path_text), flip_average)
    layers = parse_layers(layer_text)
    width = features.shape[2] // 2
    if tokens == "farl_final":
        if farl_checkpoint is None or layers != [features.shape[1] - 1]:
            raise ValueError("farl_final needs --farl-checkpoint and the last layer only")
        if flip_average:
            # Project each view separately, then average, matching per-image FaRL.
            plain = np.load(Path(path_text), mmap_mode="r")[:, layers[0], :width]
            flipped = np.load(
                Path(path_text).with_name(Path(path_text).stem + "_flip.npy"), mmap_mode="r"
            )[:, layers[0], :width]
            return (
                farl_final(np.asarray(plain, np.float32), farl_checkpoint)
                + farl_final(np.asarray(flipped, np.float32), farl_checkpoint)
            ) / 2
        return farl_final(np.asarray(features[:, layers[0], :width], np.float32), farl_checkpoint)
    selected = np.asarray(features[:, layers], dtype=np.float32).mean(axis=1)
    if tokens == "cls":
        return selected[:, :width]
    if tokens == "patch":
        return selected[:, width:]
    if tokens == "both":
        return selected
    raise ValueError(f"Unknown token selection: {tokens}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--block", action="append", required=True)
    parser.add_argument("--flip-average", action="store_true")
    parser.add_argument("--farl-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    labels = manifest_labels(args.manifest)
    blocks = [build_block(spec, args.flip_average, args.farl_checkpoint) for spec in args.block]
    rows = len(args.manifest.read_text().splitlines())
    if any(len(block) != rows for block in blocks):
        raise ValueError(f"Every block must have {rows} rows")
    matrix = np.concatenate(blocks, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("Face features contain non-finite values")
    output = matrix if labels is None else np.column_stack((matrix, labels))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_csv, output, delimiter=",", fmt="%.8g")
    report = {
        "manifest": str(args.manifest),
        "blocks": args.block,
        "flip_average": args.flip_average,
        "rows": rows,
        "feature_dim": matrix.shape[1],
    }
    args.output_csv.with_name(args.output_csv.stem + "_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
