#!/usr/bin/env python3
"""Extract deterministic FaRL face embeddings in FLAG manifest order.

Uses the official FaRL CVPR 2022 inference policy: an OpenAI CLIP ViT-B/16
image encoder, CLIP preprocessing of the original face crop, and the official
FaRL LAIONFace20M ep16 checkpoint. Identical image files are encoded once while
the output retains one row per manifest entry.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


EXPECTED_EMBEDDING_DIM = 512
OFFICIAL_MODEL_NAME = "ViT-B/16"


@dataclass(frozen=True)
class ManifestRow:
    row: int
    pair_id: str
    voice_path: str
    face_path: str
    identity: str | None
    class_index: int | None


def read_manifest(path: Path) -> tuple[list[ManifestRow], bool]:
    rows: list[ManifestRow] = []
    is_training: bool | None = None
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) not in {3, 6}:
                raise ValueError(
                    f"{path}:{line_number} has {len(fields)} fields; expected 3 or 6"
                )
            row_is_training = len(fields) == 6
            if is_training is None:
                is_training = row_is_training
            elif is_training != row_is_training:
                raise ValueError("Training and evaluation rows cannot share one manifest")
            if row_is_training and fields[1] != "1":
                raise ValueError(
                    f"Training row {line_number} is not a positive aligned pair"
                )
            rows.append(
                ManifestRow(
                    row=len(rows),
                    pair_id=fields[0],
                    voice_path=fields[2] if row_is_training else fields[1],
                    face_path=fields[3] if row_is_training else fields[2],
                    identity=fields[4] if row_is_training else None,
                    class_index=int(fields[5]) if row_is_training else None,
                )
            )
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    if len({row.pair_id for row in rows}) != len(rows):
        raise ValueError(f"Duplicate pair IDs in {path}")
    return rows, bool(is_training)


def resolve_image_paths(rows: list[ManifestRow], image_root: Path) -> list[Path]:
    paths = [image_root / row.face_path for row in rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} images; first: {missing[0]}")
    return paths


def content_hash(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def hash_image_paths(paths: list[Path]) -> tuple[list[str], dict[str, Path]]:
    row_hashes: list[str] = []
    representatives: dict[str, Path] = {}
    for path in tqdm(paths, desc="Hashing images", ncols=80):
        digest = content_hash(path)
        row_hashes.append(digest)
        representatives.setdefault(digest, path)
    return row_hashes, representatives


class UniqueImageDataset(Dataset):
    def __init__(self, items: list[tuple[str, Path]], preprocess: Any) -> None:
        self.items = items
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        digest, path = self.items[index]
        try:
            with Image.open(path) as image:
                tensor = self.preprocess(image.convert("RGB"))
        except Exception as error:
            raise RuntimeError(f"Failed to decode/preprocess {path}: {error}") from error
        return digest, tensor


def extract_unique_embeddings(
    model: Any,
    preprocess: Any,
    representatives: dict[str, Path],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    use_fp16: bool,
) -> dict[str, np.ndarray]:
    dataset = UniqueImageDataset(list(representatives.items()), preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    embeddings: dict[str, np.ndarray] = {}
    for digests, images in tqdm(loader, desc="Encoding unique faces", ncols=80):
        images = images.to(device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16 and device.type == "cuda",
        ):
            batch_embeddings = model.encode_image(images)
        batch_embeddings = F.normalize(batch_embeddings.float(), p=2, dim=1).cpu()
        if batch_embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EXPECTED_EMBEDDING_DIM}-D FaRL embeddings, "
                f"received {batch_embeddings.shape[1]}"
            )
        for digest, embedding in zip(digests, batch_embeddings.numpy()):
            embeddings[digest] = embedding.astype(np.float32)
    if set(embeddings) != set(representatives):
        raise RuntimeError("Not every unique image received a FaRL embedding")
    return embeddings


def write_outputs(
    rows: list[ManifestRow],
    row_hashes: list[str],
    embeddings: dict[str, np.ndarray],
    output_csv: Path,
    rowmap_csv: Path,
    is_training: bool,
) -> np.ndarray:
    matrix = np.stack([embeddings[digest] for digest in row_hashes])
    if matrix.shape != (len(rows), EXPECTED_EMBEDDING_DIM):
        raise RuntimeError(f"Unexpected output matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite FaRL embedding generated")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output = matrix
    if is_training:
        labels = np.asarray([row.class_index for row in rows], dtype=np.int64)
        output = np.column_stack((matrix, labels))
    np.savetxt(output_csv, output, delimiter=",", fmt="%.8g")

    rowmap_csv.parent.mkdir(parents=True, exist_ok=True)
    with rowmap_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["row", "pair_id", "face_path", "voice_path", "sha256"]
        if is_training:
            header.extend(["identity", "class_index"])
        writer.writerow(header)
        for row, digest in zip(rows, row_hashes):
            values: list[object] = [
                row.row, row.pair_id, row.face_path, row.voice_path, digest
            ]
            if is_training:
                values.extend([row.identity, row.class_index])
            writer.writerow(values)
    return matrix


def load_farl_model(clip_module: Any, checkpoint: Path, device: torch.device):
    """Follow the official FaRL FaceCLIP loading recipe exactly."""
    model, preprocess = clip_module.load(OFFICIAL_MODEL_NAME, device="cpu")
    checkpoint_object = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint_object, dict) or "state_dict" not in checkpoint_object:
        raise ValueError("FaRL checkpoint must contain a state_dict")
    load_result = model.load_state_dict(checkpoint_object["state_dict"], strict=False)
    # The official recipe uses strict=False. Record the result, but reject a
    # checkpoint that failed to load substantive parameters.
    state_keys = set(checkpoint_object["state_dict"])
    loaded_keys = state_keys - set(load_result.unexpected_keys)
    if len(loaded_keys) < 100:
        raise RuntimeError(
            f"FaRL checkpoint loaded only {len(loaded_keys)} parameter/buffer keys"
        )
    model = model.to(device).eval()
    return model, preprocess, list(load_result.missing_keys), list(load_result.unexpected_keys)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rowmap-csv", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--limit-rows",
        type=int,
        help="Process only the first N rows for an integration smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("Batch size must be positive and workers non-negative")
    if args.limit_rows is not None and args.limit_rows < 1:
        raise ValueError("--limit-rows must be positive")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"FaRL checkpoint not found: {args.checkpoint}")
    try:
        import clip
    except ImportError as error:
        raise SystemExit(
            "This extractor requires the official OpenAI CLIP package. Install "
            "it in a separate vision-feature environment."
        ) from error

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    rows, is_training = read_manifest(args.manifest)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    paths = resolve_image_paths(rows, args.image_root)
    row_hashes, representatives = hash_image_paths(paths)
    print(
        f"rows={len(rows)} unique_images={len(representatives)} "
        f"reused_rows={len(rows) - len(representatives)} device={device}"
    )
    model, preprocess, missing_keys, unexpected_keys = load_farl_model(
        clip, args.checkpoint, device
    )
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    embeddings = extract_unique_embeddings(
        model,
        preprocess,
        representatives,
        device,
        args.batch_size,
        args.num_workers,
        use_fp16,
    )
    rowmap = args.rowmap_csv or args.output_csv.with_name(
        args.output_csv.stem + "_rowmap.csv"
    )
    matrix = write_outputs(
        rows, row_hashes, embeddings, args.output_csv, rowmap, is_training
    )
    norms = np.linalg.norm(matrix, axis=1)
    report = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "output_csv": str(args.output_csv),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": content_hash(args.checkpoint),
        "model": "FaRL-Base-Patch16-LAIONFace20M-ep16",
        "clip_architecture": OFFICIAL_MODEL_NAME,
        "preprocessing": "official OpenAI CLIP ViT-B/16 preprocessing",
        "alignment_or_detection": False,
        "torch_version": torch.__version__,
        "torchvision_version": __import__("torchvision").__version__,
        "device": str(device),
        "training_manifest": is_training,
        "limited_rows": args.limit_rows,
        "rows": len(rows),
        "unique_image_hashes": len(representatives),
        "reused_rows": len(rows) - len(representatives),
        "embedding_dimension": EXPECTED_EMBEDDING_DIM,
        "batch_size": args.batch_size,
        "fp16": use_fp16,
        "checkpoint_missing_keys": missing_keys,
        "checkpoint_unexpected_keys": unexpected_keys,
        "zero_embeddings": int(np.count_nonzero(norms == 0)),
        "nonfinite_embeddings": int(np.count_nonzero(~np.isfinite(matrix).all(axis=1))),
        "norm_min": float(norms.min()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
    }
    report_path = args.report_json or args.output_csv.with_name(
        args.output_csv.stem + "_report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
