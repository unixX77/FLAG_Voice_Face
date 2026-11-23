#!/usr/bin/env python3
"""Extract deterministic WavLM Base+ speaker embeddings in manifest order.

The script accepts both six-field FLAG training manifests and three-field
development protocols. Byte-identical audio is encoded once, while output rows
still follow the input manifest exactly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import wave
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


TARGET_SAMPLE_RATE = 16_000
EXPECTED_EMBEDDING_DIM = 512


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


def resolve_audio_paths(rows: list[ManifestRow], audio_root: Path) -> list[Path]:
    paths = [audio_root / row.voice_path for row in rows]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} audio files; first: {missing[0]}")
    return paths


def content_hash(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def load_pcm16_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != TARGET_SAMPLE_RATE:
            raise ValueError(f"{path} is not {TARGET_SAMPLE_RATE} Hz")
        if handle.getnchannels() != 1:
            raise ValueError(f"{path} is not mono")
        if handle.getsampwidth() != 2 or handle.getcomptype() != "NONE":
            raise ValueError(f"{path} is not uncompressed 16-bit PCM")
        samples = np.frombuffer(
            handle.readframes(handle.getnframes()), dtype="<i2"
        ).astype(np.float32)
    if len(samples) == 0:
        raise ValueError(f"Empty waveform: {path}")
    return np.ascontiguousarray(samples / 32768.0)


def split_chunks(
    waveform: np.ndarray, chunk_samples: int, overlap_samples: int
) -> list[np.ndarray]:
    if len(waveform) <= chunk_samples:
        return [waveform]
    stride = chunk_samples - overlap_samples
    starts = list(range(0, len(waveform) - chunk_samples + 1, stride))
    final_start = len(waveform) - chunk_samples
    if starts[-1] != final_start:
        starts.append(final_start)
    return [waveform[start : start + chunk_samples] for start in starts]


def hash_audio_paths(paths: list[Path]) -> tuple[list[str], dict[str, Path]]:
    row_hashes: list[str] = []
    representative: dict[str, Path] = {}
    for path in tqdm(paths, desc="Hashing audio", ncols=80):
        digest = content_hash(path)
        row_hashes.append(digest)
        representative.setdefault(digest, path)
    return row_hashes, representative


def extract_unique_embeddings(
    model,
    feature_extractor,
    representative_paths: dict[str, Path],
    device: torch.device,
    batch_size: int,
    chunk_seconds: float,
    overlap_seconds: float,
    use_fp16: bool,
) -> tuple[dict[str, np.ndarray], dict[str, int], Counter[int]]:
    chunk_samples = int(round(chunk_seconds * TARGET_SAMPLE_RATE))
    overlap_samples = int(round(overlap_seconds * TARGET_SAMPLE_RATE))
    if chunk_samples <= 0 or overlap_samples < 0 or overlap_samples >= chunk_samples:
        raise ValueError("Require chunk_seconds > overlap_seconds >= 0")

    sums: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    chunk_histogram: Counter[int] = Counter()
    pending_waveforms: list[np.ndarray] = []
    pending_owners: list[str] = []

    def flush() -> None:
        if not pending_waveforms:
            return
        inputs = feature_extractor(
            pending_waveforms,
            sampling_rate=TARGET_SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16 and device.type == "cuda",
        ):
            embeddings = model(**inputs).embeddings
            embeddings = F.normalize(embeddings.float(), dim=1).cpu()
        if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
            raise ValueError(
                f"Expected {EXPECTED_EMBEDDING_DIM}-D WavLM embeddings, "
                f"received {embeddings.shape[1]}"
            )
        for owner, embedding in zip(pending_owners, embeddings):
            if owner not in sums:
                sums[owner] = embedding.clone()
            else:
                sums[owner].add_(embedding)
            counts[owner] += 1
        pending_waveforms.clear()
        pending_owners.clear()

    for digest, path in tqdm(
        representative_paths.items(), desc="Encoding unique audio", ncols=80
    ):
        chunks = split_chunks(load_pcm16_wav(path), chunk_samples, overlap_samples)
        chunk_histogram[len(chunks)] += 1
        for chunk in chunks:
            pending_waveforms.append(chunk)
            pending_owners.append(digest)
            if len(pending_waveforms) >= batch_size:
                flush()
    flush()

    embeddings = {
        digest: F.normalize(total, dim=0).numpy().astype(np.float32)
        for digest, total in sums.items()
    }
    if set(embeddings) != set(representative_paths):
        raise RuntimeError("Not every unique audio recording received an embedding")
    return embeddings, dict(counts), chunk_histogram


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
        raise RuntimeError("Non-finite WavLM embedding generated")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if is_training:
        labels = np.asarray([row.class_index for row in rows], dtype=np.int64)
        output = np.column_stack((matrix, labels))
    else:
        output = matrix
    np.savetxt(output_csv, output, delimiter=",", fmt="%.8g")

    rowmap_csv.parent.mkdir(parents=True, exist_ok=True)
    with rowmap_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = ["row", "pair_id", "voice_path", "face_path", "sha256"]
        if is_training:
            header.extend(["identity", "class_index"])
        writer.writerow(header)
        for row, digest in zip(rows, row_hashes):
            values: list[object] = [
                row.row, row.pair_id, row.voice_path, row.face_path, digest
            ]
            if is_training:
                values.extend([row.identity, row.class_index])
            writer.writerow(values)
    return matrix


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rowmap-csv", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument(
        "--revision",
        required=True,
        help="Pin a Hugging Face commit hash for reproducible extraction",
    )
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=8)
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
    try:
        from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
    except ImportError as error:
        raise SystemExit(
            "This extractor requires transformers. Install it in a separate audio "
            "environment; do not alter the working flag2027 training environment."
        ) from error
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    rows, is_training = read_manifest(args.manifest)
    if args.limit_rows is not None:
        if args.limit_rows < 1:
            raise ValueError("--limit-rows must be positive")
        rows = rows[: args.limit_rows]
    paths = resolve_audio_paths(rows, args.audio_root)
    row_hashes, representative_paths = hash_audio_paths(paths)

    print(
        f"rows={len(rows)} unique_audio={len(representative_paths)} "
        f"reused_rows={len(rows) - len(representative_paths)} device={device}"
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.model, revision=args.revision
    )
    model = WavLMForXVector.from_pretrained(
        args.model, revision=args.revision
    ).to(device)
    model.eval()
    embeddings, chunks_per_hash, chunk_histogram = extract_unique_embeddings(
        model,
        feature_extractor,
        representative_paths,
        device,
        args.batch_size,
        args.chunk_seconds,
        args.overlap_seconds,
        not args.no_fp16,
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
        "audio_root": str(args.audio_root),
        "output_csv": str(args.output_csv),
        "model": args.model,
        "revision": args.revision,
        "training_manifest": is_training,
        "limited_rows": args.limit_rows,
        "rows": len(rows),
        "unique_audio_hashes": len(representative_paths),
        "reused_rows": len(rows) - len(representative_paths),
        "embedding_dimension": EXPECTED_EMBEDDING_DIM,
        "sample_rate": TARGET_SAMPLE_RATE,
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "batch_size": args.batch_size,
        "fp16": not args.no_fp16 and device.type == "cuda",
        "chunk_count_histogram": {
            str(key): value for key, value in sorted(chunk_histogram.items())
        },
        "total_encoded_chunks": int(sum(chunks_per_hash.values())),
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
