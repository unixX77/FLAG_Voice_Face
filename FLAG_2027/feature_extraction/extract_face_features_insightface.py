#!/usr/bin/env python3
"""Extract row-aligned InsightFace embeddings for FLAG manifests.

The extractor accepts the six-field training manifest and the three-field
development/evaluation manifests. Byte-identical images are encoded once, but
the output always contains one row per manifest entry in the original order.

The default model pack is ``buffalo_l``: SCRFD performs detection and
five-landmark alignment before the w600k_r50 recognizer emits a 512-D vector.
If ordinary detection fails, the image is padded and tried once more. Failures
are fatal by default so a damaged feature file is never used accidentally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm


EXPECTED_EMBEDDING_DIM = 512


@dataclass(frozen=True)
class ManifestRow:
    row: int
    pair_id: str
    voice_path: str
    face_path: str
    identity: str | None
    class_index: int | None


@dataclass(frozen=True)
class DetectionRecord:
    embedding: np.ndarray
    status: str
    faces_detected: int
    detection_score: float | None
    bbox: tuple[float, float, float, float] | None


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
                raise ValueError("Training and evaluation rows cannot share a manifest")
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


def _face_value(face: Any, name: str) -> Any:
    if hasattr(face, name):
        return getattr(face, name)
    return face[name]


def select_primary_face(faces: list[Any], image_shape: tuple[int, ...]) -> Any:
    """Select deterministically: largest area, score, then centrality."""
    if not faces:
        raise ValueError("Cannot select a face from an empty list")
    height, width = image_shape[:2]
    center_x, center_y = width / 2.0, height / 2.0

    def rank(face: Any) -> tuple[float, float, float]:
        x1, y1, x2, y2 = map(float, _face_value(face, "bbox"))
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        face_x, face_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        distance_squared = (face_x - center_x) ** 2 + (face_y - center_y) ** 2
        score = float(_face_value(face, "det_score"))
        return area, score, -distance_squared

    return max(faces, key=rank)


def normalized_embedding(face: Any) -> np.ndarray:
    embedding = _face_value(face, "normed_embedding")
    if embedding is None:
        embedding = np.asarray(_face_value(face, "embedding"), dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Recognizer returned an invalid embedding")
        embedding = embedding / norm
    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if embedding.shape != (EXPECTED_EMBEDDING_DIM,):
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIM}-D face embedding, got {embedding.shape}"
        )
    if not np.isfinite(embedding).all():
        raise ValueError("Recognizer returned a non-finite embedding")
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        raise ValueError("Recognizer returned a zero embedding")
    return (embedding / norm).astype(np.float32)


def pad_image(image: np.ndarray, cv2_module: Any) -> np.ndarray:
    height, width = image.shape[:2]
    pad_y = max(1, height // 4)
    pad_x = max(1, width // 4)
    return cv2_module.copyMakeBorder(
        image,
        pad_y,
        pad_y,
        pad_x,
        pad_x,
        cv2_module.BORDER_REFLECT_101,
    )


def encode_image(
    app: Any,
    image: np.ndarray,
    cv2_module: Any,
    failure_policy: str,
    retry_det_threshold: float,
) -> DetectionRecord:
    attempts = (("detected", image), ("padded_retry", pad_image(image, cv2_module)))
    for status, candidate in attempts:
        faces = app.get(candidate)
        if faces:
            selected = select_primary_face(faces, candidate.shape)
            bbox = tuple(float(value) for value in _face_value(selected, "bbox"))
            return DetectionRecord(
                embedding=normalized_embedding(selected),
                status=status,
                faces_detected=len(faces),
                detection_score=float(_face_value(selected, "det_score")),
                bbox=bbox,
            )
    original_threshold = app.det_model.det_thresh
    try:
        app.det_model.det_thresh = retry_det_threshold
        for suffix, candidate in (
            ("original", image),
            ("padded", attempts[1][1]),
        ):
            faces = app.get(candidate)
            if faces:
                selected = select_primary_face(faces, candidate.shape)
                bbox = tuple(float(value) for value in _face_value(selected, "bbox"))
                return DetectionRecord(
                    embedding=normalized_embedding(selected),
                    status=f"low_threshold_{suffix}",
                    faces_detected=len(faces),
                    detection_score=float(_face_value(selected, "det_score")),
                    bbox=bbox,
                )
    finally:
        app.det_model.det_thresh = original_threshold
    if failure_policy == "center_crop":
        height, width = image.shape[:2]
        side = min(height, width)
        y1 = (height - side) // 2
        x1 = (width - side) // 2
        crop = image[y1 : y1 + side, x1 : x1 + side]
        crop = cv2_module.resize(crop, (112, 112), interpolation=cv2_module.INTER_AREA)
        raw_embedding = app.models["recognition"].get_feat(crop)[0]
        embedding = np.asarray(raw_embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(embedding))
        if embedding.shape != (EXPECTED_EMBEDDING_DIM,) or not np.isfinite(norm) or norm <= 0:
            raise ValueError("Direct center-crop recognition returned an invalid embedding")
        return DetectionRecord(
            embedding=embedding / norm,
            status="center_crop_fallback",
            faces_detected=0,
            detection_score=None,
            bbox=None,
        )
    if failure_policy == "zero":
        return DetectionRecord(
            embedding=np.zeros(EXPECTED_EMBEDDING_DIM, dtype=np.float32),
            status="zero_fallback",
            faces_detected=0,
            detection_score=None,
            bbox=None,
        )
    raise RuntimeError("No face detected in the original or padded image")


def extract_unique_embeddings(
    app: Any,
    representative_paths: dict[str, Path],
    cv2_module: Any,
    failure_policy: str,
    retry_det_threshold: float,
) -> tuple[dict[str, DetectionRecord], dict[str, str]]:
    records: dict[str, DetectionRecord] = {}
    failures: dict[str, str] = {}
    for digest, path in tqdm(
        representative_paths.items(), desc="Encoding unique faces", ncols=80
    ):
        image = cv2_module.imread(str(path), cv2_module.IMREAD_COLOR)
        if image is None:
            message = "OpenCV could not decode the image"
            if failure_policy == "error":
                failures[str(path)] = message
                continue
            records[digest] = DetectionRecord(
                np.zeros(EXPECTED_EMBEDDING_DIM, dtype=np.float32),
                "zero_fallback",
                0,
                None,
                None,
            )
            failures[str(path)] = message
            continue
        try:
            records[digest] = encode_image(
                app,
                image,
                cv2_module,
                failure_policy,
                retry_det_threshold,
            )
        except Exception as error:
            failures[str(path)] = str(error)
    if failures and failure_policy == "error":
        examples = "; ".join(
            f"{path}: {reason}" for path, reason in list(failures.items())[:3]
        )
        raise RuntimeError(
            f"Failed to encode {len(failures)} unique images. Examples: {examples}. "
            "Fix the inputs or deliberately use --failure-policy zero."
        )
    if set(records) != set(representative_paths):
        raise RuntimeError("Not every unique image received an embedding")
    return records, failures


def write_outputs(
    rows: list[ManifestRow],
    row_hashes: list[str],
    records: dict[str, DetectionRecord],
    output_csv: Path,
    rowmap_csv: Path,
    is_training: bool,
) -> np.ndarray:
    matrix = np.stack([records[digest].embedding for digest in row_hashes])
    if matrix.shape != (len(rows), EXPECTED_EMBEDDING_DIM):
        raise RuntimeError(f"Unexpected output matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite InsightFace embedding generated")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output: np.ndarray
    if is_training:
        labels = np.asarray([row.class_index for row in rows], dtype=np.int64)
        output = np.column_stack((matrix, labels))
    else:
        output = matrix
    np.savetxt(output_csv, output, delimiter=",", fmt="%.8g")

    rowmap_csv.parent.mkdir(parents=True, exist_ok=True)
    with rowmap_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        header = [
            "row", "pair_id", "face_path", "voice_path", "sha256", "status",
            "faces_detected", "detection_score", "bbox_x1", "bbox_y1", "bbox_x2",
            "bbox_y2",
        ]
        if is_training:
            header.extend(["identity", "class_index"])
        writer.writerow(header)
        for row, digest in zip(rows, row_hashes):
            record = records[digest]
            bbox: list[object] = list(record.bbox) if record.bbox else ["", "", "", ""]
            values: list[object] = [
                row.row,
                row.pair_id,
                row.face_path,
                row.voice_path,
                digest,
                record.status,
                record.faces_detected,
                "" if record.detection_score is None else record.detection_score,
                *bbox,
            ]
            if is_training:
                values.extend([row.identity, row.class_index])
            writer.writerow(values)
    return matrix


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--rowmap-csv", type=Path)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--model", default="buffalo_l")
    parser.add_argument(
        "--model-root", type=Path, default=Path("~/.insightface").expanduser()
    )
    parser.add_argument("--det-size", type=int, default=640)
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument(
        "--retry-det-threshold",
        type=float,
        default=0.3,
        help="Lower threshold used only after normal and padded detection fail",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--failure-policy",
        choices=("error", "center_crop", "zero"),
        default="error",
        help=(
            "error protects experiments from silent invalid rows; center_crop "
            "uses direct recognition only after all detection retries; zero is diagnostic"
        ),
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        help="Process only the first N rows for an integration smoke test",
    )
    return parser.parse_args()


def model_file_hashes(model_root: Path, model_name: str) -> dict[str, str]:
    model_dir = model_root / "models" / model_name
    if not model_dir.is_dir():
        return {}
    return {
        str(path.relative_to(model_dir)): content_hash(path)
        for path in sorted(model_dir.rglob("*.onnx"))
    }


def active_model_providers(app: Any) -> dict[str, list[str]]:
    """Return the providers actually attached to every loaded ONNX session."""
    active: dict[str, list[str]] = {}
    for task, model in app.models.items():
        session = getattr(model, "session", None)
        if session is not None:
            active[str(task)] = list(session.get_providers())
    return active


def main() -> None:
    args = parse_arguments()
    if args.det_size < 1:
        raise ValueError("--det-size must be positive")
    if not 0 <= args.det_threshold <= 1:
        raise ValueError("--det-threshold must be in [0, 1]")
    if not 0 <= args.retry_det_threshold <= args.det_threshold:
        raise ValueError(
            "--retry-det-threshold must be in [0, --det-threshold]"
        )
    if args.limit_rows is not None and args.limit_rows < 1:
        raise ValueError("--limit-rows must be positive")
    try:
        import cv2
        import insightface
        import onnxruntime
        from insightface.app import FaceAnalysis
    except ImportError as error:
        raise SystemExit(
            "This extractor requires insightface, onnxruntime-gpu, and opencv-python. "
            "Install them in a separate face-feature environment."
        ) from error

    # Recent ONNX Runtime wheels can install CUDA/cuDNN as Python packages.
    # Preload those libraries before InsightFace creates any inference session.
    if hasattr(onnxruntime, "preload_dlls"):
        onnxruntime.preload_dlls(directory="")

    available_providers = onnxruntime.get_available_providers()
    cuda_available = "CUDAExecutionProvider" in available_providers
    device = args.device
    if device == "auto":
        device = "cuda" if cuda_available else "cpu"
    if device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was requested, but ONNX Runtime has no CUDAExecutionProvider"
        )
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )

    rows, is_training = read_manifest(args.manifest)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    paths = resolve_image_paths(rows, args.image_root)
    row_hashes, representatives = hash_image_paths(paths)
    print(
        f"rows={len(rows)} unique_images={len(representatives)} "
        f"reused_rows={len(rows) - len(representatives)} device={device}"
    )

    app = FaceAnalysis(
        name=args.model,
        root=str(args.model_root),
        allowed_modules=["detection", "recognition"],
        providers=providers,
    )
    app.prepare(
        ctx_id=0 if device == "cuda" else -1,
        det_thresh=args.det_threshold,
        det_size=(args.det_size, args.det_size),
    )
    active_providers = active_model_providers(app)
    if device == "cuda":
        cpu_fallbacks = [
            task
            for task, task_providers in active_providers.items()
            if not task_providers or task_providers[0] != "CUDAExecutionProvider"
        ]
        if cpu_fallbacks:
            raise RuntimeError(
                "CUDA was requested, but these InsightFace sessions fell back to "
                f"CPU: {cpu_fallbacks}. Active providers: {active_providers}"
            )
    records, failures = extract_unique_embeddings(
        app, representatives, cv2, args.failure_policy, args.retry_det_threshold
    )
    rowmap = args.rowmap_csv or args.output_csv.with_name(
        args.output_csv.stem + "_rowmap.csv"
    )
    matrix = write_outputs(
        rows, row_hashes, records, args.output_csv, rowmap, is_training
    )
    norms = np.linalg.norm(matrix, axis=1)
    unique_records = list(records.values())
    scores = [x.detection_score for x in unique_records if x.detection_score is not None]
    report = {
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "output_csv": str(args.output_csv),
        "model": args.model,
        "model_root": str(args.model_root),
        "model_onnx_sha256": model_file_hashes(args.model_root, args.model),
        "insightface_version": insightface.__version__,
        "onnxruntime_version": onnxruntime.__version__,
        "requested_providers": providers,
        "active_model_providers": active_providers,
        "training_manifest": is_training,
        "limited_rows": args.limit_rows,
        "rows": len(rows),
        "unique_image_hashes": len(representatives),
        "reused_rows": len(rows) - len(representatives),
        "embedding_dimension": EXPECTED_EMBEDDING_DIM,
        "det_size": args.det_size,
        "det_threshold": args.det_threshold,
        "retry_det_threshold": args.retry_det_threshold,
        "failure_policy": args.failure_policy,
        "status_counts_unique": dict(sorted(Counter(x.status for x in unique_records).items())),
        "faces_detected_histogram_unique": {
            str(key): value
            for key, value in sorted(Counter(x.faces_detected for x in unique_records).items())
        },
        "failed_unique_images": len(failures),
        "failure_examples": dict(list(failures.items())[:10]),
        "detection_score_min": min(scores) if scores else None,
        "detection_score_mean": float(np.mean(scores)) if scores else None,
        "detection_score_max": max(scores) if scores else None,
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
