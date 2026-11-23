"""Evaluate model embeddings on local face--voice verification protocols."""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import compute_eer


@dataclass
class Protocol:
    """One collection of face--voice verification trials.

    The four arrays are positionally aligned: one position contains a trial's
    identifier, original face-row index, original voice-row index, and binary
    match label respectively.
    """

    pair_ids: np.ndarray
    face_indices: np.ndarray
    voice_indices: np.ndarray
    labels: np.ndarray


@dataclass
class ValidationResult:
    """EER and AUC measurements produced by one validation run."""

    standard_eer: float
    standard_auc: float
    gender_eer: float
    gender_auc: float
    mean_eer: float


def _read_csv_by_id(path: Path, value_columns: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    values: dict[str, tuple[str, ...]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"pair_id", *value_columns}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            pair_id = row["pair_id"]
            if pair_id in values:
                raise ValueError(f"Duplicate pair_id {pair_id} in {path}")
            values[pair_id] = tuple(row[column] for column in value_columns)
    return values


def load_protocol(pair_path: Path, ground_truth_path: Path) -> Protocol:
    """Load trials, joining their separately stored labels by pair ID."""
    pairs = _read_csv_by_id(pair_path, ("face_index", "voice_index"))
    truth = _read_csv_by_id(ground_truth_path, ("label",))
    if set(pairs) != set(truth):
        raise ValueError("Pair IDs differ between protocol and ground truth")
    pair_ids = np.asarray(list(pairs), dtype=str)
    labels = np.asarray([int(truth[pair_id][0]) for pair_id in pair_ids], dtype=np.int64)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Protocol must contain both binary classes")
    return Protocol(
        pair_ids=pair_ids,
        face_indices=np.asarray([int(pairs[x][0]) for x in pair_ids], dtype=np.int64),
        voice_indices=np.asarray([int(pairs[x][1]) for x in pair_ids], dtype=np.int64),
        labels=labels,
    )


def extract_validation_embeddings(
    model: torch.nn.Module,
    validation_faces: np.ndarray,
    validation_voices: np.ndarray,
    validation_original_indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Embed each validation sample once and map original row IDs to positions."""
    if not (
        len(validation_faces)
        == len(validation_voices)
        == len(validation_original_indices)
    ):
        raise ValueError("Validation arrays and original indices must align")
    if len(np.unique(validation_original_indices)) != len(validation_original_indices):
        raise ValueError("Validation original indices must be unique")

    face_parts: list[np.ndarray] = []
    voice_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(validation_faces), batch_size):
            stop = min(start + batch_size, len(validation_faces))
            face_batch = torch.from_numpy(validation_faces[start:stop]).to(device)
            voice_batch = torch.from_numpy(validation_voices[start:stop]).to(device)
            face_embedding = model.face_branch(face_batch)
            voice_embedding = model.voice_branch(voice_batch)
            face_parts.append(face_embedding.cpu().numpy())
            voice_parts.append(voice_embedding.cpu().numpy())
    row_to_position = {
        int(original): position
        for position, original in enumerate(validation_original_indices)
    }
    return np.concatenate(face_parts), np.concatenate(voice_parts), row_to_position


def score_protocol(
    protocol: Protocol,
    face_embeddings: np.ndarray,
    voice_embeddings: np.ndarray,
    row_to_position: dict[int, int],
    score_mode: str = "negative_euclidean",
) -> np.ndarray:
    """Score protocol pairs; in every mode a larger score means a better match."""
    try:
        face_positions = np.asarray(
            [row_to_position[int(index)] for index in protocol.face_indices]
        )
        voice_positions = np.asarray(
            [row_to_position[int(index)] for index in protocol.voice_indices]
        )
    except KeyError as error:
        raise ValueError(f"Protocol refers to non-validation row {error.args[0]}") from error
    face = face_embeddings[face_positions]
    voice = voice_embeddings[voice_positions]
    if score_mode == "negative_euclidean":
        scores = -np.linalg.norm(face - voice, axis=1)
    elif score_mode == "cosine":
        face = F.normalize(torch.from_numpy(face), dim=1).numpy()
        voice = F.normalize(torch.from_numpy(voice), dim=1).numpy()
        scores = np.sum(face * voice, axis=1)
    else:
        raise ValueError(f"Unknown score mode: {score_mode}")
    if not np.isfinite(scores).all():
        raise ValueError("Protocol scores contain NaN or infinity")
    return scores.astype(np.float64, copy=False)


def evaluate_protocol(protocol: Protocol, scores: np.ndarray) -> tuple[float, float]:
    """Compute EER and ROC-AUC for already-scored trials."""
    return compute_eer(protocol.labels, scores)


def evaluate_fold(
    model: torch.nn.Module,
    validation_faces: np.ndarray,
    validation_voices: np.ndarray,
    validation_original_indices: np.ndarray,
    fold_dir: Path,
    device: torch.device,
    batch_size: int = 512,
    score_mode: str = "negative_euclidean",
) -> ValidationResult:
    """Embed validation data once, then evaluate standard and gender protocols."""
    face_embeddings, voice_embeddings, row_to_position = extract_validation_embeddings(
        model,
        validation_faces,
        validation_voices,
        validation_original_indices,
        device,
        batch_size,
    )
    results: dict[str, tuple[float, float]] = {}
    for track in ("standard", "gender"):
        protocol = load_protocol(
            fold_dir / f"{track}_pairs.csv",
            fold_dir / f"{track}_ground_truth.csv",
        )
        if getattr(model, "geometry", "sphere") == "hyperbolic":
            # Score with the model's own geodesic distance, not cosine.
            face = face_embeddings[[row_to_position[int(i)] for i in protocol.face_indices]]
            voice = voice_embeddings[[row_to_position[int(i)] for i in protocol.voice_indices]]
            with torch.inference_mode():
                scores = model.pair_similarity(
                    torch.from_numpy(face), torch.from_numpy(voice)
                ).numpy().astype(np.float64)
        else:
            scores = score_protocol(
                protocol,
                face_embeddings,
                voice_embeddings,
                row_to_position,
                score_mode,
            )
        results[track] = evaluate_protocol(protocol, scores)
    standard_eer, standard_auc = results["standard"]
    gender_eer, gender_auc = results["gender"]
    return ValidationResult(
        standard_eer=standard_eer,
        standard_auc=standard_auc,
        gender_eer=gender_eer,
        gender_auc=gender_auc,
        mean_eer=(standard_eer + gender_eer) / 2.0,
    )
