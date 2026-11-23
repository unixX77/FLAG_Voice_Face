"""Evaluate a pairwise verifier on identity-disjoint validation protocols."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .metrics import compute_eer
from .model_evaluation import (
    ValidationResult,
    extract_validation_embeddings,
    load_protocol,
)


def score_protocol_with_verifier(
    model,
    protocol,
    face_embeddings: np.ndarray,
    voice_embeddings: np.ndarray,
    row_to_position: dict[int, int],
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    try:
        face_positions = np.asarray(
            [row_to_position[int(index)] for index in protocol.face_indices],
            dtype=np.int64,
        )
        voice_positions = np.asarray(
            [row_to_position[int(index)] for index in protocol.voice_indices],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(
            f"Protocol refers to non-validation row {error.args[0]}"
        ) from error

    scores: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(protocol.labels), batch_size):
            stop = min(start + batch_size, len(protocol.labels))
            faces = torch.from_numpy(face_embeddings[face_positions[start:stop]]).to(
                device
            )
            voices = torch.from_numpy(
                voice_embeddings[voice_positions[start:stop]]
            ).to(device)
            scores.append(model.score_embeddings(faces, voices).cpu().numpy())
    result = np.concatenate(scores).astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ValueError("Pairwise verifier produced non-finite scores")
    return result


def evaluate_pairwise_fold(
    model,
    validation_faces: np.ndarray,
    validation_voices: np.ndarray,
    validation_original_indices: np.ndarray,
    fold_dir: Path,
    device: torch.device,
    embed_batch_size: int = 512,
    score_batch_size: int = 1024,
) -> tuple[ValidationResult, ValidationResult]:
    """Return cosine and learned-verifier metrics on the same trial sets."""
    face_embeddings, voice_embeddings, row_to_position = extract_validation_embeddings(
        model,
        validation_faces,
        validation_voices,
        validation_original_indices,
        device,
        embed_batch_size,
    )
    result_by_mode: dict[str, dict[str, tuple[float, float]]] = {
        "cosine": {},
        "pairwise": {},
    }
    for track in ("standard", "gender"):
        protocol = load_protocol(
            fold_dir / f"{track}_pairs.csv",
            fold_dir / f"{track}_ground_truth.csv",
        )
        fpos = np.asarray(
            [row_to_position[int(index)] for index in protocol.face_indices],
            dtype=np.int64,
        )
        vpos = np.asarray(
            [row_to_position[int(index)] for index in protocol.voice_indices],
            dtype=np.int64,
        )
        cosine = np.sum(face_embeddings[fpos] * voice_embeddings[vpos], axis=1)
        verifier = score_protocol_with_verifier(
            model,
            protocol,
            face_embeddings,
            voice_embeddings,
            row_to_position,
            device,
            score_batch_size,
        )
        result_by_mode["cosine"][track] = compute_eer(protocol.labels, cosine)
        result_by_mode["pairwise"][track] = compute_eer(protocol.labels, verifier)

    results: list[ValidationResult] = []
    for mode in ("cosine", "pairwise"):
        standard_eer, standard_auc = result_by_mode[mode]["standard"]
        gender_eer, gender_auc = result_by_mode[mode]["gender"]
        results.append(
            ValidationResult(
                standard_eer=standard_eer,
                standard_auc=standard_auc,
                gender_eer=gender_eer,
                gender_auc=gender_auc,
                mean_eer=(standard_eer + gender_eer) / 2.0,
            )
        )
    return results[0], results[1]
