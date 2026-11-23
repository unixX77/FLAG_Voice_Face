"""Data loading for identity-disjoint FLAG 2027 validation folds."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class FoldData:
    """Arrays and identity metadata belonging to one train/validation fold.

    Every row in ``train_faces``, ``train_voices``, and ``train_labels``
    describes the same training sample. The two validation feature arrays are
    aligned in the same way. ``validation_original_indices`` preserves each
    validation sample's row number in the complete dataset so that evaluation
    protocols can refer to it without ambiguity.
    """

    train_faces: np.ndarray
    train_voices: np.ndarray
    train_labels: np.ndarray

    validation_faces: np.ndarray
    validation_voices: np.ndarray
    validation_original_indices: np.ndarray

    train_identities: list[str]
    validation_identities: list[str]


def load_feature_matrices(
    face_csv: Path, voice_csv: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load aligned face/voice features and their original identity labels."""
    face_rows = pd.read_csv(face_csv, header=None, dtype=np.float32).to_numpy()
    voice_rows = pd.read_csv(voice_csv, header=None, dtype=np.float32).to_numpy()
    if face_rows.ndim != 2 or face_rows.shape[1] < 2:
        raise ValueError(
            f"Expected face CSV shape (N, feature_dim + 1), found {face_rows.shape}"
        )
    if voice_rows.ndim != 2 or voice_rows.shape[1] < 2:
        raise ValueError(
            f"Expected voice CSV shape (N, feature_dim + 1), found {voice_rows.shape}"
        )
    if len(face_rows) != len(voice_rows):
        raise ValueError(
            f"Face/voice row mismatch: {len(face_rows)} versus {len(voice_rows)}"
        )
    if not np.isfinite(face_rows).all() or not np.isfinite(voice_rows).all():
        raise ValueError("Feature CSVs contain NaN or infinite values")

    face_labels = face_rows[:, -1].astype(np.int64)
    voice_labels = voice_rows[:, -1].astype(np.int64)
    if not np.array_equal(face_labels, voice_labels):
        disagreements = int(np.count_nonzero(face_labels != voice_labels))
        raise ValueError(f"Face and voice labels disagree on {disagreements} rows")
    return (
        np.ascontiguousarray(face_rows[:, :-1], dtype=np.float32),
        np.ascontiguousarray(voice_rows[:, :-1], dtype=np.float32),
        face_labels,
    )


def load_manifest_identities(manifest: Path) -> np.ndarray:
    """Return the identity name for every feature row in manifest order."""
    identities: list[str] = []
    with manifest.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(
                    f"Manifest line {line_number} has {len(fields)} fields, expected 6"
                )
            identities.append(fields[4])
    if not identities:
        raise ValueError(f"Manifest is empty: {manifest}")
    return np.asarray(identities, dtype=str)


def load_fold_metadata(metadata_path: Path) -> dict:
    """Read and minimally validate one fold's metadata file."""
    metadata = json.loads(metadata_path.read_text())
    required = {
        "fold",
        "train_identities",
        "validation_identities",
        "train_identity_label_map",
        "train_rows",
        "validation_rows",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"Missing metadata keys in {metadata_path}: {missing}")
    return metadata


def prepare_fold_data(
    face_csv: Path,
    voice_csv: Path,
    manifest: Path,
    fold_dir: Path,
) -> FoldData:
    """Load one fold and remap its training identities to contiguous classes."""
    faces, voices, original_labels = load_feature_matrices(face_csv, voice_csv)
    identities = load_manifest_identities(manifest)
    if len(identities) != len(faces):
        raise ValueError(
            f"Manifest/features row mismatch: {len(identities)} versus {len(faces)}"
        )

    # The manifest label is checked as well as the identity name, catching an
    # accidentally reordered manifest even when all identity names still exist.
    manifest_labels = []
    with manifest.open() as handle:
        for line in handle:
            manifest_labels.append(int(line.split()[5]))
    if not np.array_equal(original_labels, np.asarray(manifest_labels, dtype=np.int64)):
        raise ValueError("Feature labels do not align row-for-row with the manifest")

    metadata = load_fold_metadata(fold_dir / "metadata.json")
    train_indices = np.load(fold_dir / "train_indices.npy").astype(np.int64)
    validation_indices = np.load(fold_dir / "validation_indices.npy").astype(np.int64)
    if train_indices.ndim != 1 or validation_indices.ndim != 1:
        raise ValueError("Fold index arrays must be one-dimensional")
    if len(train_indices) + len(validation_indices) != len(faces):
        raise ValueError("Fold index row counts do not cover the complete dataset")
    all_indices = np.concatenate((train_indices, validation_indices))
    if len(all_indices) and (all_indices.min() < 0 or all_indices.max() >= len(faces)):
        raise ValueError("Fold indices contain an out-of-range dataset row")
    if len(np.unique(all_indices)) != len(faces):
        raise ValueError("Fold indices must partition every dataset row exactly once")
    if set(train_indices).intersection(validation_indices):
        raise ValueError("Training and validation indices overlap")

    label_map = {
        identity: int(label)
        for identity, label in metadata["train_identity_label_map"].items()
    }
    if sorted(label_map.values()) != list(range(len(label_map))):
        raise ValueError("Fold-local training labels are not contiguous from zero")
    train_labels = np.asarray(
        [label_map[identity] for identity in identities[train_indices]], dtype=np.int64
    )

    train_identity_set = set(identities[train_indices])
    validation_identity_set = set(identities[validation_indices])
    if train_identity_set & validation_identity_set:
        raise ValueError("Identity leakage between training and validation")
    if train_identity_set != set(metadata["train_identities"]):
        raise ValueError("Training indices disagree with metadata identities")
    if validation_identity_set != set(metadata["validation_identities"]):
        raise ValueError("Validation indices disagree with metadata identities")
    if len(train_indices) != int(metadata["train_rows"]):
        raise ValueError("Training row count disagrees with metadata")
    if len(validation_indices) != int(metadata["validation_rows"]):
        raise ValueError("Validation row count disagrees with metadata")

    return FoldData(
        train_faces=np.ascontiguousarray(faces[train_indices]),
        train_voices=np.ascontiguousarray(voices[train_indices]),
        train_labels=train_labels,
        validation_faces=np.ascontiguousarray(faces[validation_indices]),
        validation_voices=np.ascontiguousarray(voices[validation_indices]),
        validation_original_indices=validation_indices,
        train_identities=list(metadata["train_identities"]),
        validation_identities=list(metadata["validation_identities"]),
    )


class FaceVoiceFeatureDataset(Dataset):
    """PyTorch dataset of aligned face, voice, and identity-label rows."""

    def __init__(self, faces: np.ndarray, voices: np.ndarray, labels: np.ndarray):
        if not (len(faces) == len(voices) == len(labels)):
            raise ValueError("Face, voice, and label row counts must match")
        self.faces = torch.from_numpy(np.ascontiguousarray(faces, dtype=np.float32))
        self.voices = torch.from_numpy(np.ascontiguousarray(voices, dtype=np.float32))
        self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.faces[index], self.voices[index], self.labels[index]


def create_training_loader(
    fold_data: FoldData,
    batch_size: int,
    seed: int,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Construct a reproducibly shuffled loader for one fold's training rows."""
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for pair-based OPL training")
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = FaceVoiceFeatureDataset(
        fold_data.train_faces, fold_data.train_voices, fold_data.train_labels
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
