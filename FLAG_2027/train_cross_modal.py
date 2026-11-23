"""Train the proposed shared-space face--voice architecture on one fold."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:  # Package import: python -m FLAG_2027.train_cross_modal
    from .cross_modal_model import CrossModalIdentityNetwork
    from .local_validation.cross_modal_losses import (
        fused_orthogonal_projection_loss,
        paired_cosine_alignment_loss,
        symmetric_cross_modal_contrastive_loss,
        symmetric_same_gender_hard_negative_loss,
        symmetric_same_gender_supcon_loss,
    )
    from .local_validation.fold_dataset import FoldData, prepare_fold_data
    from .local_validation.identity_sampler import IdentityBalancedBatchSampler
    from .local_validation.model_evaluation import ValidationResult, evaluate_fold
    from .trait_heads import LanguageAdversary, TraitHeads, cross_lingual_alignment
    from .gamp_vf.prototypes import prototype_contrastive_loss
    from .gamp_vf.sampling import GenderAwareHardNegativeSampler, identity_cross_modal_similarity
except ImportError:  # Script execution: python FLAG_2027/train_cross_modal.py
    from cross_modal_model import CrossModalIdentityNetwork
    from local_validation.cross_modal_losses import (
        fused_orthogonal_projection_loss,
        paired_cosine_alignment_loss,
        symmetric_cross_modal_contrastive_loss,
        symmetric_same_gender_hard_negative_loss,
        symmetric_same_gender_supcon_loss,
    )
    from local_validation.fold_dataset import FoldData, prepare_fold_data
    from local_validation.identity_sampler import IdentityBalancedBatchSampler
    from local_validation.model_evaluation import ValidationResult, evaluate_fold
    from trait_heads import LanguageAdversary, TraitHeads, cross_lingual_alignment
    from gamp_vf.prototypes import prototype_contrastive_loss
    from gamp_vf.sampling import GenderAwareHardNegativeSampler, identity_cross_modal_similarity

LANGUAGES = {"English": 0, "Urdu": 1, "German": 2}  # training languages only; no Bengali anywhere


class CrossModalFeatureDataset(Dataset):
    """Aligned fixed features with fold-local identity and binary gender labels."""

    def __init__(
        self,
        fold_data: FoldData,
        genders: np.ndarray,
        voice_views: np.ndarray | None = None,
        clean_view_probability: float = 1.0,
        aux: np.ndarray | None = None,
    ) -> None:
        if len(genders) != len(fold_data.train_labels):
            raise ValueError("Training samples and gender labels must align")
        # Optional per-row auxiliary targets (age_z, build_z, language id); NaN / -1 = unknown.
        if aux is not None and len(aux) != len(fold_data.train_labels):
            raise ValueError("Auxiliary targets must align with the training rows")
        self.aux = None if aux is None else torch.from_numpy(np.asarray(aux, dtype=np.float32))
        self.faces = torch.from_numpy(fold_data.train_faces)
        self.voices = torch.from_numpy(fold_data.train_voices)
        self.labels = torch.from_numpy(fold_data.train_labels)
        self.genders = torch.from_numpy(np.asarray(genders, dtype=np.int64))
        # Optional augmented voice views (train rows, views, dim); each access
        # returns the clean feature with clean_view_probability, otherwise a
        # uniformly drawn augmented view.
        if voice_views is not None and voice_views.shape[::2] != self.voices.shape:
            raise ValueError("Voice views must have shape (train rows, views, voice dim)")
        self.voice_views = None if voice_views is None else torch.from_numpy(voice_views)
        self.clean_view_probability = clean_view_probability

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        voice = self.voices[index]
        if self.voice_views is not None and torch.rand(()) >= self.clean_view_probability:
            voice = self.voice_views[index, torch.randint(self.voice_views.shape[1], ())]
        if self.aux is not None:
            return self.faces[index], voice, self.labels[index], self.genders[index], self.aux[index]
        return (
            self.faces[index],
            voice,
            self.labels[index],
            self.genders[index],
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--validation-root", type=Path, default=Path("validation"))
    parser.add_argument(
        "--face-csv",
        type=Path,
        default=Path("train_set/features/faces/train_English_faces.csv"),
    )
    parser.add_argument(
        "--voice-csv",
        type=Path,
        default=Path("train_set/features/voices/train_English_voices.csv"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("train_set/train_set/train_English.txt")
    )
    parser.add_argument(
        "--gender-metadata", type=Path, default=Path("meta_file_train_set.csv")
    )
    parser.add_argument("--save-dir", type=Path, default=Path("runs/cross_modal_v1/fold_0"))
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--samples-per-identity", type=int, default=8)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--gender-balanced", action="store_true")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--arc-scale", type=float, default=30.0)
    parser.add_argument("--arc-margin", type=float, default=0.2)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument(
        "--contrastive-mode",
        choices=("multi_positive", "supcon"),
        default="multi_positive",
        help=(
            "multi_positive scores the positive set collectively (V1); supcon "
            "averages the log-probability of every positive candidate (V4)"
        ),
    )
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument(
        "--orthogonal-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for baseline-style Orthogonal Projection Loss on the "
            "fused face--voice pair embedding; zero preserves V1 behavior"
        ),
    )
    parser.add_argument(
        "--orthogonal-negative-weight",
        type=float,
        default=0.7,
        help="Weight of different-identity absolute similarity inside OPL",
    )
    parser.add_argument("--prototype-weight", type=float, default=0.0)
    parser.add_argument("--prototype-temperature", type=float, default=0.07)
    parser.add_argument("--prototype-momentum", type=float, default=0.9)
    parser.add_argument("--prototype-start-epoch", type=int, default=10)
    parser.add_argument(
        "--same-gender-contrastive-weight",
        type=float,
        default=0.0,
        help="Weight of supervised contrastive loss restricted to same-gender negatives",
    )
    parser.add_argument(
        "--face-hidden-dims",
        type=parse_hidden_dims,
        default=(1024, 512),
        help="Comma-separated face projector hidden widths; 'none' for a linear map",
    )
    parser.add_argument(
        "--voice-hidden-dims",
        type=parse_hidden_dims,
        default=(512, 512),
        help="Comma-separated voice projector hidden widths; 'none' for a linear map",
    )
    parser.add_argument(
        "--hidden-dropout",
        type=float,
        help="Dropout after every hidden layer; default keeps V1's 0.2 then 0.1",
    )
    parser.add_argument(
        "--standardize-inputs",
        action="store_true",
        help="Z-score each input feature using this fold's training rows",
    )
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument(
        "--input-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise added to (standardized) inputs during training",
    )
    parser.add_argument(
        "--voice-views-npy",
        type=Path,
        help="Augmented voice views (all manifest rows, views, dim) used in training only",
    )
    parser.add_argument(
        "--clean-view-probability",
        type=float,
        default=0.5,
        help="Probability of using the clean voice feature when views are given",
    )
    parser.add_argument(
        "--geometry",
        choices=("sphere", "hyperbolic"),
        default="sphere",
        help="sphere: unit embeddings and cosine (V2); hyperbolic: Poincare ball and geodesic distance",
    )
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument(
        "--voice-layers-npy",
        type=Path,
        help="Per-layer WavLM statistics (rows, layers, 2*dim) replacing the voice CSV features",
    )
    parser.add_argument(
        "--voice-layer-features",
        choices=("mean", "meanstd"),
        default="mean",
        help="Use each layer's frame mean, or mean and standard deviation",
    )
    parser.add_argument("--voice-layer-pooling", choices=("mix", "attention"), default="mix")
    parser.add_argument(
        "--clip-radius",
        type=float,
        default=1.0,
        help="Euclidean norm clip before the Poincare exponential map",
    )
    parser.add_argument("--hard-negative-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.2)
    parser.add_argument("--hard-negative-top-k", type=int, default=5)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--extra-face-csv", type=Path, help="external training faces (label = last column)")
    parser.add_argument("--extra-voice-csv", type=Path, help="external training voices, rows aligned with --extra-face-csv")
    parser.add_argument("--extra-manifest", type=Path, help="external training manifest (6 columns), rows aligned")
    parser.add_argument("--extra-gender-metadata", type=Path, help="identity,m/f for the external identities")
    parser.add_argument(
        "--official-share",
        type=float,
        help="With external data: fraction of each batch's identities drawn from the official "
             "identities (default: all identities uniformly)",
    )
    parser.add_argument(
        "--max-train-identities",
        type=int,
        help="Learning-curve option: keep only this many training identities of the fold "
             "(gender-balanced, chosen with the run seed); validation is unchanged",
    )
    # Voice -> traits -> face architecture (all off by default).
    parser.add_argument("--trait-csv", type=Path,
                        help="per-row traits for the official manifest (key,identity,age,build), person-level")
    parser.add_argument("--extra-trait-csv", type=Path, help="per-row traits for the external rows (same columns)")
    parser.add_argument("--trait-weight", type=float, default=0.0, help="weight of the shared trait-head loss")
    parser.add_argument("--trait-targets", default="age,build,gender")
    parser.add_argument("--language-adversary-weight", type=float, default=0.0)
    parser.add_argument("--language-adversary-scale", type=float, default=1.0, help="gradient-reversal scale")
    parser.add_argument("--cross-lingual-weight", type=float, default=0.0,
                        help="pull together voices of one person in two different languages")
    parser.add_argument("--exclude-train-languages", nargs="*", default=[],
                        help="drop external training clips in these languages (unheard-language protocol)")
    parser.add_argument("--holdout-external-prefix",
                        help="hold out the external identities with this prefix whose sorted index %% 5 == fold")
    parser.add_argument("--no-best-checkpoint", action="store_true",
                        help="do not write checkpoint_best.pth.tar (results always use checkpoint_last)")
    # GAMP-VF (all off by default; docs/gamp_vf/02_architecture.md).
    parser.add_argument("--prototype-contrastive-weight", type=float, default=0.0,
                        help="instance-to-identity-prototype InfoNCE across modalities (Stage A/B)")
    parser.add_argument("--prototype-language-balanced", action="store_true",
                        help="voice prototypes average per-language means first (A3)")
    parser.add_argument("--batch-strategy", default="default",
                        choices=("default", "random", "same_gender", "nearest", "same_gender_nearest",
                                 "same_gender_age"),
                        help="identity selection for P x K batches (Stage C); 'default' = existing sampler")
    parser.add_argument("--hard-ratio", type=float, default=0.5, help="share of partners chosen by the strategy")
    parser.add_argument("--hard-top-k", type=int, default=5)
    parser.add_argument("--hard-warmup", type=int, default=5, help="epochs before the first similarity refresh")
    parser.add_argument("--hard-refresh-every", type=int, default=5)
    parser.add_argument("--language-balanced-k", action="store_true",
                        help="split each identity's K samples evenly over its languages")
    parser.add_argument("--exclude-external-identities", type=Path,
                        help="file with external identity names to drop from training (hidden/calibration sets)")
    parser.add_argument("--pseudo-pairs-csv", type=Path,
                        help="pseudo-labelled training rows: face_row,voice_row,pseudo_id,weight,gender,conflict_group")
    # Phase III (docs/phase3): voice block fusion and ECAPA controls (all off by default).
    parser.add_argument("--voice-fusion", choices=("concat", "bottleneck", "block_l2", "residual"), default="concat")
    parser.add_argument("--voice-fusion-split", type=int, default=0,
                        help="number of leading voice dims forming the base block (rest = extra block)")
    parser.add_argument("--voice-fusion-dim", type=int, default=64)
    parser.add_argument("--shuffle-voice-block-start", type=int,
                        help="control E: for TRAINING rows only, replace voice dims [start:] with those of "
                             "another training speaker of the same gender and source (fixed per speaker)")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def holdout_external_identities(names, prefix: str | None, fold: int) -> set[str]:
    """External identities kept out of training in this fold (language-transfer protocol)."""
    if not prefix:
        return set()
    pool = sorted({n for n in names if n.startswith(prefix + "_")})
    return {n for i, n in enumerate(pool) if i % 5 == fold}


def manifest_language(voice_path: str) -> int:
    return LANGUAGES.get(voice_path.split("/")[1], -1)


def read_traits(path: Path) -> dict[str, tuple[float, float]]:
    """key -> (age, build); empty fields become NaN."""
    out = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            out[row["key"]] = tuple(float(row[k]) if row[k] not in ("", "nan") else np.nan for k in ("age", "build"))
    return out


def parse_hidden_dims(text: str) -> tuple[int, ...]:
    if text.strip().lower() in {"", "none"}:
        return ()
    dims = tuple(int(part) for part in text.split(","))
    if any(dim < 1 for dim in dims):
        raise argparse.ArgumentTypeError("Hidden widths must be positive")
    return dims


def validate_arguments(args: argparse.Namespace) -> None:
    nonnegative = (
        "identity_weight",
        "contrastive_weight",
        "pair_weight",
        "orthogonal_weight",
        "prototype_weight",
        "hard_negative_weight",
        "same_gender_contrastive_weight",
    )
    if any(getattr(args, name) < 0 for name in nonnegative):
        raise ValueError("Loss weights must be non-negative")
    if sum(getattr(args, name) for name in nonnegative) == 0:
        raise ValueError("At least one loss weight must be positive")
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Epoch and patience values must be positive")
    if args.prototype_start_epoch < 1:
        raise ValueError("--prototype-start-epoch must be positive")
    if not 0 <= args.clean_view_probability <= 1:
        raise ValueError("--clean-view-probability must be in [0, 1]")
    if args.geometry == "hyperbolic":
        cosine_only = ("orthogonal_weight", "prototype_weight", "hard_negative_weight",
                       "same_gender_contrastive_weight")
        if any(getattr(args, name) > 0 for name in cosine_only):
            raise ValueError("Hyperbolic geometry supports only the contrastive, pair and identity losses")
    if args.orthogonal_negative_weight < 0:
        raise ValueError("--orthogonal-negative-weight must be non-negative")
    if not 0 <= args.input_dropout < 1 or args.input_noise_std < 0:
        raise ValueError("--input-dropout must be in [0, 1) and noise non-negative")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def read_gender_metadata(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 2 or fields[1] not in {"f", "m"}:
                raise ValueError(f"Malformed gender metadata line {line_number}")
            identity, gender = fields
            if identity in mapping:
                raise ValueError(f"Duplicate gender metadata for {identity}")
            mapping[identity] = 0 if gender == "f" else 1
    return mapping


def training_genders(
    fold_data: FoldData, gender_metadata: dict[str, int]
) -> np.ndarray:
    try:
        gender_by_local_label = np.asarray(
            [gender_metadata[identity] for identity in fold_data.train_identities],
            dtype=np.int64,
        )
    except KeyError as error:
        raise ValueError(f"Missing gender metadata for {error.args[0]}") from error
    return gender_by_local_label[fold_data.train_labels]


def append_external(fold_data: FoldData, genders: np.ndarray, args) -> tuple[np.ndarray, np.ndarray]:
    """Append external identities to the training side of the fold; validation is untouched."""
    faces = np.loadtxt(args.extra_face_csv, delimiter=",", dtype=np.float32)
    voices = np.loadtxt(args.extra_voice_csv, delimiter=",", dtype=np.float32)
    rows = [line.split() for line in args.extra_manifest.open() if line.strip()]
    names = [r[4] for r in rows]
    if not (len(faces) == len(voices) == len(names)) or not np.array_equal(faces[:, -1], voices[:, -1]):
        raise ValueError("External faces, voices and manifest must be row-aligned")
    # Unheard-language protocol: drop languages and/or hold out one fifth of a source's identities.
    held_out = holdout_external_identities(names, args.holdout_external_prefix, args.fold)
    # GAMP-VF hidden-association protocol: identities whose labels are hidden (or reserved for
    # calibration) are removed from labelled training entirely.
    hidden = set()
    if getattr(args, "exclude_external_identities", None):
        hidden = {l.strip() for l in args.exclude_external_identities.read_text().splitlines() if l.strip()}
        if not hidden <= set(names):
            raise ValueError("--exclude-external-identities lists unknown identities")
    excluded = set(args.exclude_train_languages)
    all_faces, all_voices, all_rows = faces, voices, rows
    keep = np.array([r[4] not in held_out and r[4] not in hidden and r[2].split("/")[1] not in excluded
                     for r in rows])
    faces, voices = faces[keep], voices[keep]
    rows = [r for r, k in zip(rows, keep) if k]
    names = [r[4] for r in rows]
    extra_info = {"keys": [r[0] for r in rows], "languages": [manifest_language(r[2]) for r in rows],
                  "held_out": sorted(held_out), "excluded_identities": sorted(hidden)}
    official = set(fold_data.train_identities) | set(fold_data.validation_identities)
    if official & set(names):
        raise ValueError("External identities overlap official identity names")
    gender_of = read_gender_metadata(args.extra_gender_metadata)
    identities = sorted(set(names))
    offset = len(fold_data.train_identities)
    local = {name: offset + i for i, name in enumerate(identities)}
    primary = np.arange(offset)
    new_faces, new_voices = [faces[:, :-1]], [voices[:, :-1]]
    new_labels = [np.asarray([local[n] for n in names], dtype=fold_data.train_labels.dtype)]
    new_genders = [np.asarray([gender_of[n] for n in names], dtype=np.int64)]
    extra_info["weights"] = np.ones(len(fold_data.train_identities) + len(identities))
    extra_info["conflict_groups"] = np.arange(len(fold_data.train_identities) + len(identities))
    if getattr(args, "pseudo_pairs_csv", None):
        # Pseudo rows pair a face row and a voice row of the *hidden* pool (indices into the full
        # external CSV). Their identity is a pseudo label; true names are never read here.
        with args.pseudo_pairs_csv.open() as handle:
            table = list(csv.DictReader(handle))
        face_rows = np.asarray([int(t["face_row"]) for t in table], dtype=np.int64)  # may be empty
        voice_rows = np.asarray([int(t["voice_row"]) for t in table], dtype=np.int64)
        pool = np.flatnonzero([r[4] in hidden for r in all_rows])
        if not (np.isin(face_rows, pool).all() and np.isin(voice_rows, pool).all()):
            raise ValueError("Pseudo pairs must reference rows of the excluded (hidden) pool only")
        pseudo_names = sorted({t["pseudo_id"] for t in table})
        base = offset + len(identities)
        pseudo_local = {n: base + i for i, n in enumerate(pseudo_names)}
        info = {}
        for t in table:
            info.setdefault(t["pseudo_id"], (float(t["weight"]), int(t["gender"]), t["conflict_group"]))
        groups = {g: i for i, g in enumerate(sorted({v[2] for v in info.values()}))}
        new_faces.append(all_faces[face_rows, :-1])
        new_voices.append(all_voices[voice_rows, :-1])
        new_labels.append(np.asarray([pseudo_local[t["pseudo_id"]] for t in table], dtype=fold_data.train_labels.dtype))
        new_genders.append(np.asarray([info[t["pseudo_id"]][1] for t in table], dtype=np.int64))
        extra_info["keys"] += [f"pseudo:{t['pseudo_id']}" for t in table]
        extra_info["languages"] += [manifest_language(all_rows[v][2]) for v in voice_rows]
        extra_info["weights"] = np.r_[extra_info["weights"], np.asarray([info[n][0] for n in pseudo_names], float)]
        # Conflict groups of pseudo identities live after all real identities.
        extra_info["conflict_groups"] = np.r_[extra_info["conflict_groups"],
                                              np.asarray([base + groups[info[n][2]] for n in pseudo_names], np.int64)]
        identities = identities + [f"pseudo:{n}" for n in pseudo_names]
        extra_info["pseudo_identities"] = len(pseudo_names)
        extra_info["pseudo_rows"] = len(table)
    fold_data.train_faces = np.ascontiguousarray(np.concatenate([fold_data.train_faces, *new_faces]))
    fold_data.train_voices = np.ascontiguousarray(np.concatenate([fold_data.train_voices, *new_voices]))
    fold_data.train_labels = np.concatenate([fold_data.train_labels, *new_labels])
    fold_data.train_identities = list(fold_data.train_identities) + identities
    genders = np.concatenate([genders, *new_genders])
    return genders, primary, extra_info


def build_aux_targets(fold_data: FoldData, fold_dir: Path, args, extra_info) -> np.ndarray:
    """Per training row: (age_z, build_z, language). Traits are z-scored with this fold's training rows."""
    official_rows = [line.split() for line in args.manifest.open() if line.strip()]
    train_indices = np.load(fold_dir / "train_indices.npy").astype(np.int64)
    n_official = len(train_indices)
    keys = [official_rows[i][0] for i in train_indices]
    languages = [0] * n_official  # the official training set is English
    traits = read_traits(args.trait_csv) if args.trait_csv else {}
    if extra_info is not None:
        keys += extra_info["keys"]
        languages += extra_info["languages"]
        if args.extra_trait_csv:
            traits.update(read_traits(args.extra_trait_csv))
    if len(keys) != len(fold_data.train_labels):
        raise ValueError("Auxiliary keys do not align with the training rows")
    values = np.array([traits.get(k, (np.nan, np.nan)) for k in keys], dtype=np.float64)
    if args.trait_weight > 0 and ("age" in args.trait_targets or "build" in args.trait_targets):
        labelled = np.array([not k.startswith("pseudo:") for k in keys])  # pseudo rows: traits unknown by design
        missing = np.isnan(values[labelled]).any(1).mean()
        if missing > 0.05:
            raise ValueError(f"{missing:.0%} of training rows have no trait labels")
    mean, std = np.nanmean(values, 0), np.nanstd(values, 0)
    values = (values - mean) / np.where(std > 0, std, 1)
    return np.column_stack([values, np.asarray(languages, dtype=np.float64)]).astype(np.float32)


def subsample_identities(fold_data: FoldData, genders: np.ndarray, count: int, seed: int):
    """Keep ``count`` training identities, half per gender, drawn with ``seed``."""
    rng = np.random.default_rng(seed)
    identity_gender = np.zeros(len(fold_data.train_identities), dtype=np.int64)
    identity_gender[fold_data.train_labels] = genders
    kept = []
    for value, share in ((0, count // 2), (1, count - count // 2)):
        members = np.flatnonzero(identity_gender == value)
        if share > len(members):
            raise ValueError("Not enough identities of one gender for --max-train-identities")
        kept.extend(rng.choice(members, share, replace=False).tolist())
    kept = np.sort(np.asarray(kept))
    rows = np.isin(fold_data.train_labels, kept)
    remap = {old: new for new, old in enumerate(kept)}
    fold_data.train_faces = fold_data.train_faces[rows]
    fold_data.train_voices = fold_data.train_voices[rows]
    fold_data.train_labels = np.asarray([remap[int(x)] for x in fold_data.train_labels[rows]], dtype=fold_data.train_labels.dtype)
    fold_data.train_identities = [fold_data.train_identities[i] for i in kept]
    return genders[rows]


def shuffle_voice_block(fold_data: FoldData, genders: np.ndarray, start: int, seed: int) -> np.ndarray:
    """Control E: break the speaker <-> extra-block association on TRAINING rows only.

    Speakers are permuted within (gender, source) groups (source = name prefix before "_",
    official people form their own group); every row of speaker s receives the block of a
    random row of speaker pi(s). Validation features are untouched. Returns pi.
    """
    rng = np.random.default_rng(seed)
    labels = fold_data.train_labels
    U = len(fold_data.train_identities)
    gender_of = np.zeros(U, dtype=np.int64)
    gender_of[labels] = genders
    source = np.array([n.split("_")[0] if "_" in n else "official" for n in fold_data.train_identities])
    perm = np.arange(U)
    for key in {(int(g), s) for g, s in zip(gender_of, source)}:
        members = np.flatnonzero((gender_of == key[0]) & (source == key[1]))
        if len(members) > 1:
            shifted = rng.permutation(members)
            while len(members) > 1 and (shifted == members).any():  # derangement: nobody keeps its own
                shifted = rng.permutation(members)
            perm[members] = shifted
    rows_of = [np.flatnonzero(labels == u) for u in range(U)]
    original = fold_data.train_voices[:, start:].copy()
    for u in range(U):
        donor = rows_of[perm[u]]
        fold_data.train_voices[rows_of[u], start:] = original[rng.choice(donor, len(rows_of[u]))]
    return perm


def uses_gamp_sampler(args: argparse.Namespace) -> bool:
    return args.batch_strategy != "default" or args.language_balanced_k or args.pseudo_pairs_csv is not None


@torch.no_grad()
def identity_similarity_from_model(model, fold_data: FoldData, device: torch.device, chunk: int = 8192) -> np.ndarray:
    """Identity x identity cross-modal confusability from the current model (training rows only)."""
    was_training = model.training
    model.eval()
    U = len(fold_data.train_identities)
    labels = torch.from_numpy(fold_data.train_labels).to(device)
    face_sum = voice_sum = None
    for start in range(0, len(labels), chunk):
        f = torch.from_numpy(fold_data.train_faces[start:start + chunk]).to(device)
        v = torch.from_numpy(fold_data.train_voices[start:start + chunk]).to(device)
        fe, ve = model(f, v)
        if face_sum is None:
            face_sum = torch.zeros(U, fe.shape[1], device=device)
            voice_sum = torch.zeros(U, ve.shape[1], device=device)
        face_sum.index_add_(0, labels[start:start + chunk], fe)
        voice_sum.index_add_(0, labels[start:start + chunk], ve)
    model.train(was_training)
    return identity_cross_modal_similarity(face_sum.cpu().numpy(), voice_sum.cpu().numpy())


def create_loader(
    fold_data: FoldData,
    genders: np.ndarray,
    args: argparse.Namespace,
    pin_memory: bool,
    voice_views: np.ndarray | None = None,
    aux: np.ndarray | None = None,
    extra_info: dict | None = None,
    row_languages: np.ndarray | None = None,
) -> tuple[DataLoader, IdentityBalancedBatchSampler]:
    dataset = CrossModalFeatureDataset(
        fold_data, genders, voice_views, args.clean_view_probability, aux
    )
    if uses_gamp_sampler(args):
        U = len(fold_data.train_identities)
        identity_genders = np.full(U, -1, dtype=np.int64)
        identity_genders[fold_data.train_labels] = genders
        ages = None
        if args.batch_strategy == "same_gender_age":
            if aux is None:
                raise ValueError("same_gender_age needs --trait-csv (age targets)")
            ages = np.full(U, np.nan)
            for u in range(U):
                a = aux[fold_data.train_labels == u, 0]
                a = a[np.isfinite(a)]
                ages[u] = np.median(a) if len(a) else np.nan
        weights = conflicts = None
        if extra_info is not None:
            weights, conflicts = extra_info["weights"], extra_info["conflict_groups"]
        if args.batches_per_epoch is None:
            raise ValueError("GAMP-VF sampling needs --batches-per-epoch")
        sampler = GenderAwareHardNegativeSampler(
            fold_data.train_labels, args.identities_per_batch, args.samples_per_identity, args.seed,
            args.batches_per_epoch, identity_genders,
            strategy="random" if args.batch_strategy == "default" else args.batch_strategy,
            hard_ratio=args.hard_ratio, top_k=args.hard_top_k,
            primary_identities=getattr(args, "primary_identities", None), primary_share=args.official_share,
            row_languages=row_languages, language_balanced_k=args.language_balanced_k,
            identity_weights=weights, identity_ages=ages, conflict_groups=conflicts)
        loader = DataLoader(dataset, batch_sampler=sampler, num_workers=args.num_workers, pin_memory=pin_memory)
        return loader, sampler
    sampler = IdentityBalancedBatchSampler(
        fold_data.train_labels,
        args.identities_per_batch,
        args.samples_per_identity,
        args.seed,
        batches_per_epoch=args.batches_per_epoch,
        genders=genders,
        gender_balanced=args.gender_balanced,
        primary_identities=getattr(args, "primary_identities", None),
        primary_share=args.official_share,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return loader, sampler


def train_one_epoch(
    model: CrossModalIdentityNetwork,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    aux_modules: nn.ModuleDict | None = None,
) -> dict[str, float]:
    model.train()
    if aux_modules is not None:
        aux_modules.train()
    names = (
        "total", "identity", "contrastive", "pair", "orthogonal",
        "orthogonal_positive", "orthogonal_negative", "prototype",
        "hard_negative", "same_gender_contrastive", "positive_similarity",
        "negative_similarity", "trait", "language_adversary", "language_accuracy", "cross_lingual",
        "prototype_contrastive",
    )
    totals = {name: 0.0 for name in names}
    examples = 0
    for batch in loader:
        faces, voices, labels, genders = batch[:4]
        aux = batch[4].to(device, non_blocking=True) if len(batch) > 4 else None
        faces = faces.to(device, non_blocking=True)
        voices = voices.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)
        face_embeddings, voice_embeddings = model(faces, voices)
        trait_loss = language_loss = language_accuracy = cross_lingual_loss = face_embeddings.sum() * 0.0
        if aux is not None:
            age, build, language = aux[:, 0], aux[:, 1], aux[:, 2].long()
            if args.trait_weight > 0:
                targets = tuple(args.trait_targets.split(","))
                heads = aux_modules["traits"]
                trait_loss = 0.5 * (heads.loss(face_embeddings, age, build, genders, targets)
                                    + heads.loss(voice_embeddings, age, build, genders, targets))
            if args.language_adversary_weight > 0:
                language_loss, language_accuracy = aux_modules["language"].loss(
                    voice_embeddings, language, args.language_adversary_scale,
                    getattr(aux_modules["language"], "class_weights", None))
            if args.cross_lingual_weight > 0:
                cross_lingual_loss = cross_lingual_alignment(voice_embeddings, labels, language)

        face_logits = model.identity_logits(face_embeddings, labels)
        voice_logits = model.identity_logits(voice_embeddings, labels)
        identity_loss = 0.5 * (
            nn.functional.cross_entropy(face_logits, labels)
            + nn.functional.cross_entropy(voice_logits, labels)
        )
        contrastive_loss, similarities = symmetric_cross_modal_contrastive_loss(
            face_embeddings,
            voice_embeddings,
            labels,
            args.contrastive_temperature,
            args.contrastive_mode,
            similarities=model.similarity_matrix(face_embeddings, voice_embeddings),
        )
        if model.geometry == "hyperbolic":
            # Mean geodesic distance between each face and its own voice.
            pair_loss = -model.pair_similarity(face_embeddings, voice_embeddings).mean()
        else:
            pair_loss = paired_cosine_alignment_loss(face_embeddings, voice_embeddings)
        orthogonal_loss, orthogonal_positive, orthogonal_negative = (
            fused_orthogonal_projection_loss(
                face_embeddings,
                voice_embeddings,
                labels,
                args.orthogonal_negative_weight,
            )
        )

        model.voice_prototypes.update(voice_embeddings.detach(), labels)
        prototype_loss = similarities.sum() * 0.0
        if args.prototype_weight > 0 and epoch >= args.prototype_start_epoch:
            prototype_logits, prototype_targets = (
                model.voice_prototypes.logits_and_targets(
                    face_embeddings, labels, args.prototype_temperature
                )
            )
            prototype_loss = nn.functional.cross_entropy(
                prototype_logits, prototype_targets
            )

        hard_negative_loss = similarities.sum() * 0.0
        if args.hard_negative_weight > 0:
            hard_negative_loss = symmetric_same_gender_hard_negative_loss(
                face_embeddings,
                voice_embeddings,
                labels,
                genders,
                args.hard_negative_margin,
                args.hard_negative_top_k,
            )

        prototype_contrastive = similarities.sum() * 0.0
        if args.prototype_contrastive_weight > 0:
            batch_languages = aux[:, 2].long() if (aux is not None and args.prototype_language_balanced) else None
            prototype_contrastive = prototype_contrastive_loss(
                face_embeddings, voice_embeddings, labels, args.contrastive_temperature,
                batch_languages, args.prototype_language_balanced)

        same_gender_contrastive_loss = similarities.sum() * 0.0
        if args.same_gender_contrastive_weight > 0:
            same_gender_contrastive_loss = symmetric_same_gender_supcon_loss(
                face_embeddings,
                voice_embeddings,
                labels,
                genders,
                args.contrastive_temperature,
            )

        total_loss = (
            args.identity_weight * identity_loss
            + args.contrastive_weight * contrastive_loss
            + args.pair_weight * pair_loss
            + args.orthogonal_weight * orthogonal_loss
            + args.prototype_weight * prototype_loss
            + args.hard_negative_weight * hard_negative_loss
            + args.same_gender_contrastive_weight * same_gender_contrastive_loss
            + args.trait_weight * trait_loss
            + args.language_adversary_weight * language_loss
            + args.cross_lingual_weight * cross_lingual_loss
            + args.prototype_contrastive_weight * prototype_contrastive
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.gradient_clip > 0:
            parameters = list(model.parameters()) + ([] if aux_modules is None else list(aux_modules.parameters()))
            nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
        optimizer.step()

        same_identity = labels[:, None].eq(labels[None, :])
        count = len(labels)
        examples += count
        values = {
            "total": total_loss,
            "identity": identity_loss,
            "contrastive": contrastive_loss,
            "pair": pair_loss,
            "orthogonal": orthogonal_loss,
            "orthogonal_positive": orthogonal_positive,
            "orthogonal_negative": orthogonal_negative,
            "prototype": prototype_loss,
            "hard_negative": hard_negative_loss,
            "same_gender_contrastive": same_gender_contrastive_loss,
            "positive_similarity": similarities[same_identity].mean(),
            "negative_similarity": similarities[~same_identity].mean(),
            "trait": trait_loss,
            "language_adversary": language_loss,
            "language_accuracy": language_accuracy,
            "cross_lingual": cross_lingual_loss,
            "prototype_contrastive": prototype_contrastive,
        }
        for name, value in values.items():
            totals[name] += float(value.detach()) * count
    if examples == 0:
        raise RuntimeError("The training loader produced no batches")
    return {name: total / examples for name, total in totals.items()}


def save_checkpoint(
    path: Path,
    model: CrossModalIdentityNetwork,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    result: ValidationResult,
    args: argparse.Namespace,
    aux_modules: nn.ModuleDict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "cross_modal_identity_network",
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "aux_state_dict": None if aux_modules is None else aux_modules.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation": vars(result),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def append_log(
    path: Path,
    epoch: int,
    learning_rate: float,
    metrics: dict[str, float],
    result: ValidationResult,
    best_eer: float,
) -> None:
    row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        **metrics,
        **vars(result),
        "best_mean_eer": best_eer,
    }
    new_file = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    fold_dir = args.validation_root / f"fold_{args.fold}"
    fold_data = prepare_fold_data(
        args.face_csv, args.voice_csv, args.manifest, fold_dir
    )
    genders = training_genders(
        fold_data, read_gender_metadata(args.gender_metadata)
    )
    if args.voice_layers_npy is not None:
        layers = np.load(args.voice_layers_npy).astype(np.float32)
        if layers.ndim != 3 or len(layers) != len(fold_data.train_labels) + len(fold_data.validation_original_indices):
            raise ValueError("Voice layer array must be (all manifest rows, layers, 2*dim)")
        if args.voice_layer_features == "mean":
            layers = layers[:, :, : layers.shape[2] // 2]
        train_indices = np.load(fold_dir / "train_indices.npy").astype(np.int64)
        fold_data.train_voices = np.ascontiguousarray(layers[train_indices])
        fold_data.validation_voices = np.ascontiguousarray(layers[fold_data.validation_original_indices])
    if args.max_train_identities is not None:
        if args.voice_layers_npy is not None or args.voice_views_npy is not None:
            raise ValueError("--max-train-identities supports plain voice features only")
        genders = subsample_identities(fold_data, genders, args.max_train_identities, 1000 * args.seed + args.fold)
    if args.extra_face_csv is not None:
        if any(x is None for x in (args.extra_voice_csv, args.extra_manifest, args.extra_gender_metadata)):
            raise ValueError("--extra-face-csv needs --extra-voice-csv, --extra-manifest and --extra-gender-metadata")
        if args.voice_layers_npy is not None or args.voice_views_npy is not None:
            raise ValueError("External data supports plain voice features only")
        genders, primary, extra_info = append_external(fold_data, genders, args)
        args.primary_identities = primary.tolist()
        args.held_out_external = extra_info["held_out"]
    elif args.official_share is not None:
        raise ValueError("--official-share needs external data")
    else:
        extra_info = None
    use_aux = (args.trait_weight > 0 or args.language_adversary_weight > 0 or args.cross_lingual_weight > 0
               or args.prototype_language_balanced or args.batch_strategy == "same_gender_age")
    if (genders < 0).any() and (args.hard_negative_weight > 0 or args.same_gender_contrastive_weight > 0
                                or args.gender_balanced):
        raise ValueError("Unknown genders (pseudo identities) are not supported by the older same-gender losses")
    n_official = len(fold_data.train_labels) - (0 if extra_info is None else len(extra_info["languages"]))
    row_languages = np.r_[np.zeros(n_official, dtype=np.int64),
                          [] if extra_info is None else np.asarray(extra_info["languages"], dtype=np.int64)]
    if args.pseudo_pairs_csv is not None:
        print(f"pseudo pairs: {extra_info['pseudo_rows']} rows, {extra_info['pseudo_identities']} pseudo identities; "
              f"excluded identities: {len(extra_info['excluded_identities'])}")
    aux = None
    language_weights = None
    if use_aux:
        if args.max_train_identities is not None or args.voice_layers_npy is not None or args.voice_views_npy is not None:
            raise ValueError("Auxiliary heads support plain voice features without subsampling only")
        aux = build_aux_targets(fold_data, fold_dir, args, extra_info)
        if args.language_adversary_weight > 0:
            counts = np.bincount(aux[:, 2][aux[:, 2] >= 0].astype(np.int64), minlength=len(LANGUAGES))
            weights = np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / (counts > 0).sum(), 0.0)
            language_weights = torch.tensor(weights, dtype=torch.float32)
            args.language_counts = counts.tolist()
    voice_views = None
    if args.voice_views_npy is not None:
        all_views = np.load(args.voice_views_npy)
        if len(all_views) != len(fold_data.train_labels) + len(fold_data.validation_original_indices):
            raise ValueError("Voice views must cover every manifest row")
        train_indices = np.load(fold_dir / "train_indices.npy").astype(np.int64)
        voice_views = np.ascontiguousarray(all_views[train_indices], dtype=np.float32)
    if args.shuffle_voice_block_start is not None:
        shuffle_voice_block(fold_data, genders, args.shuffle_voice_block_start, 7919 * args.seed + args.fold)
    loader, sampler = create_loader(
        fold_data, genders, args, pin_memory=device.type == "cuda", voice_views=voice_views, aux=aux,
        extra_info=extra_info, row_languages=row_languages,
    )
    model = CrossModalIdentityNetwork(
        face_dim=fold_data.train_faces.shape[1],
        voice_dim=fold_data.train_voices.shape[-1],
        embedding_dim=args.embedding_dim,
        num_classes=len(fold_data.train_identities),
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
        prototype_momentum=args.prototype_momentum,
        face_hidden_dims=args.face_hidden_dims,
        voice_hidden_dims=args.voice_hidden_dims,
        hidden_dropout=args.hidden_dropout,
        standardize_inputs=args.standardize_inputs,
        input_dropout=args.input_dropout,
        input_noise_std=args.input_noise_std,
        geometry=args.geometry,
        curvature=args.curvature,
        clip_radius=args.clip_radius,
        voice_layers=fold_data.train_voices.shape[1] if fold_data.train_voices.ndim == 3 else 0,
        voice_layer_pooling=args.voice_layer_pooling,
        voice_fusion=args.voice_fusion,
        voice_fusion_split=args.voice_fusion_split,
        voice_fusion_dim=args.voice_fusion_dim,
    )
    if args.standardize_inputs:
        # Statistics come from this fold's training rows only.
        for branch, features in (
            (model.face_branch, fold_data.train_faces),
            (model.voice_branch, fold_data.train_voices),
        ):
            branch.set_input_statistics(
                torch.from_numpy(features.mean(axis=0)),
                torch.from_numpy(features.std(axis=0)),
            )
    model = model.to(device)
    aux_modules = None
    if use_aux:
        aux_modules = nn.ModuleDict({
            "traits": TraitHeads(args.embedding_dim),
            "language": LanguageAdversary(args.embedding_dim, len(LANGUAGES)),
        }).to(device)
        if language_weights is not None:
            aux_modules["language"].register_buffer("class_weights", language_weights.to(device))
    trainable = list(model.parameters()) + ([] if aux_modules is None else list(aux_modules.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "training_log.csv"
    if log_path.exists():
        raise FileExistsError(f"Refusing to mix runs in existing log: {log_path}")

    print(
        f"fold={args.fold} device={device} train={len(fold_data.train_labels)} "
        f"validation={len(fold_data.validation_original_indices)} "
        f"identities={len(fold_data.train_identities)} "
        f"batch=P{args.identities_per_batch}xK{args.samples_per_identity}="
        f"{sampler.batch_size} batches/epoch={len(sampler)} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(
        "loss weights: "
        f"identity={args.identity_weight} contrastive={args.contrastive_weight} "
        f"pair={args.pair_weight} prototype={args.prototype_weight} "
        f"orthogonal={args.orthogonal_weight} "
        f"orthogonal_negative={args.orthogonal_negative_weight} "
        f"hard_negative={args.hard_negative_weight} "
        f"same_gender_contrastive={args.same_gender_contrastive_weight} "
        f"contrastive_mode={args.contrastive_mode}"
    )
    print(
        f"projectors: face={args.face_hidden_dims} voice={args.voice_hidden_dims} "
        f"embedding={args.embedding_dim} hidden_dropout={args.hidden_dropout} "
        f"standardize={args.standardize_inputs} input_dropout={args.input_dropout} "
        f"input_noise={args.input_noise_std}"
    )
    best_eer = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        if (args.batch_strategy in ("nearest", "same_gender_nearest") and epoch > args.hard_warmup
                and (epoch - args.hard_warmup - 1) % args.hard_refresh_every == 0):
            # Training rows only; validation identities never enter the similarity table.
            sampler.set_identity_similarity(identity_similarity_from_model(model, fold_data, device))
        sampler.set_epoch(epoch)
        metrics = train_one_epoch(model, loader, optimizer, device, args, epoch, aux_modules)
        result = evaluate_fold(
            model,
            fold_data.validation_faces,
            fold_data.validation_voices,
            fold_data.validation_original_indices,
            fold_dir,
            device,
            args.eval_batch_size,
            score_mode="cosine",
        )
        improved = result.mean_eer < best_eer - args.min_delta
        if improved:
            best_eer = result.mean_eer
            best_epoch = epoch
            stale_epochs = 0
            if not args.no_best_checkpoint:
                save_checkpoint(
                    args.save_dir / "checkpoint_best.pth.tar",
                    model,
                    optimizer,
                    epoch,
                    result,
                    args,
                    aux_modules,
                )
        else:
            stale_epochs += 1
        save_checkpoint(
            args.save_dir / "checkpoint_last.pth.tar",
            model,
            optimizer,
            epoch,
            result,
            args,
            aux_modules,
        )
        append_log(
            log_path,
            epoch,
            optimizer.param_groups[0]["lr"],
            metrics,
            result,
            best_eer,
        )
        print(
            f"epoch={epoch:03d} loss={metrics['total']:.4f} "
            f"opl={metrics['orthogonal']:.4f} "
            f"pos={metrics['positive_similarity']:.3f} "
            f"neg={metrics['negative_similarity']:.3f} "
            f"standard_EER={result.standard_eer:.4f} "
            f"gender_EER={result.gender_eer:.4f} mean_EER={result.mean_eer:.4f} "
            f"best={best_eer:.4f}@{best_epoch} patience={stale_epochs}/{args.patience}"
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after epoch {epoch}")
            break
    print(
        f"Finished fold {args.fold}; best mean EER={best_eer:.4f} at epoch "
        f"{best_epoch}. Checkpoint: {args.save_dir / 'checkpoint_best.pth.tar'}"
    )


if __name__ == "__main__":
    main()
