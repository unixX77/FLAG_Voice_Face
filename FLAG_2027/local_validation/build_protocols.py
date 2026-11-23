"""Build deterministic identity-disjoint FLAG 2027 validation folds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Sample:
    row_index: int
    sample_id: str
    identity: str
    numeric_label: int
    gender: str
    voice_path: str
    face_path: str


def read_gender_metadata(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 2 or row[1] not in {"m", "f"}:
                raise ValueError(f"Malformed gender row in {path}: {row}")
            identity, gender = row
            if identity in result:
                raise ValueError(f"Duplicate identity {identity} in {path}")
            result[identity] = gender
    return result


def read_manifest(path: Path, genders: dict[str, str]) -> list[Sample]:
    samples: list[Sample] = []
    identity_labels: dict[str, int] = {}
    with path.open() as handle:
        for row_index, line in enumerate(handle):
            fields = line.split()
            if len(fields) != 6:
                raise ValueError(f"Manifest line {row_index + 1} has {len(fields)} fields")
            sample_id, _, voice_path, face_path, identity, label_text = fields
            if identity not in genders:
                raise ValueError(f"Missing gender for {identity}")
            label = int(label_text)
            old_label = identity_labels.setdefault(identity, label)
            if old_label != label:
                raise ValueError(f"Identity {identity} maps to multiple numeric labels")
            samples.append(
                Sample(
                    row_index=row_index,
                    sample_id=sample_id,
                    identity=identity,
                    numeric_label=label,
                    gender=genders[identity],
                    voice_path=voice_path,
                    face_path=face_path,
                )
            )
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("Manifest sample IDs are not unique")
    return samples


def validate_feature_labels(
    face_csv: Path, voice_csv: Path, samples: list[Sample]
) -> None:
    """Validate CSV dimensions and label columns without loading all features."""
    with face_csv.open(newline="") as handle:
        face_columns = len(next(csv.reader(handle)))
    with voice_csv.open(newline="") as handle:
        voice_columns = len(next(csv.reader(handle)))
    if face_columns != 4097 or voice_columns < 2:
        raise ValueError(
            f"Wrong feature dimensions: face={face_columns}, voice={voice_columns}"
        )

    face_labels = pd.read_csv(face_csv, header=None, usecols=[4096]).iloc[:, 0].to_numpy()
    voice_labels = (
        pd.read_csv(voice_csv, header=None, usecols=[voice_columns - 1])
        .iloc[:, 0]
        .to_numpy()
    )
    manifest_labels = np.asarray([sample.numeric_label for sample in samples])
    if not (len(face_labels) == len(voice_labels) == len(manifest_labels)):
        raise ValueError(
            f"Row mismatch: face={len(face_labels)}, voice={len(voice_labels)}, "
            f"manifest={len(manifest_labels)}"
        )
    if not np.array_equal(face_labels.astype(np.int64), manifest_labels):
        raise ValueError("Face CSV labels do not align with the manifest")
    if not np.array_equal(voice_labels.astype(np.int64), manifest_labels):
        raise ValueError("Voice CSV labels do not align with the manifest")


def assign_identity_folds(
    identities: list[str], genders: dict[str, str], counts: Counter[str], seed: int
) -> list[list[str]]:
    """Create five folds with 8 male/6 female IDs and balanced sample totals."""
    if len(identities) != 70:
        raise ValueError(f"Expected 70 identities, found {len(identities)}")
    if Counter(genders[identity] for identity in identities) != {"m": 40, "f": 30}:
        raise ValueError("Expected 40 male and 30 female training identities")

    rng = np.random.default_rng(seed)
    tie_break = {identity: float(rng.random()) for identity in identities}
    ordered = sorted(identities, key=lambda x: (-counts[x], tie_break[x], x))
    folds: list[list[str]] = [[] for _ in range(5)]
    totals = [0] * 5
    gender_counts = [Counter() for _ in range(5)]
    quota = {"m": 8, "f": 6}

    for identity in ordered:
        gender = genders[identity]
        eligible = [i for i in range(5) if gender_counts[i][gender] < quota[gender]]
        fold_index = min(
            eligible,
            key=lambda i: (totals[i], len(folds[i]), gender_counts[i][gender], i),
        )
        folds[fold_index].append(identity)
        totals[fold_index] += counts[identity]
        gender_counts[fold_index][gender] += 1

    for fold in folds:
        fold.sort()
    return folds


def positive_trials(
    validation_ids: list[str], rows_by_identity: dict[str, list[int]], per_identity: int, rng: np.random.Generator
) -> tuple[list[tuple[int, int, str]], int]:
    trials: list[tuple[int, int, str]] = []
    duplicate_count = 0
    for identity in validation_ids:
        rows = rows_by_identity[identity]
        combinations = [(face, voice) for face in rows for voice in rows if face != voice]
        if not combinations:
            raise ValueError(f"Identity {identity} lacks two distinct samples")
        rng.shuffle(combinations)
        for index in range(per_identity):
            if index >= len(combinations):
                duplicate_count += 1
            face_index, voice_index = combinations[index % len(combinations)]
            trials.append((face_index, voice_index, identity))
    return trials, duplicate_count


def negative_trials(
    positives: list[tuple[int, int, str]],
    validation_ids: list[str],
    rows_by_identity: dict[str, list[int]],
    genders: dict[str, str],
    same_gender: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int, str, str]]:
    donor_usage: Counter[str] = Counter()
    negatives: list[tuple[int, int, str, str]] = []
    for face_index, _, anchor_identity in positives:
        candidates = [
            identity
            for identity in validation_ids
            if identity != anchor_identity
            and (not same_gender or genders[identity] == genders[anchor_identity])
        ]
        minimum_usage = min(donor_usage[identity] for identity in candidates)
        least_used = [identity for identity in candidates if donor_usage[identity] == minimum_usage]
        donor_identity = least_used[int(rng.integers(len(least_used)))]
        donor_rows = rows_by_identity[donor_identity]
        voice_index = donor_rows[int(rng.integers(len(donor_rows)))]
        donor_usage[donor_identity] += 1
        negatives.append((face_index, voice_index, anchor_identity, donor_identity))
    return negatives


def stable_pair_id(seed: int, fold_index: int, track: str, order: int) -> str:
    payload = f"{seed}:{fold_index}:{track}:{order}".encode()
    return "local_" + hashlib.sha256(payload).hexdigest()[:12]


def write_protocol(
    fold_dir: Path,
    fold_index: int,
    track: str,
    positives: list[tuple[int, int, str]],
    negatives: list[tuple[int, int, str, str]],
    samples: list[Sample],
    seed: int,
) -> dict[str, object]:
    raw_trials = [
        (face, voice, 1, identity, identity)
        for face, voice, identity in positives
    ] + [
        (face, voice, 0, face_identity, voice_identity)
        for face, voice, face_identity, voice_identity in negatives
    ]
    rng = np.random.default_rng(seed + 10_000 * (fold_index + 1) + (1 if track == "gender" else 0))
    permutation = rng.permutation(len(raw_trials))
    trials = [raw_trials[int(index)] for index in permutation]

    pair_path = fold_dir / f"{track}_pairs.csv"
    truth_path = fold_dir / f"{track}_ground_truth.csv"
    with pair_path.open("w", newline="") as pair_handle, truth_path.open("w", newline="") as truth_handle:
        pair_writer = csv.writer(pair_handle)
        truth_writer = csv.writer(truth_handle)
        pair_writer.writerow(["pair_id", "face_index", "voice_index"])
        truth_writer.writerow(["pair_id", "label"])
        for order, (face, voice, label, _, _) in enumerate(trials):
            pair_id = stable_pair_id(seed, fold_index, track, order)
            pair_writer.writerow([pair_id, face, voice])
            truth_writer.writerow([pair_id, label])

    labels = [trial[2] for trial in trials]
    negative_gender_pairs = Counter(
        samples[face].gender + samples[voice].gender
        for face, voice, label, _, _ in trials
        if label == 0
    )
    return {
        "pairs": len(trials),
        "positives": int(sum(labels)),
        "negatives": int(len(labels) - sum(labels)),
        "negative_gender_pairs": dict(sorted(negative_gender_pairs.items())),
        "unique_face_rows": len({trial[0] for trial in trials}),
        "unique_voice_rows": len({trial[1] for trial in trials}),
    }


def validate_protocol(
    pair_path: Path,
    truth_path: Path,
    validation_ids: set[str],
    samples: list[Sample],
    same_gender: bool,
) -> None:
    with pair_path.open(newline="") as handle:
        pairs = list(csv.DictReader(handle))
    with truth_path.open(newline="") as handle:
        truth_rows = list(csv.DictReader(handle))
    truth = {row["pair_id"]: int(row["label"]) for row in truth_rows}
    pair_ids = [row["pair_id"] for row in pairs]
    if len(pair_ids) != len(set(pair_ids)) or set(pair_ids) != set(truth):
        raise ValueError(f"Pair ID integrity failure in {pair_path}")
    labels = [truth[pair_id] for pair_id in pair_ids]
    if Counter(labels)[0] != Counter(labels)[1]:
        raise ValueError(f"Protocol is not label-balanced: {pair_path}")

    for row in pairs:
        face = samples[int(row["face_index"])]
        voice = samples[int(row["voice_index"])]
        label = truth[row["pair_id"]]
        if face.identity not in validation_ids or voice.identity not in validation_ids:
            raise ValueError(f"Non-validation identity in {pair_path}")
        if label == 1 and face.identity != voice.identity:
            raise ValueError(f"Invalid positive pair in {pair_path}")
        if label == 0 and face.identity == voice.identity:
            raise ValueError(f"Invalid negative pair in {pair_path}")
        if same_gender and label == 0 and face.gender != voice.gender:
            raise ValueError(f"Cross-gender negative in {pair_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("train_set/train_set/train_English.txt"))
    parser.add_argument("--gender-metadata", type=Path, default=Path("meta_file_train_set.csv"))
    parser.add_argument("--face-csv", type=Path, default=Path("train_set/features/faces/train_English_faces.csv"))
    parser.add_argument("--voice-csv", type=Path, default=Path("train_set/features/voices/train_English_voices.csv"))
    parser.add_argument("--output-root", type=Path, default=Path("validation"))
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--trials-per-identity", type=int, default=50)
    parser.add_argument("--skip-feature-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    genders = read_gender_metadata(args.gender_metadata)
    samples = read_manifest(args.manifest, genders)
    if not args.skip_feature_validation:
        validate_feature_labels(args.face_csv, args.voice_csv, samples)

    rows_by_identity: dict[str, list[int]] = defaultdict(list)
    for sample in samples:
        rows_by_identity[sample.identity].append(sample.row_index)
    identities = sorted(rows_by_identity)
    counts = Counter({identity: len(rows) for identity, rows in rows_by_identity.items()})
    folds = assign_identity_folds(identities, genders, counts, args.seed)
    args.output_root.mkdir(parents=True, exist_ok=True)

    fold_summary = {
        "seed": args.seed,
        "trials_per_identity": args.trials_per_identity,
        "identities": len(identities),
        "samples": len(samples),
        "folds": [],
    }
    all_validation_ids: list[str] = []
    all_ids = set(identities)

    for fold_index, validation_ids in enumerate(folds):
        fold_dir = args.output_root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        validation_set = set(validation_ids)
        train_ids = sorted(all_ids - validation_set)
        train_indices = np.asarray(
            [sample.row_index for sample in samples if sample.identity in train_ids], dtype=np.int64
        )
        validation_indices = np.asarray(
            [sample.row_index for sample in samples if sample.identity in validation_set], dtype=np.int64
        )
        np.save(fold_dir / "train_indices.npy", train_indices)
        np.save(fold_dir / "validation_indices.npy", validation_indices)

        rng = np.random.default_rng(args.seed + fold_index)
        positives, positive_duplicates = positive_trials(
            validation_ids, rows_by_identity, args.trials_per_identity, rng
        )
        standard_negatives = negative_trials(
            positives, validation_ids, rows_by_identity, genders, False, rng
        )
        gender_negatives = negative_trials(
            positives, validation_ids, rows_by_identity, genders, True, rng
        )
        standard_stats = write_protocol(
            fold_dir, fold_index, "standard", positives, standard_negatives, samples, args.seed
        )
        gender_stats = write_protocol(
            fold_dir, fold_index, "gender", positives, gender_negatives, samples, args.seed
        )
        validate_protocol(
            fold_dir / "standard_pairs.csv",
            fold_dir / "standard_ground_truth.csv",
            validation_set,
            samples,
            False,
        )
        validate_protocol(
            fold_dir / "gender_pairs.csv",
            fold_dir / "gender_ground_truth.csv",
            validation_set,
            samples,
            True,
        )

        local_label_map = {identity: index for index, identity in enumerate(train_ids)}
        metadata = {
            "fold": fold_index,
            "train_identities": train_ids,
            "validation_identities": validation_ids,
            "train_identity_label_map": local_label_map,
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "validation_gender_counts": dict(Counter(genders[x] for x in validation_ids)),
            "validation_sample_count": int(sum(counts[x] for x in validation_ids)),
            "validation_samples_per_identity": {x: counts[x] for x in validation_ids},
            "positive_duplicate_trials": positive_duplicates,
            "standard": standard_stats,
            "gender": gender_stats,
        }
        (fold_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        fold_summary["folds"].append(metadata)
        all_validation_ids.extend(validation_ids)
        print(
            f"fold {fold_index}: train IDs={len(train_ids):2d}, val IDs={len(validation_ids):2d}, "
            f"val rows={len(validation_indices):4d}, gender={metadata['validation_gender_counts']}"
        )

    if Counter(all_validation_ids) != Counter({identity: 1 for identity in identities}):
        raise ValueError("Each identity must occur in validation exactly once")
    (args.output_root / "folds.json").write_text(json.dumps(fold_summary, indent=2) + "\n")
    print(f"Wrote and validated five folds under {args.output_root}")


if __name__ == "__main__":
    main()
