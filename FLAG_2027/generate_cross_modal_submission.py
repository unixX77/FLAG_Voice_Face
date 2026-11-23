"""Generate and validate a CodaBench ZIP from a full-data cross-modal checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:  # Package execution: python -m FLAG_2027.generate_cross_modal_submission
    from .cross_modal_model import CrossModalIdentityNetwork
except ImportError:  # Direct imports used by local tests
    from cross_modal_model import CrossModalIdentityNetwork


PROTOCOLS = (
    ("no_gender", "English", "sub_score_v4_English_heard.txt"),
    ("no_gender", "Bangla", "sub_score_v4_Bangla_unheard.txt"),
    ("gender", "English", "sub_score_v4_English_heard.txt"),
    ("gender", "Bangla", "sub_score_v4_Bangla_unheard.txt"),
)


def read_ids(path: Path) -> list[str]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if any(len(row) != 3 for row in rows):
        raise ValueError(f"Malformed three-field protocol: {path}")
    ids = [row[0] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate pair ID in {path}")
    return ids


def read_features(path: Path, dimension: int) -> np.ndarray:
    values = pd.read_csv(path, header=None, dtype=np.float32).to_numpy(copy=False)
    if values.shape != (len(values), dimension):
        raise ValueError(f"Expected {dimension} columns in {path}, found {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite feature in {path}")
    return np.ascontiguousarray(values)


def score(model, faces, voices, device, batch_size) -> np.ndarray:
    parts = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(faces), batch_size):
            stop = min(start + batch_size, len(faces))
            face, voice = model(
                torch.from_numpy(faces[start:stop]).to(device),
                torch.from_numpy(voices[start:stop]).to(device),
            )
            # Official evaluation applies ROC to -submitted_score, so write a
            # distance: smaller is a stronger match. Since both outputs are
            # unit-normalized, 1-cosine is monotonic with Euclidean distance.
            # Hyperbolic models write their geodesic distance directly.
            if getattr(model, "geometry", "sphere") == "hyperbolic":
                parts.append((-model.pair_similarity(face, voice)).cpu().numpy())
            else:
                parts.append((1.0 - (face * voice).sum(dim=1)).cpu().numpy())
    return np.concatenate(parts).astype(np.float64, copy=False)


def attribute_terms(config: dict, track: str, voice_attributes: Path, face_attributes: Path,
                    expected_ids: list[str]) -> np.ndarray:
    """Fixed age/gender-consistency distance terms for one cell, from per-trial estimates.

    Uses only each trial's own voice clip and face image estimates, and the
    training-fitted correction and weights stored in ``config``.
    """
    import csv

    voice = list(csv.DictReader(voice_attributes.open()))
    face = list(csv.DictReader(face_attributes.open()))
    if len(voice) != len(expected_ids) or [row["key"] for row in face] != expected_ids:
        raise ValueError(f"Attribute files do not align with the protocol: {voice_attributes}, {face_attributes}")
    voice_age = np.array([float(r["age_years"]) for r in voice])
    voice_male = np.array([float(r["p_male"]) / max(float(r["p_male"]) + float(r["p_female"]), 1e-6) for r in voice])
    face_age = np.array([float(r["age"]) if r["age"] else np.nan for r in face])
    face_male = np.array([1.0 if r["gender_pred"] == "m" else 0.0 if r["gender_pred"] == "f" else np.nan for r in face])
    corrected = voice_age.copy()
    for name, mask in (("male", voice_male >= 0.5), ("female", voice_male < 0.5)):
        fit = config["voice_age_correction"][name]
        corrected[mask] = fit["intercept"] + fit["slope"] * voice_age[mask]
    weights = config["weights"][track]
    age_gap = np.nan_to_num(np.abs(corrected - face_age) / 10.0)
    gender_gap = np.nan_to_num(np.abs(voice_male - face_male))
    return weights["w_age_per_10_years"] * age_gap + weights["w_gender"] * gender_gap


def validate_score_file(path: Path, expected_ids: list[str]) -> None:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != len(expected_ids) or any(len(row) != 2 for row in rows):
        raise ValueError(f"Invalid score-file shape: {path}")
    if [row[0] for row in rows] != expected_ids:
        raise ValueError(f"Pair IDs/order differ from protocol: {path}")
    scores = np.asarray([float(row[1]) for row in rows])
    if not all(math.isfinite(value) for value in scores) or scores.std() == 0:
        raise ValueError(f"Invalid or constant scores: {path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True,
                        help="one checkpoint, or several for a fixed equal-weight ensemble (mean of per-trial "
                             "distances; every trial is still scored independently)")
    parser.add_argument(
        "--dev-root",
        type=Path,
        default=Path("dev_set"),
        help="Folder holding <track>/<language>_<split>.txt trial lists",
    )
    parser.add_argument(
        "--split",
        default="test",
        help='File-name suffix: "test" for the dev set, "valid" for the evaluation phase',
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--zip-path", type=Path, required=True)
    parser.add_argument(
        "--attribute-terms",
        type=Path,
        help="JSON from analysis/finalize_attribute_terms.py; adds fixed age/gender-consistency terms",
    )
    parser.add_argument("--voice-attribute-root", type=Path, help="<root>/<track>/<language>_<split>_attributes.csv")
    parser.add_argument("--face-attribute-root", type=Path, help="<root>/<track>/<language>_<split>_face_age.csv")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    models = []
    for checkpoint_path in args.checkpoint:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not str(checkpoint.get("architecture", "")).endswith("_full"):
            raise ValueError("Checkpoint is not a full-data cross-modal checkpoint")
        if "model_kwargs" in checkpoint:
            model = CrossModalIdentityNetwork(**checkpoint["model_kwargs"])
        else:  # V1 checkpoints written before model_kwargs were stored
            model = CrossModalIdentityNetwork(
                checkpoint["face_dim"],
                checkpoint["voice_dim"],
                checkpoint["embedding_dim"],
                checkpoint["num_classes"],
                arc_scale=checkpoint["config"]["arc_scale"],
                arc_margin=checkpoint["config"]["arc_margin"],
            )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        models.append(model.to(device).eval())
        dims = (checkpoint["face_dim"], checkpoint["voice_dim"])
        if len(models) == 1:
            first_dims = dims
        elif dims != first_dims:
            raise ValueError("Ensemble members must share face/voice input dimensions")

    expected_archive_names = []
    for track, language, filename in PROTOCOLS:
        protocol = args.dev_root / track / f"{language}_{args.split}.txt"
        face_path = args.feature_root / track / f"{language}_{args.split}_faces.csv"
        voice_path = args.feature_root / track / f"{language}_{args.split}_voices.csv"
        ids = read_ids(protocol)
        faces = read_features(face_path, checkpoint["face_dim"])
        voices = read_features(voice_path, checkpoint["voice_dim"])
        if not (len(ids) == len(faces) == len(voices)):
            raise ValueError(
                f"Row mismatch for {track}/{language}: "
                f"protocol={len(ids)} face={len(faces)} voice={len(voices)}"
            )
        scores = np.mean([score(m, faces, voices, device, args.batch_size) for m in models], axis=0)
        if args.attribute_terms is not None:
            terms = attribute_terms(
                json.loads(args.attribute_terms.read_text()), track,
                args.voice_attribute_root / track / f"{language}_{args.split}_attributes.csv",
                args.face_attribute_root / track / f"{language}_{args.split}_face_age.csv", ids)
            scores = scores + terms
        output = args.output_root / track / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w") as handle:
            for pair_id, value in zip(ids, scores):
                handle.write(f"{pair_id} {value:.9f}\n")
        validate_score_file(output, ids)
        archive_name = f"{track}/{filename}"
        expected_archive_names.append(archive_name)
        print(
            f"{track:10s} {language:7s}: {len(scores)} rows "
            f"min={scores.min():.6f} max={scores.max():.6f} "
            f"mean={scores.mean():.6f} std={scores.std():.6f}"
        )

    args.zip_path.parent.mkdir(parents=True, exist_ok=True)
    if args.zip_path.exists():
        raise FileExistsError(f"Refusing to overwrite {args.zip_path}")
    with zipfile.ZipFile(args.zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_name in expected_archive_names:
            archive.write(args.output_root / archive_name, archive_name)
    with zipfile.ZipFile(args.zip_path) as archive:
        actual = archive.namelist()
        if actual != expected_archive_names or archive.testzip() is not None:
            raise RuntimeError("Generated ZIP failed structure or CRC validation")
    print(f"Validated CodaBench archive: {args.zip_path}")


if __name__ == "__main__":
    main()
