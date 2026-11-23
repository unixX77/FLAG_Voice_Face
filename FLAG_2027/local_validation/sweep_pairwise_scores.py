"""Compare cosine, pairwise, and residual-combined scores across all folds.

Each fold uses its own saved checkpoint and its identity-disjoint validation
protocol. Score normalization uses only the unlabeled score distribution of
each protocol cell. The alpha grid should be fixed before looking at results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

try:
    from ..pairwise_model import PairwiseVerificationNetwork
    from .fold_dataset import prepare_fold_data
    from .metrics import compute_eer
    from .model_evaluation import extract_validation_embeddings, load_protocol
except ImportError:
    from pairwise_model import PairwiseVerificationNetwork
    from local_validation.fold_dataset import prepare_fold_data
    from local_validation.metrics import compute_eer
    from local_validation.model_evaluation import (
        extract_validation_embeddings,
        load_protocol,
    )


DEFAULT_ALPHAS = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 1.0)


def parse_alpha_grid(value: str) -> tuple[float, ...]:
    try:
        alphas = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from error
    if not alphas or any(alpha < 0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must be non-negative")
    if len(set(alphas)) != len(alphas):
        raise argparse.ArgumentTypeError("alphas must be unique")
    if 0.0 not in alphas:
        raise argparse.ArgumentTypeError("include alpha=0.0 as the cosine reference")
    return alphas


def zscore_unlabeled(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scale = float(scores.std())
    if not np.isfinite(scores).all() or scale <= 1e-12:
        raise ValueError("Cannot normalize non-finite or constant score values")
    return (scores - float(scores.mean())) / scale


def protocol_scores(model, protocol, face_embeddings, voice_embeddings, row_to_position,
                    device: torch.device, batch_size: int):
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
        raise ValueError(f"Protocol references non-validation index {error.args[0]}") from error

    cosine = np.sum(face_embeddings[face_positions] * voice_embeddings[voice_positions], axis=1)
    verifier_parts: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(face_positions), batch_size):
            stop = min(start + batch_size, len(face_positions))
            faces = torch.from_numpy(face_embeddings[face_positions[start:stop]]).to(device)
            voices = torch.from_numpy(voice_embeddings[voice_positions[start:stop]]).to(device)
            verifier_parts.append(model.score_embeddings(faces, voices).cpu().numpy())
    verifier = np.concatenate(verifier_parts).astype(np.float64, copy=False)
    if not np.isfinite(cosine).all() or not np.isfinite(verifier).all():
        raise ValueError("A scoring method produced NaN or infinity")
    return cosine.astype(np.float64, copy=False), verifier


def create_model(checkpoint: dict, face_dim: int, voice_dim: int, num_classes: int):
    config = checkpoint["config"]
    model = PairwiseVerificationNetwork(
        face_dim=face_dim,
        voice_dim=voice_dim,
        embedding_dim=int(config["embedding_dim"]),
        num_classes=num_classes,
        arc_scale=float(config["arc_scale"]),
        arc_margin=float(config["arc_margin"]),
        verifier_hidden_dim=int(config["verifier_hidden_dim"]),
        verifier_dropout=float(config["verifier_dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


def evaluate_fold(fold: int, args, device: torch.device, alphas: tuple[float, ...]):
    checkpoint_path = (
        args.run_root / f"fold_{fold}" / args.checkpoint_name
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing fold {fold} checkpoint: {checkpoint_path}. "
            "Train all five folds before aggregating this comparison."
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    expected_architecture = PairwiseVerificationNetwork.architecture_name
    if checkpoint.get("architecture") != expected_architecture:
        raise ValueError(f"Unexpected checkpoint architecture in {checkpoint_path}")

    fold_dir = args.validation_root / f"fold_{fold}"
    data = prepare_fold_data(args.face_csv, args.voice_csv, args.manifest, fold_dir)
    model = create_model(
        checkpoint,
        data.train_faces.shape[1],
        data.train_voices.shape[1],
        len(data.train_identities),
    ).to(device)
    face_embeddings, voice_embeddings, row_to_position = extract_validation_embeddings(
        model,
        data.validation_faces,
        data.validation_voices,
        data.validation_original_indices,
        device,
        args.embed_batch_size,
    )

    output: dict[str, dict[str, tuple[float, float]]] = {
        "cosine": {},
        "verifier": {},
        **{f"combined_{alpha:g}": {} for alpha in alphas},
    }
    for track in ("standard", "gender"):
        protocol = load_protocol(
            fold_dir / f"{track}_pairs.csv",
            fold_dir / f"{track}_ground_truth.csv",
        )
        cosine, verifier = protocol_scores(
            model,
            protocol,
            face_embeddings,
            voice_embeddings,
            row_to_position,
            device,
            args.score_batch_size,
        )
        cosine_scaled = zscore_unlabeled(cosine)
        verifier_scaled = zscore_unlabeled(verifier)
        output["cosine"][track] = compute_eer(protocol.labels, cosine)
        output["verifier"][track] = compute_eer(protocol.labels, verifier)
        for alpha in alphas:
            combined = cosine_scaled + alpha * verifier_scaled
            output[f"combined_{alpha:g}"][track] = compute_eer(
                protocol.labels, combined
            )
    return int(checkpoint["epoch"]), output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default="checkpoint_best.pth.tar")
    parser.add_argument("--face-csv", type=Path, required=True)
    parser.add_argument("--voice-csv", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("train_set/train_set/train_English.txt")
    )
    parser.add_argument("--validation-root", type=Path, default=Path("validation"))
    parser.add_argument("--alphas", type=parse_alpha_grid, default=DEFAULT_ALPHAS)
    parser.add_argument("--embed-batch-size", type=int, default=512)
    parser.add_argument("--score-batch-size", type=int, default=1024)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.embed_batch_size < 1 or args.score_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)

    records: list[dict[str, float | int | str]] = []
    mode_specs = [("cosine", None), ("verifier", None)] + [
        (f"combined_{alpha:g}", alpha) for alpha in args.alphas
    ]
    print("fold  epoch  score_mode       weight  standard_EER  gender_EER  mean_EER")
    for fold in range(5):
        epoch, metrics_by_mode = evaluate_fold(fold, args, device, args.alphas)
        for mode, alpha in mode_specs:
            standard_eer, standard_auc = metrics_by_mode[mode]["standard"]
            gender_eer, gender_auc = metrics_by_mode[mode]["gender"]
            mean_eer = 0.5 * (standard_eer + gender_eer)
            records.append(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "score_mode": mode,
                    "verifier_weight": "" if alpha is None else alpha,
                    "standard_eer": standard_eer,
                    "standard_auc": standard_auc,
                    "gender_eer": gender_eer,
                    "gender_auc": gender_auc,
                    "mean_eer": mean_eer,
                }
            )
            weight_text = "" if alpha is None else f"{alpha:.2f}"
            print(
                f"{fold:>4d}  {epoch:>5d}  {mode:15s} {weight_text:>6s}  "
                f"{standard_eer:>12.4f}  {gender_eer:>10.4f}  {mean_eer:>8.4f}"
            )

    print("\nFive-fold aggregate (mean +/- population std)")
    print("score_mode       weight  standard_EER        gender_EER          mean_EER")
    for mode, alpha in mode_specs:
        selected = [row for row in records if row["score_mode"] == mode]
        standard = np.asarray([row["standard_eer"] for row in selected])
        gender = np.asarray([row["gender_eer"] for row in selected])
        mean = np.asarray([row["mean_eer"] for row in selected])
        print(
            f"{mode:15s} {'' if alpha is None else f'{alpha:.2f}':>6s}  "
            f"{standard.mean():.4f} +/- {standard.std():.4f}  "
            f"{gender.mean():.4f} +/- {gender.std():.4f}  "
            f"{mean.mean():.4f} +/- {mean.std():.4f}"
        )
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"Saved per-fold metrics to {args.output_csv}")


if __name__ == "__main__":
    main()
