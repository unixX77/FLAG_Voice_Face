"""Train GADH-V1 without modifying the original cross-modal V1 baseline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .gender_aware_model import GenderAwareDualHeadNetwork
from .local_validation.cross_modal_losses import (
    paired_cosine_alignment_loss,
    symmetric_cross_modal_contrastive_loss,
    symmetric_multi_positive_same_gender_loss,
)
from .local_validation.fold_dataset import prepare_fold_data
from .local_validation.identity_sampler import IdentityBalancedBatchSampler
from .local_validation.model_evaluation import ValidationResult, evaluate_fold
from .train_cross_modal import (
    CrossModalFeatureDataset,
    read_gender_metadata,
    resolve_device,
    set_seed,
    training_genders,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, choices=range(5), default=0)
    parser.add_argument("--validation-root", type=Path, default=Path("validation"))
    parser.add_argument("--face-csv", type=Path, required=True)
    parser.add_argument("--voice-csv", type=Path, required=True)
    parser.add_argument(
        "--manifest", type=Path, default=Path("train_set/train_set/train_English.txt")
    )
    parser.add_argument(
        "--gender-metadata", type=Path, default=Path("meta_file_train_set.csv")
    )
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument(
        "--architecture-mode",
        choices=("dual_head", "single_256"),
        default="dual_head",
        help="Full GADH-V1 or same-gender loss on the existing 256-D space",
    )
    parser.add_argument("--global-dim", type=int, default=256)
    parser.add_argument("--fine-dim", type=int, default=128)
    parser.add_argument("--score-global-weight", type=float, default=0.5)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--identities-per-batch", type=int, default=32)
    parser.add_argument("--samples-per-identity", type=int, default=4)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument(
        "--no-gender-balanced",
        action="store_true",
        help="Disable the default equal female/male identity count per batch",
    )
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--arc-scale", type=float, default=30.0)
    parser.add_argument("--arc-margin", type=float, default=0.2)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--same-gender-weight", type=float, default=0.5)
    parser.add_argument("--same-gender-margin", type=float, default=0.1)
    parser.add_argument("--same-gender-temperature", type=float, default=0.07)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.identities_per_batch < 2 or args.samples_per_identity < 2:
        raise ValueError("P and K must both be at least 2")
    if not args.no_gender_balanced and args.identities_per_batch % 2:
        raise ValueError("Gender-balanced sampling requires an even P")
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Epoch and patience values must be positive")
    if args.global_dim < 1 or args.fine_dim < 1:
        raise ValueError("Embedding dimensions must be positive")
    if not 0.0 <= args.score_global_weight <= 1.0:
        raise ValueError("--score-global-weight must lie in [0, 1]")
    weights = (
        args.identity_weight,
        args.contrastive_weight,
        args.pair_weight,
        args.same_gender_weight,
    )
    if any(weight < 0 for weight in weights) or sum(weights) == 0:
        raise ValueError("Loss weights must be non-negative and not all zero")
    if args.same_gender_margin < 0:
        raise ValueError("Same-gender margin must be non-negative")
    if args.contrastive_temperature <= 0 or args.same_gender_temperature <= 0:
        raise ValueError("Temperatures must be positive")


def create_loader(fold_data, genders, args, pin_memory: bool):
    dataset = CrossModalFeatureDataset(fold_data, genders)
    sampler = IdentityBalancedBatchSampler(
        fold_data.train_labels,
        args.identities_per_batch,
        args.samples_per_identity,
        args.seed,
        batches_per_epoch=args.batches_per_epoch,
        genders=genders,
        gender_balanced=not args.no_gender_balanced,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return loader, sampler


def train_one_epoch(model, loader, optimizer, device, args) -> dict[str, float]:
    model.train()
    metric_names = (
        "total",
        "identity",
        "contrastive_global",
        "pair_global",
        "same_gender",
        "same_gender_positive",
        "same_gender_negative",
    )
    totals = {name: 0.0 for name in metric_names}
    examples = 0
    for faces, voices, labels, genders in loader:
        faces = faces.to(device, non_blocking=True)
        voices = voices.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)
        face, voice = model(faces, voices)

        identity_loss = 0.5 * (
            F.cross_entropy(model.identity_logits(face.global_embedding, labels), labels)
            + F.cross_entropy(model.identity_logits(voice.global_embedding, labels), labels)
        )
        contrastive_loss, _ = symmetric_cross_modal_contrastive_loss(
            face.global_embedding,
            voice.global_embedding,
            labels,
            args.contrastive_temperature,
            "multi_positive",
        )
        pair_loss = paired_cosine_alignment_loss(
            face.global_embedding, voice.global_embedding
        )
        same_gender_loss, positive_score, negative_score = (
            symmetric_multi_positive_same_gender_loss(
                face.fine_embedding,
                voice.fine_embedding,
                labels,
                genders,
                args.same_gender_margin,
                args.same_gender_temperature,
            )
        )
        total_loss = (
            args.identity_weight * identity_loss
            + args.contrastive_weight * contrastive_loss
            + args.pair_weight * pair_loss
            + args.same_gender_weight * same_gender_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        count = len(labels)
        examples += count
        values = {
            "total": total_loss,
            "identity": identity_loss,
            "contrastive_global": contrastive_loss,
            "pair_global": pair_loss,
            "same_gender": same_gender_loss,
            "same_gender_positive": positive_score,
            "same_gender_negative": negative_score,
        }
        for name, value in values.items():
            totals[name] += float(value.detach()) * count
    if examples == 0:
        raise RuntimeError("The training loader produced no batches")
    return {name: value / examples for name, value in totals.items()}


def save_checkpoint(path, model, optimizer, epoch, result, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": GenderAwareDualHeadNetwork.architecture_name,
            "architecture_mode": args.architecture_mode,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation": vars(result),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def append_log(path, epoch, learning_rate, metrics, result, best_eer) -> None:
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
    loader, sampler = create_loader(
        fold_data, genders, args, pin_memory=device.type == "cuda"
    )
    model = GenderAwareDualHeadNetwork(
        face_dim=fold_data.train_faces.shape[1],
        voice_dim=fold_data.train_voices.shape[1],
        num_classes=len(fold_data.train_identities),
        global_dim=args.global_dim,
        fine_dim=args.fine_dim,
        score_global_weight=args.score_global_weight,
        architecture_mode=args.architecture_mode,
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "training_log.csv"
    if log_path.exists():
        raise FileExistsError(f"Refusing to mix runs in existing log: {log_path}")

    print(
        f"architecture={model.architecture_name} mode={args.architecture_mode} "
        f"fold={args.fold} device={device} train={len(fold_data.train_labels)} "
        f"validation={len(fold_data.validation_original_indices)} "
        f"identities={len(fold_data.train_identities)} "
        f"batch=P{args.identities_per_batch}xK{args.samples_per_identity}="
        f"{sampler.batch_size} gender_balanced={not args.no_gender_balanced} "
        f"batches/epoch={len(sampler)} "
        f"parameters={sum(parameter.numel() for parameter in model.parameters())}"
    )
    print(
        "loss weights: "
        f"identity={args.identity_weight} contrastive={args.contrastive_weight} "
        f"pair={args.pair_weight} same_gender={args.same_gender_weight}; "
        f"score_global_weight={args.score_global_weight}"
    )
    best_eer = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        sampler.set_epoch(epoch)
        metrics = train_one_epoch(model, loader, optimizer, device, args)
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
            save_checkpoint(
                args.save_dir / "checkpoint_best.pth.tar",
                model,
                optimizer,
                epoch,
                result,
                args,
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
            f"sg={metrics['same_gender']:.4f} "
            f"sg_pos={metrics['same_gender_positive']:.3f} "
            f"sg_neg={metrics['same_gender_negative']:.3f} "
            f"standard_EER={result.standard_eer:.4f} "
            f"gender_EER={result.gender_eer:.4f} mean_EER={result.mean_eer:.4f} "
            f"best={best_eer:.4f}@{best_epoch} patience={stale_epochs}/{args.patience}"
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after epoch {epoch}")
            break
    print(
        f"Finished {model.architecture_name} fold {args.fold}; best mean EER="
        f"{best_eer:.4f} at epoch {best_epoch}. Checkpoint: "
        f"{args.save_dir / 'checkpoint_best.pth.tar'}"
    )


if __name__ == "__main__":
    main()
