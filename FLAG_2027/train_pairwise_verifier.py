"""Train V1 projectors with a learned pair verifier and hard same-gender pairs."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .local_validation.cross_modal_losses import (
        paired_cosine_alignment_loss,
        symmetric_cross_modal_contrastive_loss,
    )
    from .local_validation.fold_dataset import prepare_fold_data
    from .local_validation.identity_sampler import IdentityBalancedBatchSampler
    from .local_validation.pairwise_evaluation import evaluate_pairwise_fold
    from .pairwise_model import PairwiseVerificationNetwork
    from .train_cross_modal import (
        CrossModalFeatureDataset,
        read_gender_metadata,
        resolve_device,
        training_genders,
    )
except ImportError:
    from local_validation.cross_modal_losses import (
        paired_cosine_alignment_loss,
        symmetric_cross_modal_contrastive_loss,
    )
    from local_validation.fold_dataset import prepare_fold_data
    from local_validation.identity_sampler import IdentityBalancedBatchSampler
    from local_validation.pairwise_evaluation import evaluate_pairwise_fold
    from pairwise_model import PairwiseVerificationNetwork
    from train_cross_modal import (
        CrossModalFeatureDataset,
        read_gender_metadata,
        resolve_device,
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
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--samples-per-identity", type=int, default=8)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument(
        "--no-gender-balanced",
        action="store_true",
        help="Disable the default equal number of female and male identities per batch",
    )
    parser.add_argument("--hard-negatives-per-anchor", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--verifier-hidden-dim", type=int, default=256)
    parser.add_argument("--verifier-dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--verifier-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--arc-scale", type=float, default=30.0)
    parser.add_argument("--arc-margin", type=float, default=0.2)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument("--verifier-weight", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("Epoch and patience values must be positive")
    if args.identities_per_batch < 2 or args.samples_per_identity < 2:
        raise ValueError("P and K must both be at least 2")
    if args.hard_negatives_per_anchor < 1:
        raise ValueError("Need at least one mined negative per anchor")
    if args.embedding_dim < 1 or args.verifier_hidden_dim < 2:
        raise ValueError("Embedding and verifier dimensions are too small")
    weights = (
        args.identity_weight,
        args.contrastive_weight,
        args.pair_weight,
        args.verifier_weight,
    )
    if any(weight < 0 for weight in weights) or sum(weights) == 0:
        raise ValueError("Loss weights must be non-negative and not all zero")


def mine_training_pairs(
    similarities: torch.Tensor,
    labels: torch.Tensor,
    genders: torch.Tensor,
    hard_negatives_per_anchor: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select all within-batch positives and hardest same-gender negatives.

    Selection is based on detached cosine scores. Gradients still flow through
    the selected pair embeddings and the verifier. Every anchor must have at
    least one same-gender impostor in the sampled batch.
    """
    if similarities.shape != (len(labels), len(labels)):
        raise ValueError("Expected a square batch similarity matrix")
    if len(labels) != len(genders):
        raise ValueError("Each training row must have a gender label")
    same_identity = labels[:, None].eq(labels[None, :])
    same_gender_negative = labels[:, None].ne(labels[None, :]) & genders[:, None].eq(
        genders[None, :]
    )
    positive_faces, positive_voices = same_identity.nonzero(as_tuple=True)
    negative_face_parts: list[torch.Tensor] = []
    negative_voice_parts: list[torch.Tensor] = []
    detached = similarities.detach()
    for anchor in range(len(labels)):
        candidate_indices = same_gender_negative[anchor].nonzero(as_tuple=False).flatten()
        if len(candidate_indices) == 0:
            raise ValueError(
                "Every anchor needs a different-identity, same-gender candidate; "
                "increase identities_per_batch or balance gender sampling"
            )
        count = min(hard_negatives_per_anchor, len(candidate_indices))
        hardest_positions = detached[anchor, candidate_indices].topk(count).indices
        selected = candidate_indices[hardest_positions]
        negative_face_parts.append(torch.full_like(selected, anchor))
        negative_voice_parts.append(selected)
    negative_faces = torch.cat(negative_face_parts)
    negative_voices = torch.cat(negative_voice_parts)
    face_indices = torch.cat((positive_faces, negative_faces))
    voice_indices = torch.cat((positive_voices, negative_voices))
    targets = torch.cat(
        (
            torch.ones(len(positive_faces), device=labels.device),
            torch.zeros(len(negative_faces), device=labels.device),
        )
    )
    return face_indices, voice_indices, targets


def train_one_epoch(model, loader, optimizer, device, args) -> dict[str, float]:
    model.train()
    metric_names = (
        "total",
        "identity",
        "contrastive",
        "pair",
        "verifier",
        "positive_logit",
        "negative_logit",
    )
    totals = {name: 0.0 for name in metric_names}
    examples = 0
    for faces, voices, labels, genders in loader:
        faces = faces.to(device, non_blocking=True)
        voices = voices.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        genders = genders.to(device, non_blocking=True)
        face_embeddings, voice_embeddings = model.encode(faces, voices)
        face_logits = model.identity_logits(face_embeddings, labels)
        voice_logits = model.identity_logits(voice_embeddings, labels)
        identity_loss = 0.5 * (
            F.cross_entropy(face_logits, labels)
            + F.cross_entropy(voice_logits, labels)
        )
        contrastive_loss, similarities = symmetric_cross_modal_contrastive_loss(
            face_embeddings,
            voice_embeddings,
            labels,
            args.contrastive_temperature,
            "multi_positive",
        )
        pair_loss = paired_cosine_alignment_loss(face_embeddings, voice_embeddings)
        face_indices, voice_indices, targets = mine_training_pairs(
            similarities,
            labels,
            genders,
            args.hard_negatives_per_anchor,
        )
        pair_logits = model.score_embeddings(
            face_embeddings[face_indices], voice_embeddings[voice_indices]
        )
        verifier_loss = F.binary_cross_entropy_with_logits(pair_logits, targets)
        total_loss = (
            args.identity_weight * identity_loss
            + args.contrastive_weight * contrastive_loss
            + args.pair_weight * pair_loss
            + args.verifier_weight * verifier_loss
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()

        with torch.no_grad():
            positive_logit = pair_logits[targets == 1].mean()
            negative_logit = pair_logits[targets == 0].mean()
        count = len(labels)
        examples += count
        values = {
            "total": total_loss,
            "identity": identity_loss,
            "contrastive": contrastive_loss,
            "pair": pair_loss,
            "verifier": verifier_loss,
            "positive_logit": positive_logit,
            "negative_logit": negative_logit,
        }
        for name, value in values.items():
            totals[name] += float(value.detach()) * count
    if examples == 0:
        raise RuntimeError("Training loader produced no batches")
    return {name: value / examples for name, value in totals.items()}


def save_checkpoint(path, model, optimizer, epoch, pairwise_result, cosine_result, args):
    torch.save(
        {
            "architecture": PairwiseVerificationNetwork.architecture_name,
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation_pairwise": vars(pairwise_result),
            "validation_cosine": vars(cosine_result),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        path,
    )


def append_log(path, epoch, learning_rate, metrics, cosine_result, pairwise_result, best):
    row = {
        "epoch": epoch,
        "learning_rate": learning_rate,
        **metrics,
        **{f"cosine_{key}": value for key, value in vars(cosine_result).items()},
        **{f"pairwise_{key}": value for key, value in vars(pairwise_result).items()},
        "best_pairwise_mean_eer": best,
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    fold_dir = args.validation_root / f"fold_{args.fold}"
    fold_data = prepare_fold_data(
        args.face_csv, args.voice_csv, args.manifest, fold_dir
    )
    genders = training_genders(
        fold_data, read_gender_metadata(args.gender_metadata)
    )
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
        pin_memory=device.type == "cuda",
    )
    model = PairwiseVerificationNetwork(
        face_dim=fold_data.train_faces.shape[1],
        voice_dim=fold_data.train_voices.shape[1],
        embedding_dim=args.embedding_dim,
        num_classes=len(fold_data.train_identities),
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
        verifier_hidden_dim=args.verifier_hidden_dim,
        verifier_dropout=args.verifier_dropout,
    ).to(device)
    encoder_params = list(model.encoder.parameters())
    verifier_params = list(model.verifier.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.learning_rate},
            {"params": verifier_params, "lr": args.verifier_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    args.save_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.save_dir / "training_log.csv"
    best_path = args.save_dir / "checkpoint_best.pth.tar"
    if log_path.exists() or best_path.exists():
        raise FileExistsError(f"Refusing to mix with an existing run in {args.save_dir}")

    print(
        f"architecture={model.architecture_name} fold={args.fold} device={device} "
        f"rows={len(fold_data.train_labels)} identities={len(fold_data.train_identities)} "
        f"batch=P{args.identities_per_batch}xK{args.samples_per_identity}="
        f"{sampler.batch_size} balanced_gender={not args.no_gender_balanced} "
        f"hard_negatives_per_anchor={args.hard_negatives_per_anchor}"
    )
    best_eer = float("inf")
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        sampler.set_epoch(epoch)
        metrics = train_one_epoch(model, loader, optimizer, device, args)
        cosine_result, pairwise_result = evaluate_pairwise_fold(
            model,
            fold_data.validation_faces,
            fold_data.validation_voices,
            fold_data.validation_original_indices,
            fold_dir,
            device,
            embed_batch_size=args.eval_batch_size,
        )
        improved = pairwise_result.mean_eer < best_eer - args.min_delta
        if improved:
            best_eer = pairwise_result.mean_eer
            best_epoch = epoch
            stale_epochs = 0
            save_checkpoint(
                best_path,
                model,
                optimizer,
                epoch,
                pairwise_result,
                cosine_result,
                args,
            )
        else:
            stale_epochs += 1
        save_checkpoint(
            args.save_dir / "checkpoint_last.pth.tar",
            model,
            optimizer,
            epoch,
            pairwise_result,
            cosine_result,
            args,
        )
        append_log(
            log_path,
            epoch,
            optimizer.param_groups[0]["lr"],
            metrics,
            cosine_result,
            pairwise_result,
            best_eer,
        )
        print(
            f"epoch={epoch:03d} loss={metrics['total']:.4f} "
            f"verifier={metrics['verifier']:.4f} "
            f"cosine(mean={cosine_result.mean_eer:.4f}, "
            f"gender={cosine_result.gender_eer:.4f}) "
            f"pairwise(mean={pairwise_result.mean_eer:.4f}, "
            f"gender={pairwise_result.gender_eer:.4f}) "
            f"best={best_eer:.4f}@{best_epoch} "
            f"patience={stale_epochs}/{args.patience}"
        )
        if stale_epochs >= args.patience:
            print(f"Early stopping after epoch {epoch}")
            break
    print(
        f"Finished {model.architecture_name} fold {args.fold}; "
        f"best pairwise mean EER={best_eer:.4f} at epoch {best_epoch}. "
        f"Checkpoint: {best_path}"
    )


if __name__ == "__main__":
    main()
