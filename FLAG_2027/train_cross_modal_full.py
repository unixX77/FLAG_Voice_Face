"""Train the cross-modal architecture (V1 or V2 options) on the complete train set.

Model selection must be completed with identity-disjoint validation before
running this script. Full-data training therefore uses a fixed epoch count and
does not inspect development/test pairs or labels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

try:  # Package execution: python -m FLAG_2027.train_cross_modal_full
    from .cross_modal_model import CrossModalIdentityNetwork
    from .local_validation.cross_modal_losses import (
        paired_cosine_alignment_loss,
        symmetric_cross_modal_contrastive_loss,
    )
    from .local_validation.fold_dataset import load_feature_matrices
    from .local_validation.identity_sampler import IdentityBalancedBatchSampler
    from .train_cross_modal import (LANGUAGES, manifest_language, parse_hidden_dims, read_gender_metadata,
                                    read_traits, resolve_device, set_seed)
    from .trait_heads import LanguageAdversary, TraitHeads, cross_lingual_alignment
except ImportError:  # Direct imports used by local tests
    from cross_modal_model import CrossModalIdentityNetwork
    from local_validation.cross_modal_losses import (
        paired_cosine_alignment_loss,
        symmetric_cross_modal_contrastive_loss,
    )
    from local_validation.fold_dataset import load_feature_matrices
    from local_validation.identity_sampler import IdentityBalancedBatchSampler
    from train_cross_modal import (LANGUAGES, manifest_language, parse_hidden_dims, read_gender_metadata,
                                   read_traits, resolve_device, set_seed)
    from trait_heads import LanguageAdversary, TraitHeads, cross_lingual_alignment


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-csv", type=Path, required=True)
    parser.add_argument("--voice-csv", type=Path, required=True)
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--identities-per-batch", type=int, default=16)
    parser.add_argument("--samples-per-identity", type=int, default=8)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--arc-scale", type=float, default=30.0)
    parser.add_argument("--arc-margin", type=float, default=0.2)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--identity-weight", type=float, default=1.0)
    parser.add_argument("--contrastive-weight", type=float, default=1.0)
    parser.add_argument("--pair-weight", type=float, default=0.25)
    parser.add_argument(
        "--contrastive-mode", choices=("multi_positive", "supcon"), default="multi_positive"
    )
    parser.add_argument("--face-hidden-dims", type=parse_hidden_dims, default=(1024, 512))
    parser.add_argument("--voice-hidden-dims", type=parse_hidden_dims, default=(512, 512))
    parser.add_argument("--hidden-dropout", type=float)
    parser.add_argument("--standardize-inputs", action="store_true")
    parser.add_argument("--input-dropout", type=float, default=0.0)
    parser.add_argument("--input-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--architecture-name",
        default="cross_modal_identity_network_v1_full",
        help="Label stored in the checkpoint, e.g. cross_modal_identity_network_v2_full",
    )
    parser.add_argument("--extra-face-csv", type=Path,
                        help="external training faces (label = last column, classes disjoint from official)")
    parser.add_argument("--extra-voice-csv", type=Path, help="external training voices, rows aligned")
    parser.add_argument("--official-share", type=float,
                        help="fraction of each batch's identities drawn from the official identities")
    # Voice -> traits -> face auxiliary losses (training only; off by default).
    parser.add_argument("--manifest", type=Path, default=Path("train_set/train_set/train_English.txt"))
    parser.add_argument("--gender-metadata", type=Path, default=Path("meta_file_train_set.csv"))
    parser.add_argument("--extra-manifest", type=Path, help="external manifest, rows aligned with --extra-face-csv")
    parser.add_argument("--extra-gender-metadata", type=Path)
    parser.add_argument("--trait-csv", type=Path)
    parser.add_argument("--extra-trait-csv", type=Path)
    parser.add_argument("--trait-weight", type=float, default=0.0)
    parser.add_argument("--trait-targets", default="age,build,gender")
    parser.add_argument("--language-adversary-weight", type=float, default=0.0)
    parser.add_argument("--language-adversary-scale", type=float, default=1.0)
    parser.add_argument("--cross-lingual-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")
    if args.identities_per_batch < 2 or args.samples_per_identity < 2:
        raise ValueError("P and K must both be at least 2")
    if args.embedding_dim < 1 or args.contrastive_temperature <= 0:
        raise ValueError("Embedding dimension and temperature must be positive")
    weights = (args.identity_weight, args.contrastive_weight, args.pair_weight)
    if any(weight < 0 for weight in weights) or sum(weights) == 0:
        raise ValueError("Loss weights must be non-negative and not all zero")


def contiguous_labels(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    classes, remapped = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("Full-data training requires at least two identities")
    return np.ascontiguousarray(remapped, dtype=np.int64), classes


def main() -> None:
    args = parse_arguments()
    validate_arguments(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    faces, voices, original_labels = load_feature_matrices(
        args.face_csv, args.voice_csv
    )
    official_classes = np.unique(original_labels)
    if args.extra_face_csv is not None:
        extra_faces, extra_voices, extra_labels = load_feature_matrices(args.extra_face_csv, args.extra_voice_csv)
        if np.isin(np.unique(extra_labels), official_classes).any():
            raise ValueError("External class indices overlap official ones")
        faces = np.ascontiguousarray(np.concatenate([faces, extra_faces]))
        voices = np.ascontiguousarray(np.concatenate([voices, extra_voices]))
        original_labels = np.concatenate([original_labels, extra_labels])
    elif args.official_share is not None:
        raise ValueError("--official-share needs external data")
    labels, original_classes = contiguous_labels(original_labels)
    use_aux = args.trait_weight > 0 or args.language_adversary_weight > 0 or args.cross_lingual_weight > 0
    tensors = [torch.from_numpy(faces), torch.from_numpy(voices), torch.from_numpy(labels)]
    language_weights = None
    if use_aux:
        # Per row: age_z, build_z, language, gender -- same construction as the fold runs.
        rows = [line.split() for line in args.manifest.open() if line.strip()]
        if args.extra_face_csv is not None:
            if args.extra_manifest is None or args.extra_gender_metadata is None:
                raise ValueError("Auxiliary losses with external data need --extra-manifest and --extra-gender-metadata")
            rows += [line.split() for line in args.extra_manifest.open() if line.strip()]
        if len(rows) != len(labels):
            raise ValueError("Manifests are not aligned with the feature rows")
        gender_of = read_gender_metadata(args.gender_metadata)
        if args.extra_gender_metadata is not None:
            gender_of.update(read_gender_metadata(args.extra_gender_metadata))
        traits = read_traits(args.trait_csv) if args.trait_csv else {}
        if args.extra_trait_csv:
            traits.update(read_traits(args.extra_trait_csv))
        values = np.array([traits.get(r[0], (np.nan, np.nan)) for r in rows], dtype=np.float64)
        if args.trait_weight > 0 and np.isnan(values).any(1).mean() > 0.05:
            raise ValueError("More than 5% of rows have no trait labels")
        values = (values - np.nanmean(values, 0)) / np.where(np.nanstd(values, 0) > 0, np.nanstd(values, 0), 1)
        language = np.array([manifest_language(r[2]) for r in rows])
        gender = np.array([gender_of[r[4]] for r in rows])
        aux = np.column_stack([values, language, gender]).astype(np.float32)
        tensors.append(torch.from_numpy(aux))
        counts = np.bincount(language[language >= 0], minlength=len(LANGUAGES))
        language_weights = torch.tensor(
            np.where(counts > 0, counts.sum() / np.maximum(counts, 1) / (counts > 0).sum(), 0.0), dtype=torch.float32)
        args.language_counts = counts.tolist()
    dataset = TensorDataset(*tensors)
    sampler = IdentityBalancedBatchSampler(
        labels,
        args.identities_per_batch,
        args.samples_per_identity,
        args.seed,
        batches_per_epoch=args.batches_per_epoch,
        primary_identities=np.flatnonzero(np.isin(original_classes, official_classes)),
        primary_share=args.official_share,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    # Everything needed to rebuild the network at inference time.
    model_kwargs = dict(
        face_dim=faces.shape[1],
        voice_dim=voices.shape[1],
        embedding_dim=args.embedding_dim,
        num_classes=len(original_classes),
        arc_scale=args.arc_scale,
        arc_margin=args.arc_margin,
        face_hidden_dims=tuple(args.face_hidden_dims),
        voice_hidden_dims=tuple(args.voice_hidden_dims),
        hidden_dropout=args.hidden_dropout,
        standardize_inputs=args.standardize_inputs,
        input_dropout=args.input_dropout,
        input_noise_std=args.input_noise_std,
    )
    model = CrossModalIdentityNetwork(**model_kwargs)
    if args.standardize_inputs:
        model.face_branch.set_input_statistics(
            torch.from_numpy(faces.mean(axis=0)), torch.from_numpy(faces.std(axis=0))
        )
        model.voice_branch.set_input_statistics(
            torch.from_numpy(voices.mean(axis=0)), torch.from_numpy(voices.std(axis=0))
        )
    model = model.to(device)
    aux_modules = None
    if use_aux:
        aux_modules = torch.nn.ModuleDict({
            "traits": TraitHeads(args.embedding_dim),
            "language": LanguageAdversary(args.embedding_dim, len(LANGUAGES)),
        }).to(device)
        aux_modules["language"].register_buffer("class_weights", language_weights.to(device))
    trainable = list(model.parameters()) + ([] if aux_modules is None else list(aux_modules.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.save_dir / "checkpoint_full.pth.tar"
    log_path = args.save_dir / "training_log.csv"
    if checkpoint_path.exists() or log_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing full-data run in {args.save_dir}"
        )

    print(
        f"architecture={args.architecture_name} device={device} "
        f"rows={len(labels)} identities={len(original_classes)} "
        f"face_dim={faces.shape[1]} voice_dim={voices.shape[1]} "
        f"batch=P{args.identities_per_batch}xK{args.samples_per_identity}="
        f"{sampler.batch_size} batches/epoch={len(sampler)} epochs={args.epochs}"
    )
    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        if aux_modules is not None:
            aux_modules.train()
        totals = {"total": 0.0, "identity": 0.0, "contrastive": 0.0, "pair": 0.0, "trait": 0.0,
                  "language_adversary": 0.0, "cross_lingual": 0.0}
        examples = 0
        for batch in loader:
            face_batch, voice_batch, label_batch = batch[:3]
            face_batch = face_batch.to(device, non_blocking=True)
            voice_batch = voice_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)
            face_embedding, voice_embedding = model(face_batch, voice_batch)
            identity_loss = 0.5 * (
                F.cross_entropy(model.identity_logits(face_embedding, label_batch), label_batch)
                + F.cross_entropy(model.identity_logits(voice_embedding, label_batch), label_batch)
            )
            contrastive_loss, _ = symmetric_cross_modal_contrastive_loss(
                face_embedding,
                voice_embedding,
                label_batch,
                args.contrastive_temperature,
                args.contrastive_mode,
            )
            pair_loss = paired_cosine_alignment_loss(face_embedding, voice_embedding)
            trait_loss = language_loss = cross_lingual_loss = pair_loss * 0.0
            if aux_modules is not None:
                aux = batch[3].to(device, non_blocking=True)
                age, build, language, gender = aux[:, 0], aux[:, 1], aux[:, 2].long(), aux[:, 3]
                if args.trait_weight > 0:
                    targets = tuple(args.trait_targets.split(","))
                    heads = aux_modules["traits"]
                    trait_loss = 0.5 * (heads.loss(face_embedding, age, build, gender, targets)
                                        + heads.loss(voice_embedding, age, build, gender, targets))
                if args.language_adversary_weight > 0:
                    language_loss, _ = aux_modules["language"].loss(
                        voice_embedding, language, args.language_adversary_scale,
                        aux_modules["language"].class_weights)
                if args.cross_lingual_weight > 0:
                    cross_lingual_loss = cross_lingual_alignment(voice_embedding, label_batch, language)
            total_loss = (
                args.identity_weight * identity_loss
                + args.contrastive_weight * contrastive_loss
                + args.pair_weight * pair_loss
                + args.trait_weight * trait_loss
                + args.language_adversary_weight * language_loss
                + args.cross_lingual_weight * cross_lingual_loss
            )
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if args.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
            optimizer.step()
            count = len(label_batch)
            examples += count
            for name, value in (
                ("total", total_loss),
                ("identity", identity_loss),
                ("contrastive", contrastive_loss),
                ("pair", pair_loss),
                ("trait", trait_loss),
                ("language_adversary", language_loss),
                ("cross_lingual", cross_lingual_loss),
            ):
                totals[name] += float(value.detach()) * count
        metrics = {name: value / examples for name, value in totals.items()}
        row = {"epoch": epoch, **metrics}
        with log_path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(row)
        print(
            f"epoch={epoch:03d}/{args.epochs:03d} loss={metrics['total']:.4f} "
            f"identity={metrics['identity']:.4f} "
            f"contrastive={metrics['contrastive']:.4f} pair={metrics['pair']:.4f}"
        )

    torch.save(
        {
            "architecture": args.architecture_name,
            "model_kwargs": model_kwargs,
            "epoch": args.epochs,
            "state_dict": model.state_dict(),
            "aux_state_dict": None if aux_modules is None else aux_modules.state_dict(),
            "face_dim": faces.shape[1],
            "voice_dim": voices.shape[1],
            "embedding_dim": args.embedding_dim,
            "num_classes": len(original_classes),
            "original_classes": original_classes.tolist(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        },
        checkpoint_path,
    )
    print(f"Saved full-data checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
