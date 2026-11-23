#!/usr/bin/env python3
"""Extract per-layer ViT face features (FaRL or DINOv2), with a flipped copy.

The final FaRL embedding is trained to match captions, so it summarizes a face
coarsely. Intermediate transformer layers keep finer appearance cues (age,
build, face shape) that may be shared with the voice, mirroring what the
layer-wise WavLM features showed on the voice side.

For every manifest row and every hidden state (index 0 = patch embeddings,
1..12 = transformer blocks) the script stores ``[CLS token, mean patch
token]``, shape ``(rows, 13, 2 * 768)``, as float16 NumPy. The same features
for the horizontally flipped image are written to ``*_flip.npy`` so that
flip test-time augmentation can be applied when building CSVs.

Manifest handling, deduplication, and the FaRL loading recipe are shared with
``extract_face_features_farl.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from extract_face_features_farl import (
    hash_image_paths,
    load_farl_model,
    read_manifest,
    resolve_image_paths,
)


class FlipPairDataset(Dataset):
    def __init__(self, paths: list[Path], preprocess) -> None:
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            return self.preprocess(image), self.preprocess(ImageOps.mirror(image))


def pool(tokens: torch.Tensor) -> torch.Tensor:
    """(batch, tokens, width) -> (batch, 2 * width) as [CLS, mean patch]."""
    return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)


class FarlLayers:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        import clip

        self.model, self.preprocess, _, _ = load_farl_model(clip, checkpoint, device)
        visual = self.model.visual
        self.captured: list[torch.Tensor] = []
        # ln_pre output is (batch, tokens, width); OpenAI CLIP blocks run in
        # (tokens, batch, width) layout.
        visual.ln_pre.register_forward_hook(
            lambda _m, _i, out: self.captured.append(out.float())
        )
        for block in visual.transformer.resblocks:
            block.register_forward_hook(
                lambda _m, _i, out: self.captured.append(out.permute(1, 0, 2).float())
            )

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        self.captured.clear()
        self.model.encode_image(images)
        return torch.stack([pool(tokens) for tokens in self.captured], dim=1)


class Dinov2Layers:
    def __init__(self, model_name: str, revision: str, device: torch.device) -> None:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name, revision=revision)
        self.model = AutoModel.from_pretrained(model_name, revision=revision).to(device).eval()
        self.preprocess = lambda image: processor(
            images=image, return_tensors="pt"
        )["pixel_values"][0]

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.model(pixel_values=images, output_hidden_states=True).hidden_states
        return torch.stack([pool(tokens.float()) for tokens in hidden], dim=1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("farl", "dinov2"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="FaRL checkpoint")
    parser.add_argument("--model", default="facebook/dinov2-base")
    parser.add_argument("--revision", default="f9e44c814b77203eaa57a6bdbbd535f21ede1415")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    rows, is_training = read_manifest(args.manifest)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    paths = resolve_image_paths(rows, args.image_root)
    row_hashes, representatives = hash_image_paths(paths)
    if args.backend == "farl":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for the FaRL backend")
        encoder = FarlLayers(args.checkpoint, device)
    else:
        encoder = Dinov2Layers(args.model, args.revision, device)

    digests = list(representatives)
    loader = DataLoader(
        FlipPairDataset([representatives[d] for d in digests], encoder.preprocess),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    use_fp16 = not args.no_fp16 and device.type == "cuda"
    plain_parts, flip_parts = [], []
    for images, flipped in tqdm(loader, desc=f"Encoding faces ({args.backend})", ncols=80):
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, dtype=torch.float16, enabled=use_fp16
        ):
            plain_parts.append(encoder(images.to(device)).cpu().numpy())
            flip_parts.append(encoder(flipped.to(device)).cpu().numpy())
    position = {digest: index for index, digest in enumerate(digests)}
    order = np.asarray([position[digest] for digest in row_hashes])
    plain = np.concatenate(plain_parts)[order]
    flipped = np.concatenate(flip_parts)[order]
    if not (np.isfinite(plain).all() and np.isfinite(flipped).all()):
        raise RuntimeError("Non-finite face feature")

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, plain.astype(np.float16))
    np.save(args.output_npy.with_name(args.output_npy.stem + "_flip.npy"), flipped.astype(np.float16))
    report = {
        "backend": args.backend,
        "manifest": str(args.manifest),
        "image_root": str(args.image_root),
        "checkpoint": str(args.checkpoint),
        "model": args.model if args.backend == "dinov2" else "FaRL-Base-Patch16-LAIONFace20M-ep16",
        "revision": args.revision if args.backend == "dinov2" else None,
        "training_manifest": is_training,
        "limited_rows": args.limit_rows,
        "rows": len(rows),
        "unique_image_hashes": len(digests),
        "shape": list(plain.shape),
        "layout": "(rows, hidden_states incl. patch embeddings at 0, [CLS, mean patch] * width)",
        "fp16": use_fp16,
    }
    args.output_npy.with_name(args.output_npy.stem + "_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
