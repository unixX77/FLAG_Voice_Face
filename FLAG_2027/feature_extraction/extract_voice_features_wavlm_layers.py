#!/usr/bin/env python3
"""Extract pooled WavLM hidden states from every transformer layer.

The x-vector head of ``wavlm-base-plus-sv`` is trained to keep only speaker
identity. The intermediate layers retain broader voice traits (pitch, timbre,
age, vocal effort) that are more likely to be shared with the face. This
script stores, for every manifest row, the frame mean and standard deviation of
all hidden states, shape ``(rows, layers, 2 * hidden_dim)``, as float16 NumPy.
``build_voice_feature_csv.py`` then selects layers and writes the CSV format
used by the training scripts.

Manifest handling and byte-identical audio deduplication are shared with
``extract_voice_features_wavlm.py``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from extract_voice_features_wavlm import (
    TARGET_SAMPLE_RATE,
    hash_audio_paths,
    load_pcm16_wav,
    read_manifest,
    resolve_audio_paths,
    split_chunks,
)


def pooled_hidden_states(
    model, feature_extractor, waveform: np.ndarray, device: torch.device, use_fp16: bool
) -> tuple[torch.Tensor, int]:
    """Return per-layer [mean, std] over frames and the number of frames."""
    inputs = feature_extractor(
        waveform, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt"
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=use_fp16 and device.type == "cuda",
    ):
        outputs = model(inputs["input_values"].to(device), output_hidden_states=True)
    # (layers, frames, hidden)
    hidden = torch.stack(outputs.hidden_states, dim=0)[:, 0].float()
    pooled = torch.cat((hidden.mean(dim=1), hidden.std(dim=1, unbiased=False)), dim=1)
    return pooled.cpu(), hidden.shape[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument(
        "--revision",
        required=True,
        help="Pin a Hugging Face commit hash for reproducible extraction",
    )
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=5.0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    rows, is_training = read_manifest(args.manifest)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    paths = resolve_audio_paths(rows, args.audio_root)
    row_hashes, representative_paths = hash_audio_paths(paths)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        args.model, revision=args.revision
    )
    model = WavLMModel.from_pretrained(args.model, revision=args.revision).to(device)
    model.eval()

    chunk_samples = int(round(args.chunk_seconds * TARGET_SAMPLE_RATE))
    overlap_samples = int(round(args.overlap_seconds * TARGET_SAMPLE_RATE))
    use_fp16 = not args.no_fp16
    pooled_by_hash: dict[str, np.ndarray] = {}
    chunk_histogram: Counter[int] = Counter()
    for digest, path in tqdm(
        representative_paths.items(), desc="Encoding unique audio", ncols=80
    ):
        chunks = split_chunks(load_pcm16_wav(path), chunk_samples, overlap_samples)
        chunk_histogram[len(chunks)] += 1
        total, frames = None, 0
        for chunk in chunks:
            pooled, chunk_frames = pooled_hidden_states(
                model, feature_extractor, chunk, device, use_fp16
            )
            # Frame-weighted average of chunk statistics.
            weighted = pooled * chunk_frames
            total = weighted if total is None else total + weighted
            frames += chunk_frames
        pooled_by_hash[digest] = (total / frames).numpy()

    matrix = np.stack([pooled_by_hash[digest] for digest in row_hashes])
    if not np.isfinite(matrix).all():
        raise RuntimeError("Non-finite pooled hidden state")
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, matrix.astype(np.float16))
    if is_training:
        labels = np.asarray([row.class_index for row in rows], dtype=np.int64)
        np.save(args.output_npy.with_name(args.output_npy.stem + "_labels.npy"), labels)

    report = {
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "output_npy": str(args.output_npy),
        "model": args.model,
        "revision": args.revision,
        "training_manifest": is_training,
        "limited_rows": args.limit_rows,
        "rows": len(rows),
        "unique_audio_hashes": len(representative_paths),
        "shape": list(matrix.shape),
        "layout": "(rows, hidden_states incl. CNN output at index 0, [mean, std] * hidden_dim)",
        "chunk_seconds": args.chunk_seconds,
        "overlap_seconds": args.overlap_seconds,
        "fp16": use_fp16 and device.type == "cuda",
        "chunk_count_histogram": {
            str(key): value for key, value in sorted(chunk_histogram.items())
        },
    }
    report_path = args.output_npy.with_name(args.output_npy.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
