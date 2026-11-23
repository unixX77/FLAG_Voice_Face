#!/usr/bin/env python3
"""Extract augmented views of the V2 voice feature (WavLM all-layer mean).

For every manifest row, ``--views`` augmented copies of the waveform are encoded
with exactly the V2 recipe: the frame mean of every WavLM hidden state,
averaged over all 13 layers (10 s chunks, 5 s overlap, frame-weighted). Output
shape is ``(rows, views, hidden_dim)`` as float16 NumPy.

Augmentations keep the speaker's identity cues: no pitch shift or speed change.
Each view always takes a random segment of 50-100 % of the clip (at least
1.5 s). It then applies, independently and at random:
synthetic coloured noise (5-20 dB SNR), a synthetic room impulse response
(RT60 0.2-0.7 s), a telephone or low-pass band limit, and 8-bit mu-law coding.
No external audio is used. Every view is seeded from (seed, row, view), so the
output is reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, fftconvolve, sosfiltfilt
from tqdm import tqdm

from extract_voice_features_wavlm import (
    TARGET_SAMPLE_RATE,
    load_pcm16_wav,
    read_manifest,
    resolve_audio_paths,
    split_chunks,
)
from extract_voice_features_wavlm_layers import pooled_hidden_states

SR = TARGET_SAMPLE_RATE


def random_segment(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    minimum = int(1.5 * SR)
    if len(wave) <= minimum:
        return wave
    length = max(minimum, int(len(wave) * rng.uniform(0.5, 1.0)))
    start = rng.integers(0, len(wave) - length + 1)
    return wave[start : start + length]


def coloured_noise(n: int, exponent: float, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    frequencies = np.fft.rfftfreq(n)
    frequencies[0] = frequencies[1]
    noise = np.fft.irfft(spectrum / frequencies ** (exponent / 2), n)
    return noise / (noise.std() + 1e-12)


def add_noise(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    snr_db = rng.uniform(5.0, 20.0)
    noise = coloured_noise(len(wave), rng.choice([0.0, 1.0, 2.0]), rng)
    return wave + noise * wave.std() / (10 ** (snr_db / 20))


def add_reverb(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    rt60 = rng.uniform(0.2, 0.7)
    t = np.arange(int(rt60 * SR)) / SR
    impulse = rng.standard_normal(len(t)) * np.exp(-6.9 * t / rt60)
    impulse[0] = 1.0 / rng.uniform(0.3, 1.0)  # direct path versus reflections
    wet = fftconvolve(wave, impulse)[: len(wave)]
    return wet * wave.std() / (wet.std() + 1e-12)


def band_limit(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        sos = butter(6, (300, 3400), btype="bandpass", fs=SR, output="sos")
    else:
        sos = butter(6, rng.uniform(3000, 6000), btype="lowpass", fs=SR, output="sos")
    return sosfiltfilt(sos, wave)


def mu_law(wave: np.ndarray) -> np.ndarray:
    peak = np.abs(wave).max() + 1e-12
    x = wave / peak
    encoded = np.round((np.sign(x) * np.log1p(255 * np.abs(x)) / np.log(256) + 1) / 2 * 255)
    y = encoded / 255 * 2 - 1
    return np.sign(y) * ((256 ** np.abs(y)) - 1) / 255 * peak


def augment(wave: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    wave = random_segment(wave, rng).astype(np.float64)
    if rng.random() < 0.4:
        wave = add_reverb(wave, rng)
    if rng.random() < 0.6:
        wave = add_noise(wave, rng)
    if rng.random() < 0.3:
        wave = band_limit(wave, rng)
    if rng.random() < 0.2:
        wave = mu_law(wave)
    return wave.astype(np.float32)


def encode(model, feature_extractor, wave, device, chunk_samples, overlap_samples, use_fp16):
    """V2 voice feature: frame-weighted mean over chunks, then mean over layers."""
    total, frames = None, 0
    for chunk in split_chunks(wave, chunk_samples, overlap_samples):
        pooled, chunk_frames = pooled_hidden_states(model, feature_extractor, chunk, device, use_fp16)
        weighted = pooled * chunk_frames
        total = weighted if total is None else total + weighted
        frames += chunk_frames
    pooled = (total / frames).numpy()
    hidden = pooled.shape[1] // 2
    return pooled[:, :hidden].mean(axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--views", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true", help="encode clean audio (pipeline check)")
    parser.add_argument("--chunk-seconds", type=float, default=10.0)
    parser.add_argument("--overlap-seconds", type=float, default=5.0)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--limit-rows", type=int)
    args = parser.parse_args()
    from transformers import Wav2Vec2FeatureExtractor, WavLMModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, _ = read_manifest(args.manifest)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    paths = resolve_audio_paths(rows, args.audio_root)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model, revision=args.revision)
    model = WavLMModel.from_pretrained(args.model, revision=args.revision).to(device).eval()
    chunk_samples = int(round(args.chunk_seconds * SR))
    overlap_samples = int(round(args.overlap_seconds * SR))
    views = 1 if args.no_augment else args.views

    output = None
    for row, path in enumerate(tqdm(paths, desc="Encoding views", ncols=80)):
        wave = load_pcm16_wav(path)
        for view in range(views):
            rng = np.random.default_rng([args.seed, row, view])
            audio = wave if args.no_augment else augment(wave, rng)
            feature = encode(model, feature_extractor, audio, device, chunk_samples,
                             overlap_samples, not args.no_fp16)
            if output is None:
                output = np.zeros((len(paths), views, len(feature)), dtype=np.float32)
            output[row, view] = feature
    if not np.isfinite(output).all():
        raise RuntimeError("Non-finite augmented feature")
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, output.astype(np.float16))
    report = {
        "manifest": str(args.manifest), "audio_root": str(args.audio_root), "model": args.model,
        "revision": args.revision, "shape": list(output.shape), "augmented": not args.no_augment,
        "seed": args.seed, "feature": "WavLM all-layer (0-12) mean of frame means, as V2",
        "augmentations": {"segment": "50-100% of clip, >=1.5 s, always", "reverb": "p=0.4, RT60 0.2-0.7 s",
                          "noise": "p=0.6, white/pink/brown, SNR 5-20 dB",
                          "band_limit": "p=0.3, 300-3400 Hz or low-pass 3-6 kHz", "mu_law": "p=0.2, 8 bit"},
    }
    args.output_npy.with_name(args.output_npy.stem + "_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("shape", "augmented")}))


if __name__ == "__main__":
    main()
