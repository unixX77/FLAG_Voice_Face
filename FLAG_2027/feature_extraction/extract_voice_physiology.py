#!/usr/bin/env python3
"""Extract language-independent physiological voice descriptors.

Pitch and formant statistics reflect larynx size and vocal-tract length,
which relate to body size, age, and facial structure rather than to the words
or language spoken. They complement learned speaker embeddings, which are
trained for identity and also absorb language and channel information.

Features per row (NaN when a recording has too few voiced frames; the CSV
builder imputes them):

    f0_median_log, f0_p10_log, f0_p90_log, f0_std_log, f0_jitter_log,
    voiced_fraction, periodicity,
    f1_log, f2_log, f3_log, f4_log,
    formant_dispersion_log, formant_spacing_log,
    spectral_tilt, spectral_centroid_log

Only numpy and scipy are required. Manifest handling and byte-identical audio
deduplication are shared with ``extract_voice_features_wavlm.py``.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.linalg import solve_toeplitz
from scipy.signal import resample_poly
from tqdm import tqdm

from extract_voice_features_wavlm import (
    TARGET_SAMPLE_RATE,
    hash_audio_paths,
    load_pcm16_wav,
    read_manifest,
    resolve_audio_paths,
)

FEATURE_NAMES = (
    "f0_median_log",
    "f0_p10_log",
    "f0_p90_log",
    "f0_std_log",
    "f0_jitter_log",
    "voiced_fraction",
    "periodicity",
    "f1_log",
    "f2_log",
    "f3_log",
    "f4_log",
    "formant_dispersion_log",
    "formant_spacing_log",
    "spectral_tilt",
    "spectral_centroid_log",
)

HOP = 160  # 10 ms at 16 kHz
F0_FRAME = 1024
F0_MIN, F0_MAX = 60.0, 400.0
YIN_THRESHOLD = 0.2
SILENCE_DB = 35.0
MIN_VOICED_FRAMES = 10
FORMANT_RATE = 10_000
LPC_ORDER = 12
MAX_FORMANT_FRAMES = 400


def frame_signal(signal: np.ndarray, length: int, hop: int) -> np.ndarray:
    if len(signal) < length:
        signal = np.pad(signal, (0, length - len(signal)))
    count = 1 + (len(signal) - length) // hop
    index = np.arange(length)[None, :] + hop * np.arange(count)[:, None]
    return signal[index]


def yin_pitch(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-frame F0 (Hz, NaN if unvoiced), periodicity, and voiced mask."""
    tau_min = int(TARGET_SAMPLE_RATE / F0_MAX)
    tau_max = int(TARGET_SAMPLE_RATE / F0_MIN)
    window = F0_FRAME - tau_max
    frames = frame_signal(waveform, F0_FRAME, HOP).astype(np.float64)
    fft_size = 2 * F0_FRAME
    head = np.zeros_like(frames)
    head[:, :window] = frames[:, :window]
    correlation = np.fft.irfft(
        np.conj(np.fft.rfft(head, fft_size)) * np.fft.rfft(frames, fft_size), fft_size
    )[:, : tau_max + 1]
    squares = np.concatenate(
        (np.zeros((len(frames), 1)), np.cumsum(frames**2, axis=1)), axis=1
    )
    taus = np.arange(tau_max + 1)
    energy_0 = squares[:, window][:, None]
    energy_tau = squares[:, taus + window] - squares[:, taus]
    difference = energy_0 + energy_tau - 2.0 * correlation
    difference[:, 0] = 0.0
    cumulative = np.cumsum(difference[:, 1:], axis=1) / np.arange(1, tau_max + 1)
    normalized = np.ones_like(difference)
    normalized[:, 1:] = difference[:, 1:] / np.maximum(cumulative, 1e-12)

    search = normalized[:, tau_min : tau_max + 1]
    below = search < YIN_THRESHOLD
    # First dip below threshold, then walk to its local minimum.
    first = np.where(below.any(axis=1), below.argmax(axis=1), search.argmin(axis=1))
    best = first.copy()
    for row in range(len(search)):
        position = first[row]
        while position + 1 < search.shape[1] and search[row, position + 1] < search[row, position]:
            position += 1
        best[row] = position
    tau = best + tau_min
    periodicity = 1.0 - normalized[np.arange(len(tau)), tau]

    # Parabolic interpolation for sub-sample lag.
    left = normalized[np.arange(len(tau)), np.clip(tau - 1, 1, tau_max)]
    centre = normalized[np.arange(len(tau)), tau]
    right = normalized[np.arange(len(tau)), np.clip(tau + 1, 1, tau_max)]
    denominator = left - 2 * centre + right
    shift = np.where(np.abs(denominator) > 1e-12, 0.5 * (left - right) / denominator, 0.0)
    refined = tau + np.clip(shift, -1, 1)

    rms_db = 10 * np.log10(np.mean(frames[:, :window] ** 2, axis=1) + 1e-12)
    loud = rms_db > rms_db.max() - SILENCE_DB
    voiced = loud & (normalized[np.arange(len(tau)), tau] < YIN_THRESHOLD)
    f0 = np.where(voiced, TARGET_SAMPLE_RATE / refined, np.nan)
    return f0, periodicity, voiced


def lpc_formants(frame: np.ndarray) -> np.ndarray:
    """Return the four lowest plausible formant frequencies (NaN if missing)."""
    autocorrelation = np.correlate(frame, frame, mode="full")[len(frame) - 1 :]
    if autocorrelation[0] <= 0:
        return np.full(4, np.nan)
    autocorrelation = autocorrelation[: LPC_ORDER + 1].copy()
    autocorrelation[0] *= 1.0 + 1e-9
    coefficients = solve_toeplitz(autocorrelation[:-1], autocorrelation[1:])
    roots = np.roots(np.concatenate(([1.0], -coefficients)))
    roots = roots[np.imag(roots) > 0]
    frequencies = np.angle(roots) * FORMANT_RATE / (2 * np.pi)
    bandwidths = -FORMANT_RATE / np.pi * np.log(np.abs(roots) + 1e-12)
    keep = (frequencies > 90) & (frequencies < FORMANT_RATE / 2 - 50) & (bandwidths < 400)
    formants = np.sort(frequencies[keep])[:4]
    return np.pad(formants, (0, 4 - len(formants)), constant_values=np.nan)


def spectral_shape(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame log-log spectral slope (100-5000 Hz) and spectral centroid."""
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(frames.shape[1]), axis=1)) ** 2
    frequencies = np.fft.rfftfreq(frames.shape[1], 1.0 / TARGET_SAMPLE_RATE)
    band = (frequencies >= 100) & (frequencies <= 5000)
    log_frequency = np.log(frequencies[band])
    log_power = 10 * np.log10(spectrum[:, band] + 1e-12)
    centred = log_frequency - log_frequency.mean()
    slope = (log_power - log_power.mean(axis=1, keepdims=True)) @ centred / (centred @ centred)
    centroid = (spectrum[:, band] * frequencies[band]).sum(axis=1) / (
        spectrum[:, band].sum(axis=1) + 1e-12
    )
    return slope, centroid


def describe(path: Path) -> np.ndarray:
    waveform = load_pcm16_wav(path).astype(np.float64)
    waveform = waveform - waveform.mean()
    f0, periodicity, voiced = yin_pitch(waveform)
    features = np.full(len(FEATURE_NAMES), np.nan)
    features[5] = voiced.mean()
    if voiced.sum() < MIN_VOICED_FRAMES:
        return features
    log_f0 = np.log(f0[voiced])
    features[0] = np.median(log_f0)
    features[1], features[2] = np.percentile(log_f0, [10, 90])
    features[3] = log_f0.std()
    consecutive = voiced[1:] & voiced[:-1]
    if consecutive.any():
        features[4] = np.median(np.abs(np.diff(np.log(f0)))[consecutive])
    features[6] = np.median(periodicity[voiced])

    # Formants on voiced frames, 25 ms windows centred on each pitch frame.
    downsampled = resample_poly(waveform, FORMANT_RATE // 1000, TARGET_SAMPLE_RATE // 1000)
    emphasized = np.append(downsampled[0], downsampled[1:] - 0.97 * downsampled[:-1])
    window_length = int(0.025 * FORMANT_RATE)
    voiced_frames = np.flatnonzero(voiced)
    if len(voiced_frames) > MAX_FORMANT_FRAMES:
        voiced_frames = voiced_frames[
            np.linspace(0, len(voiced_frames) - 1, MAX_FORMANT_FRAMES).astype(int)
        ]
    hamming = np.hamming(window_length)
    formant_rows = []
    for frame_index in voiced_frames:
        centre = int((frame_index * HOP + F0_FRAME // 2) * FORMANT_RATE / TARGET_SAMPLE_RATE)
        start = max(0, centre - window_length // 2)
        segment = emphasized[start : start + window_length]
        if len(segment) == window_length:
            formant_rows.append(lpc_formants(segment * hamming))
    if formant_rows:
        formants = np.asarray(formant_rows)
        complete = np.isfinite(formants).all(axis=1)
        if complete.sum() >= MIN_VOICED_FRAMES:
            formants = formants[complete]
            features[7:11] = np.log(np.median(formants, axis=0))
            features[11] = np.log(np.median((formants[:, 3] - formants[:, 0]) / 3.0))
            # Reby & McComb formant spacing: F_i = (2i - 1) / 2 * spacing.
            positions = (2 * np.arange(1, 5) - 1) / 2.0
            spacing = formants @ positions / (positions @ positions)
            features[12] = np.log(np.median(spacing))

    frames = frame_signal(waveform, 512, HOP)[: len(voiced)]
    slope, centroid = spectral_shape(frames[voiced[: len(frames)]])
    features[13] = np.median(slope)
    features[14] = np.log(np.median(centroid))
    return features


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    rows, is_training = read_manifest(args.manifest)
    if args.limit_rows is not None:
        rows = rows[: args.limit_rows]
    paths = resolve_audio_paths(rows, args.audio_root)
    row_hashes, representative_paths = hash_audio_paths(paths)
    digests = list(representative_paths)
    with Pool(args.workers) as pool:
        results = list(
            tqdm(
                pool.imap(describe, [representative_paths[d] for d in digests], chunksize=8),
                total=len(digests),
                desc="Voice physiology",
                ncols=80,
            )
        )
    by_hash = dict(zip(digests, results))
    matrix = np.stack([by_hash[digest] for digest in row_hashes]).astype(np.float32)
    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, matrix)
    missing = np.isnan(matrix).mean(axis=0)
    report = {
        "manifest": str(args.manifest),
        "audio_root": str(args.audio_root),
        "training_manifest": is_training,
        "rows": len(rows),
        "unique_audio_hashes": len(digests),
        "features": list(FEATURE_NAMES),
        "missing_fraction": dict(zip(FEATURE_NAMES, missing.round(4).tolist())),
        "median": dict(zip(FEATURE_NAMES, np.nanmedian(matrix, axis=0).round(4).tolist())),
    }
    report_path = args.output_npy.with_name(args.output_npy.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
