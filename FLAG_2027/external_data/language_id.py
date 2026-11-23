#!/usr/bin/env python3
"""Frozen spoken-language identification for every clip of a training manifest.

Model: SpeechBrain ``lang-id-voxlingua107-ecapa`` (Apache-2.0, 107 languages,
pinned revision). Used only to *remove* Bengali speech from external training
data, as required by the FLAG 2027 rules. Each clip is classified as a whole
(up to ``--max-seconds``, the first part of the clip).

Writes ``row, top1, p_top1, p_bengali, p_assamese, p_hindi, p_urdu, drop`` per
row. A clip is dropped when Bengali is the top language or its probability is
at least ``--bengali-threshold`` (a deliberately low, conservative threshold).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feature_extraction"))
from extract_voice_features_wavlm import load_pcm16_wav, read_manifest, resolve_audio_paths  # noqa: E402

SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"
REVISION = "0253049ae131d6a4be1c4f0d8b0ff483a0f8c8e9"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("models/hf_cache"))
    parser.add_argument("--max-seconds", type=float, default=60.0)
    parser.add_argument("--bengali-threshold", type=float, default=0.05)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from speechbrain.inference.classifiers import EncoderClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = snapshot_download(SOURCE, revision=REVISION, cache_dir=str(args.cache_dir))
    model = EncoderClassifier.from_hparams(source=local, savedir=str(args.cache_dir / "langid_savedir"),
                                           overrides={"pretrained_path": local}, run_opts={"device": device})
    labels = [model.hparams.label_encoder.decode_ndim(i) for i in range(len(model.hparams.label_encoder))]
    code = [label.split(":")[0] for label in labels]
    watch = {name: code.index(name) for name in ("bn", "as", "hi", "ur")}

    rows, _ = read_manifest(args.manifest)
    paths = resolve_audio_paths(rows, args.audio_root)
    limit = int(args.max_seconds * 16000)
    out = []
    for row, path in zip(rows, tqdm(paths, desc="Language ID", ncols=80)):
        wave = torch.from_numpy(load_pcm16_wav(path)[:limit]).float().unsqueeze(0)
        with torch.inference_mode():
            log_probs = model.classify_batch(wave.to(device))[0]
        p = torch.softmax(log_probs.reshape(-1), 0).cpu().numpy()
        top = int(np.argmax(p))
        drop = top == watch["bn"] or p[watch["bn"]] >= args.bengali_threshold
        out.append([row.row, code[top], f"{p[top]:.4f}", *(f"{p[watch[k]]:.4f}" for k in ("bn", "as", "hi", "ur")),
                    int(drop)])

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "top1", "p_top1", "p_bengali", "p_assamese", "p_hindi", "p_urdu", "drop"])
        writer.writerows(out)
    tops = {}
    for r in out:
        tops[r[1]] = tops.get(r[1], 0) + 1
    report = {"model": SOURCE, "revision": REVISION, "manifest": str(args.manifest), "rows": len(out),
              "dropped": sum(r[-1] for r in out), "bengali_threshold": args.bengali_threshold,
              "top1_counts": dict(sorted(tops.items(), key=lambda kv: -kv[1])[:15])}
    Path(f"{args.output_csv}.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
