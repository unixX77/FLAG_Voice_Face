#!/usr/bin/env python3
"""Voice input = V2's 768-d WavLM all-layer mean + 88 eGeMAPS functionals (856-d).

The eGeMAPS block gets a fixed preprocessing fitted on the *official training*
clips only: missing values -> training median, then clipping to the training
[0.5, 99.5] percentile range (eGeMAPS has heavy tails, e.g. on silent or
clipped audio). The projector's own z-scoring is applied afterwards as usual.

Writes the official training CSV (label last), the external combined CSV
(label last, rows aligned with ``train_manifest.txt``) and the evaluation CSVs
(no label column, the format ``generate_cross_modal_submission.py`` reads),
plus symlinks to the unchanged eval face CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_egemaps(paths):
    table = {}
    for path in paths:
        with Path(path).open() as handle:
            reader = csv.reader(handle)
            header = next(reader)
            for row in reader:
                table[row[0]] = np.array([float(x) if x != "" else np.nan for x in row[1:]])
    return header[1:], table


def keys_of(manifest):
    return [line.split()[0] for line in Path(manifest).open() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--egemaps-dir", type=Path, default=Path("features/egemaps"))
    parser.add_argument("--official-voices", type=Path, default=Path("features/voice_v4_wavlm_allmean/train_English_voices.csv"))
    parser.add_argument("--official-manifest", type=Path, default=Path("train_set/train_set/train_English.txt"))
    parser.add_argument("--external-dir", type=Path, default=Path("features/external/combined_all"))
    parser.add_argument("--eval-root", type=Path, default=Path("features/eval_v2"))
    parser.add_argument("--eval-manifests", type=Path, default=Path("features/eval_manifests"))
    parser.add_argument("--out-train", type=Path, default=Path("features/voice_v8_wavlm_egemaps"))
    parser.add_argument("--out-eval", type=Path, default=Path("features/eval_v8"))
    args = parser.parse_args()
    d = args.egemaps_dir

    names, train_table = read_egemaps([d / "train_English.csv"])
    keys = keys_of(args.official_manifest)
    raw = np.stack([train_table[k] for k in keys])
    median = np.nanmedian(raw, 0)
    lo, hi = np.nanpercentile(raw, 0.5, 0), np.nanpercentile(raw, 99.5, 0)

    def prep(x):
        x = np.where(np.isnan(x), median, x)
        return np.clip(x, lo, hi)

    report = {"features": len(names), "missing_official_rows": int(np.isnan(raw).any(1).sum())}
    args.out_train.mkdir(parents=True, exist_ok=True)
    wavlm = np.loadtxt(args.official_voices, delimiter=",", dtype=np.float64)
    out = np.hstack([wavlm[:, :-1], prep(raw), wavlm[:, -1:]])
    np.savetxt(args.out_train / "train_English_voices.csv", out, delimiter=",", fmt="%.8g")

    _, ext_table = read_egemaps([d / f"{s}_train.csv" for s in ("mav3", "vox2test", "mav1")])
    ext_keys = keys_of(args.external_dir / "train_manifest.txt")
    ext_raw = np.stack([ext_table[k] for k in ext_keys])
    wavlm = np.loadtxt(args.external_dir / "train_voices.csv", delimiter=",", dtype=np.float64)
    out = np.hstack([wavlm[:, :-1], prep(ext_raw), wavlm[:, -1:]])
    np.savetxt(args.external_dir / "train_voices_egemaps.csv", out, delimiter=",", fmt="%.8g")
    report["missing_external_rows"] = int(np.isnan(ext_raw).any(1).sum())

    for track in ("no_gender", "gender"):
        for language in ("English", "Bangla"):
            _, table = read_egemaps([d / f"eval_{track}_{language}_valid.csv"])
            k = keys_of(args.eval_manifests / track / f"{language}_valid.txt")
            e_raw = np.stack([table[x] for x in k])
            wavlm = np.loadtxt(args.eval_root / track / f"{language}_valid_voices.csv", delimiter=",", dtype=np.float64)
            if len(wavlm) != len(k):
                raise ValueError(f"eval {track}/{language}: row count mismatch")
            target = args.out_eval / track
            target.mkdir(parents=True, exist_ok=True)
            np.savetxt(target / f"{language}_valid_voices.csv", np.hstack([wavlm, prep(e_raw)]), delimiter=",",
                       fmt="%.8g")
            faces = target / f"{language}_valid_faces.csv"
            if not faces.exists():
                faces.symlink_to((args.eval_root / track / f"{language}_valid_faces.csv").resolve())
            report[f"missing_eval_{track}_{language}"] = int(np.isnan(e_raw).any(1).sum())
    (args.out_train / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
