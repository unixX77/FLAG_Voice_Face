#!/usr/bin/env python3
"""eGeMAPS v02 paralinguistic functionals (88 per clip) for every voice in a manifest.

openSMILE's extended Geneva Minimalistic Acoustic Parameter Set describes *how*
a person speaks, largely independently of what is said or in which language:
pitch (F0 level, range, variation), loudness, voice quality (jitter, shimmer,
HNR, H1-H2), vocal effort / spectral balance (alpha ratio, Hammarberg index,
spectral slopes), formants F1-F3 (vocal-tract length) and temporal measures.
Fixed signal processing: no training, no learned parameters.

Manifest formats: the 6-column training format (key, label, voice, face, identity,
class) or the 3-column trial format (key, voice, face). Output: ``key`` + 88
columns, in manifest order. Clips that cannot be processed get empty values.
"""

from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

_SMILE = None


def _init():
    global _SMILE
    import opensmile
    _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                             feature_level=opensmile.FeatureLevel.Functionals)


def _work(path):
    try:
        return _SMILE.process_file(str(path)).to_numpy(dtype=np.float64)[0]
    except Exception:  # noqa: BLE001 - reported as an empty row
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    rows = [line.split() for line in args.manifest.open() if line.strip()]
    voice_col = 2 if len(rows[0]) == 6 else 1
    keys = [r[0] for r in rows]
    paths = [args.audio_root / r[voice_col] for r in rows]

    import opensmile
    names = list(opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                 feature_level=opensmile.FeatureLevel.Functionals).feature_names)
    with Pool(args.workers, initializer=_init) as pool:
        values = []
        for done, v in enumerate(pool.imap(_work, paths, chunksize=16), start=1):
            values.append(v)
            if done % 2000 == 0:
                print(f"{done}/{len(paths)}", flush=True)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["key", *names])
        for key, v in zip(keys, values):
            writer.writerow([key, *([""] * len(names) if v is None else [f"{x:.6g}" for x in v])])
    failed = sum(v is None for v in values)
    report = {"manifest": str(args.manifest), "rows": len(rows), "failed": failed, "features": len(names),
              "feature_set": "eGeMAPSv02 functionals", "opensmile": opensmile.__version__}
    Path(f"{args.output_csv}.report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
