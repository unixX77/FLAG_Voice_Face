#!/usr/bin/env python3
"""Person-level trait labels (age, build) for every training row, from frozen face models only.

* age: median InsightFace ``buffalo_l`` apparent age over the identity's face images;
* build: mean FaRL zero-shot "heavy-set vs slim" score over the identity's images.

Person-level (not per image) because age and build are properties of the person
and single-image estimates are noisy (about +-6 years). Training data only.
Output: ``key,identity,age,build`` in manifest order.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def person_level(identities, values, reducer):
    groups = defaultdict(list)
    for identity, v in zip(identities, values):
        if np.isfinite(v):
            groups[identity].append(v)
    return {i: float(reducer(v)) for i, v in groups.items()}


def write(path, keys, identities, age, build):
    with path.open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(["key", "identity", "age", "build"])
        for k, i in zip(keys, identities):
            w.writerow([k, i, f"{age.get(i, np.nan):.3f}", f"{build.get(i, np.nan):.6f}"])


def to_float(x):
    return float(x) if x not in ("", None) else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-manifest", type=Path, default=Path("train_set/train_set/train_English.txt"))
    parser.add_argument("--official-age", type=Path, default=Path("features/face_age/train_English_face_age.csv"))
    parser.add_argument("--official-build", type=Path, default=Path("features/face_build/train_English_farl_attributes.csv"))
    parser.add_argument("--external-manifest", type=Path, default=Path("features/external/combined_all/train_manifest.txt"))
    parser.add_argument("--external-build", type=Path, default=Path("features/traits/combined_all_farl_attributes.csv"))
    parser.add_argument("--external-age", type=Path, nargs="+", default=[
        Path("features/external/mav3/train_face_age.csv"), Path("features/external/vox2test/train_face_age.csv"),
        Path("features/external/mav1/train_face_age.csv")])
    parser.add_argument("--output-dir", type=Path, default=Path("features/traits"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [line.split() for line in args.official_manifest.open() if line.strip()]
    keys, ids = [r[0] for r in rows], [r[4] for r in rows]
    age_rows = list(csv.DictReader(args.official_age.open()))
    build_rows = list(csv.DictReader(args.official_build.open()))
    if [r["key"] for r in age_rows] != keys or len(build_rows) != len(rows):
        raise ValueError("Official attribute files are not aligned with the manifest")
    age = person_level(ids, [to_float(r["age"]) for r in age_rows], np.median)
    build = person_level(ids, [to_float(r["build"]) for r in build_rows], np.mean)
    write(args.output_dir / "train_English_traits.csv", keys, ids, age, build)

    rows = [line.split() for line in args.external_manifest.open() if line.strip()]
    keys, ids = [r[0] for r in rows], [r[4] for r in rows]
    age_by_key = {}
    for path in args.external_age:
        for r in csv.DictReader(path.open()):
            age_by_key[r["key"]] = to_float(r["age"])
    build_rows = list(csv.DictReader(args.external_build.open()))
    if len(build_rows) != len(rows):
        raise ValueError("External build scores are not aligned with the combined manifest")
    missing = sum(k not in age_by_key for k in keys)
    if missing:
        raise ValueError(f"{missing} external rows have no face-age estimate")
    age = person_level(ids, [age_by_key[k] for k in keys], np.median)
    build = person_level(ids, [to_float(r["build"]) for r in build_rows], np.mean)
    write(args.output_dir / "combined_all_traits.csv", keys, ids, age, build)
    print(f"official identities={len(set(r[4] for r in [l.split() for l in args.official_manifest.open() if l.strip()]))} "
          f"external identities={len(age)}; external age median={np.median(list(age.values())):.1f}")


if __name__ == "__main__":
    main()
