#!/usr/bin/env python3
"""Merge external sources into one filtered training set (official data untouched).

For every source (``features/external/<name>/`` + manifest + language CSV +
face-age CSV) this

1. drops the clips the language-ID step flagged (Bengali);
2. assigns one gender per identity from two frozen/training-only cues: the
   InsightFace face-gender majority, and a logistic-regression voice-gender
   classifier fitted on the *official training* WavLM features and metadata.
   Identities where the two disagree are reported and need an entry in
   ``--gender-overrides`` (identity -> m/f, from manual inspection) or are
   dropped;
3. finds duplicate identities with FaRL face centroids: against the official
   training identities (the external copy is dropped) and between external
   identities (merged under the first name). The threshold is calibrated on
   official training data only: centroid similarity between two disjoint halves
   of the same identity versus between different identities;
4. writes ``<out>/train_faces.csv``, ``train_voices.csv`` (label = new class
   index), ``train_manifest.txt``, ``gender.csv`` and ``report.json``.

No development or evaluation data is read.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def read_rows(manifest):
    return [line.split() for line in Path(manifest).open() if line.strip()]


def centroid(x):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    c = x.mean(0)
    return c / np.linalg.norm(c)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True, help="source names, e.g. mav3 mav1")
    parser.add_argument("--official-faces", type=Path, default=Path("features/face_v2_farl/train_English_faces.csv"))
    parser.add_argument("--official-voices", type=Path,
                        default=Path("features/voice_v4_wavlm_allmean/train_English_voices.csv"))
    parser.add_argument("--official-manifest", type=Path, default=Path("train_set/train_set/train_English.txt"))
    parser.add_argument("--official-gender", type=Path, default=Path("meta_file_train_set.csv"))
    parser.add_argument("--gender-overrides", type=Path, help="JSON {identity: 'm'|'f'} from manual inspection")
    parser.add_argument("--merge-overrides", type=Path,
                        help="JSON {'not_duplicates': [[a, b], ...]} to reject candidate pairs after inspection")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Official training data: voice-gender classifier and duplicate threshold.
    of = np.loadtxt(args.official_faces, delimiter=",", dtype=np.float32)[:, :-1]
    ov = np.loadtxt(args.official_voices, delimiter=",", dtype=np.float32)[:, :-1]
    oid = np.array([r[4] for r in read_rows(args.official_manifest)])
    ogender = dict(row for row in csv.reader(args.official_gender.open()))
    scaler = StandardScaler().fit(ov)
    clf = LogisticRegression(max_iter=2000, C=0.1).fit(scaler.transform(ov), [ogender[i] == "m" for i in oid])

    rng = np.random.default_rng(0)
    same, different, halves = [], [], {}
    for identity in np.unique(oid):
        rows = rng.permutation(np.flatnonzero(oid == identity))
        halves[identity] = (centroid(of[rows[: len(rows) // 2]]), centroid(of[rows[len(rows) // 2:]]))
        same.append(float(halves[identity][0] @ halves[identity][1]))
    names = list(halves)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            different.append(float(halves[a][0] @ halves[b][1]))
    # Above every different-identity pair and inside the same-identity range.
    threshold = float(max(max(different) + 0.01, np.percentile(same, 1)))
    official_centroids = {i: centroid(of[oid == i]) for i in np.unique(oid)}
    # Face-outlier cut-off (wrong-person frames): the 1st percentile of official
    # rows' similarity to their own identity's face centroid.
    ofn = of / np.linalg.norm(of, axis=1, keepdims=True)
    own = np.array([ofn[k] @ official_centroids[oid[k]] for k in range(len(oid))])
    outlier_cut = float(np.percentile(own, 1))

    gender_overrides = json.loads(args.gender_overrides.read_text()) if args.gender_overrides else {}
    rejected = set()
    if args.merge_overrides:
        rejected = {tuple(sorted(p)) for p in json.loads(args.merge_overrides.read_text())["not_duplicates"]}

    faces, voices, manifest, ident = [], [], [], []
    report = {"sources": {}, "duplicate_threshold": threshold,
              "official_same_identity_half_similarity": [round(min(same), 3), round(float(np.median(same)), 3)],
              "official_different_identity_max_similarity": round(max(different), 3),
              "face_outlier_cut": round(outlier_cut, 3)}
    gender, unresolved = {}, {}
    for name in args.sources:
        base = Path("features/external") / name
        rows = read_rows(Path("features/external/manifests") / f"{name}_train.txt")
        f = np.loadtxt(base / "train_faces.csv", delimiter=",", dtype=np.float32)
        v = np.loadtxt(base / "train_voices.csv", delimiter=",", dtype=np.float32)
        lang = list(csv.DictReader((Path("features/external") / f"{name}_train_language.csv").open()))
        # Gender: dataset metadata when the source provides it (features/external/<name>_gender.csv),
        # otherwise the face/voice vote below, which needs the InsightFace face-age file.
        meta_gender_path = Path("features/external") / f"{name}_gender.csv"
        meta_gender = dict(csv.reader(meta_gender_path.open())) if meta_gender_path.exists() else None
        age = None if meta_gender else list(csv.DictReader((base / "train_face_age.csv").open()))
        if not (len(rows) == len(f) == len(v) == len(lang) == len(age if age is not None else rows)):
            raise ValueError(f"{name}: inputs are not row-aligned")
        keep = np.array([r["drop"] == "0" for r in lang])
        ids = np.array([r[4] for r in rows])
        fn = f[:, :-1] / np.linalg.norm(f[:, :-1], axis=1, keepdims=True)
        sim = np.zeros(len(ids))
        for identity in np.unique(ids):
            m = ids == identity
            sim[m] = fn[m] @ centroid(f[m & keep, :-1])
        outlier = sim < outlier_cut
        language_dropped = int((~keep).sum())
        keep &= ~outlier
        p_male_voice = clf.predict_proba(scaler.transform(v[:, :-1]))[:, 1]
        if meta_gender:
            for identity in np.unique(ids):
                vm = float(p_male_voice[(ids == identity) & keep].mean())
                gender[identity] = gender_overrides.get(identity, meta_gender[identity])
                if (vm > 0.5) != (gender[identity] == "m"):
                    report.setdefault("metadata_gender_voice_disagreements", {})[identity] = round(vm, 2)
            age = []
        face_male = np.array([1.0 if r["gender_pred"] == "m" else 0.0 if r["gender_pred"] == "f" else np.nan for r in age])
        for identity in (np.unique(ids) if not meta_gender else []):
            m = (ids == identity) & keep
            fm, vm = float(np.nanmean(face_male[m])), float(p_male_voice[m].mean())
            if identity in gender_overrides:
                gender[identity] = gender_overrides[identity]
            elif (fm > 0.5) == (vm > 0.5) and abs(fm - 0.5) > 0.2:
                gender[identity] = "m" if fm > 0.5 else "f"
            else:
                unresolved[identity] = {"face_male_fraction": round(fm, 2), "voice_male_probability": round(vm, 2)}
        report["sources"][name] = {"rows": len(rows), "language_dropped": language_dropped,
                                   "face_outliers_dropped": int((outlier & (np.array([r["drop"] == "0" for r in lang]))).sum()),
                                   "identities": int(len(np.unique(ids)))}
        faces.append(f[keep, :-1]), voices.append(v[keep, :-1])
        manifest += [r for r, k in zip(rows, keep) if k]
        ident.append(ids[keep])
    faces, voices, ident = np.concatenate(faces), np.concatenate(voices), np.concatenate(ident)

    # Duplicates.
    ext_centroids = {i: centroid(faces[ident == i]) for i in np.unique(ident)}
    vs_official = sorted(((float(c @ official_centroids[o]), e, o) for e, c in ext_centroids.items()
                          for o in official_centroids), reverse=True)
    dup_official = {e: (o, round(s, 3)) for s, e, o in vs_official if s >= threshold}
    ext = sorted(ext_centroids)
    merge_to = {}
    pairs = []
    for i, a in enumerate(ext):
        for b in ext[i + 1:]:
            s = float(ext_centroids[a] @ ext_centroids[b])
            if s >= threshold and tuple(sorted((a, b))) not in rejected:
                pairs.append((round(s, 3), a, b))
                root = merge_to.get(a, a)
                merge_to[b] = root
    report["closest_to_official"] = [[e, o, round(s, 3)] for s, e, o in vs_official[:10]]
    report["duplicates_of_official_dropped"] = dup_official
    report["external_pairs_merged"] = pairs
    report["gender_unresolved_dropped"] = unresolved

    drop = set(dup_official) | set(unresolved)
    final = np.array([merge_to.get(i, i) for i in ident])
    keep = np.array([i not in drop and f not in drop for i, f in zip(ident, final)])
    names = sorted(set(final[keep]))
    class_of = {n: 10000 + k for k, n in enumerate(names)}
    labels = np.array([class_of[n] for n in final[keep]], dtype=np.float32)[:, None]

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    np.savetxt(out / "train_faces.csv", np.hstack([faces[keep], labels]), delimiter=",", fmt="%.8g")
    np.savetxt(out / "train_voices.csv", np.hstack([voices[keep], labels]), delimiter=",", fmt="%.8g")
    with (out / "train_manifest.txt").open("w") as handle:
        for row, n in zip((r for r, k in zip(manifest, keep) if k), final[keep]):
            handle.write(" ".join([*row[:4], n, str(class_of[n])]) + "\n")
    with (out / "gender.csv").open("w") as handle:
        for n in names:
            handle.write(f"{n},{gender[n]}\n")
    counts = Counter(gender[n] for n in names)
    report["final"] = {"rows": int(keep.sum()), "identities": len(names), "male": counts["m"], "female": counts["f"],
                       "clips_per_identity_median": float(np.median(list(Counter(final[keep]).values())))}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
