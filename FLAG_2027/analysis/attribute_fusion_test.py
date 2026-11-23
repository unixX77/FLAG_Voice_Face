"""Does V2 use face build when matching, and would explicit attribute scores help?

Fold-honest, same-gender pairs of held-out identities only.
1. Among similar-age impostors (age gap < 5 years), V2 AUC by build-gap tercile:
   if V2 used build, impostors with a large build gap would be rejected more easily.
2. Fusion: per attribute, ridge regressors fitted on the fold's training rows predict
   the attribute from raw voice and raw face; the attribute score is -|voice - face|.
   z(v2) + sum_k w_k z(attr_k), with weights chosen on the other four folds.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from age_fusion_test import eer, ridge  # noqa: E402
from probe_embedding_age import load_fold_model  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices  # noqa: E402

GRID = (0.0, 0.1, 0.25, 0.5, 0.75)
COMBOS = {"v2": (), "v2+age": ("age",), "v2+build": ("build",), "v2+age+build": ("age", "build")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("face-csv", "voice-csv", "age-csv", "attribute-csv", "meta-csv", "validation-root", "output-json"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--checkpoints", required=True)
    args = p.parse_args()

    F, V, _ = load_feature_matrices(args.face_csv, args.voice_csv)
    rows = list(csv.DictReader(args.age_csv.open()))
    labels = np.array([r["identity"] for r in rows])
    sample_age = np.array([float(r["age"]) if r["age"] else np.nan for r in rows])
    build_sample = np.array([float(r["build"]) for r in csv.DictReader(args.attribute_csv.open())])
    ids = np.unique(labels)
    ident = {"age": {i: float(np.nanmedian(sample_age[labels == i])) for i in ids},
             "build": {i: float(np.median(build_sample[labels == i])) for i in ids}}
    gender = dict(row for row in csv.reader(args.meta_csv.open()))
    sg = np.array([gender[l] for l in labels])
    z = lambda s: (s - s.mean()) / s.std()

    folds = []
    for fold in range(5):
        d = args.validation_root / f"fold_{fold}"
        train, test = np.load(d / "train_indices.npy"), np.load(d / "validation_indices.npy")
        model = load_fold_model(Path(args.checkpoints.format(fold=fold)), F.shape[1], V.shape[1])
        with torch.no_grad():
            Fe = model.face_branch(torch.from_numpy(F[test])).numpy()
            Ve = model.voice_branch(torch.from_numpy(V[test])).numpy()
        tl, tg = labels[test], sg[test]
        mask = tg[:, None] == tg[None, :]
        entry = {"y": (tl[:, None] == tl[None, :])[mask].astype(int), "v2": z((Ve @ Fe.T)[mask])}
        for k, values in ident.items():
            target = np.array([values[l] for l in labels])
            pv, pf = ridge(V, train, target)[test], ridge(F, train, target)[test]
            entry[k] = z(-np.abs(pv[:, None] - pf[None, :])[mask])
            true = np.array([values[l] for l in tl])
            entry[f"{k}_gap"] = np.abs(true[:, None] - true[None, :])[mask]
        folds.append(entry)

    # 1. Does V2 already use build among similar-age impostors?
    all_build_gaps = np.concatenate([f["build_gap"][(f["y"] == 0) & (f["age_gap"] < 5)] for f in folds])
    edges = np.percentile(all_build_gaps, [33.3, 66.7])
    usage = {}
    for name, (lo, hi) in zip(("small", "medium", "large"), ((0, edges[0]), (edges[0], edges[1]), (edges[1], np.inf))):
        aucs = []
        for f in folds:
            m = (f["y"] == 1) | ((f["age_gap"] < 5) & (f["build_gap"] >= lo) & (f["build_gap"] < hi))
            aucs.append({s: roc_auc_score(f["y"][m], f[s][m]) for s in ("v2", "build", "age")})
        usage[f"build_gap_{name}"] = {s: round(float(np.mean([a[s] for a in aucs])), 3) for s in ("v2", "build", "age")}

    # 2. Fusion with weights chosen on the other folds.
    def fused(f, attrs, w):
        return f["v2"] + sum(wi * f[a] for wi, a in zip(w, attrs))

    fusion = {}
    for combo, attrs in COMBOS.items():
        per_fold = []
        for k, f in enumerate(folds):
            others = [g for j, g in enumerate(folds) if j != k]
            w = max(itertools.product(GRID, repeat=len(attrs)),
                    key=lambda w: np.mean([roc_auc_score(g["y"], fused(g, attrs, w)) for g in others]))
            s = fused(f, attrs, w)
            near = (f["y"] == 1) | (f["age_gap"] < 5)
            per_fold.append({"eer": eer(f["y"], s), "similar_age_auc": roc_auc_score(f["y"][near], s[near]), "w": w})
        fusion[combo] = {"same_gender_eer": round(float(np.mean([r["eer"] for r in per_fold])), 4),
                         "per_fold_eer": [round(r["eer"], 4) for r in per_fold],
                         "similar_age_auc": round(float(np.mean([r["similar_age_auc"] for r in per_fold])), 3),
                         "weights": [r["w"] for r in per_fold]}
    base = fusion["v2"]["per_fold_eer"]
    for combo in fusion:
        fusion[combo]["folds_better_than_v2"] = sum(a < b for a, b in zip(fusion[combo]["per_fold_eer"], base))
    report = {"similar_age_impostors_by_build_gap_auc": usage, "fusion": fusion}
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
