"""Would an explicit age signal add anything on top of V2? Fold-honest fusion test.

Per fold, ridge regressors fitted on training rows predict age from raw voice and
from raw face. For held-out same-gender pairs, an age score is -|age(voice) -
age(face)|. It is fused with the V2 cosine as z(v2) + w * z(age); w is chosen per
fold on the other four folds, so the reported gain is not tuned on the fold it is
measured on.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_embedding_age import GAP_BUCKETS, load_fold_model  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices  # noqa: E402

WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s)
    i = np.nanargmin(np.abs(fpr - (1 - tpr)))
    return float((fpr[i] + 1 - tpr[i]) / 2)


def ridge(X, train, target):
    scaler = StandardScaler().fit(X[train])
    model = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(scaler.transform(X[train]), target[train])
    return model.predict(scaler.transform(X))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("face-csv", "voice-csv", "age-csv", "meta-csv", "validation-root", "output-json"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--checkpoints", required=True)
    args = p.parse_args()

    F, V, _ = load_feature_matrices(args.face_csv, args.voice_csv)
    rows = list(csv.DictReader(args.age_csv.open()))
    labels = np.array([r["identity"] for r in rows])
    sample_age = np.array([float(r["age"]) if r["age"] else np.nan for r in rows])
    age = {i: float(np.nanmedian(sample_age[labels == i])) for i in np.unique(labels)}
    gender = dict(row for row in csv.reader(args.meta_csv.open()))
    target = np.array([age[l] for l in labels]); sg = np.array([gender[l] for l in labels])

    per_fold = []
    for fold in range(5):
        d = args.validation_root / f"fold_{fold}"
        train, test = np.load(d / "train_indices.npy"), np.load(d / "validation_indices.npy")
        model = load_fold_model(Path(args.checkpoints.format(fold=fold)), F.shape[1], V.shape[1])
        with torch.no_grad():
            Fe = model.face_branch(torch.from_numpy(F[test])).numpy()
            Ve = model.voice_branch(torch.from_numpy(V[test])).numpy()
        age_v, age_f = ridge(V, train, target)[test], ridge(F, train, target)[test]
        tl, tg = labels[test], sg[test]
        ta = np.array([age[l] for l in tl])
        mask = tg[:, None] == tg[None, :]
        y = (tl[:, None] == tl[None, :])[mask].astype(int)
        v2 = (Ve @ Fe.T)[mask]
        a = -np.abs(age_v[:, None] - age_f[None, :])[mask]
        gap = np.abs(ta[:, None] - ta[None, :])[mask]
        z = lambda s: (s - s.mean()) / s.std()
        per_fold.append({"y": y, "v2": z(v2), "age": z(a), "gap": gap})

    results = {"weights_chosen": [], "folds": []}
    for k, f in enumerate(per_fold):
        others = [g for j, g in enumerate(per_fold) if j != k]
        w = max(WEIGHTS, key=lambda w: np.mean([roc_auc_score(g["y"], g["v2"] + w * g["age"]) for g in others]))
        results["weights_chosen"].append(w)
        fused = f["v2"] + w * f["age"]
        row = {"v2_eer": eer(f["y"], f["v2"]), "fused_eer": eer(f["y"], fused), "age_only_eer": eer(f["y"], f["age"])}
        for lo, hi in GAP_BUCKETS:
            m = (f["y"] == 1) | ((f["gap"] >= lo) & (f["gap"] < hi))
            row[f"gap_{lo}-{hi}"] = {"v2_auc": roc_auc_score(f["y"][m], f["v2"][m]),
                                     "age_only_auc": roc_auc_score(f["y"][m], f["age"][m]),
                                     "fused_auc": roc_auc_score(f["y"][m], fused[m])}
        results["folds"].append(row)
    summary = {k: round(float(np.mean([r[k] for r in results["folds"]])), 4) for k in ("v2_eer", "fused_eer", "age_only_eer")}
    summary["fused_minus_v2_per_fold"] = [round(r["fused_eer"] - r["v2_eer"], 4) for r in results["folds"]]
    for lo, hi in GAP_BUCKETS:
        summary[f"gap_{lo}-{hi}_auc"] = {s: round(float(np.mean([r[f"gap_{lo}-{hi}"][s] for r in results["folds"]])), 3)
                                         for s in ("v2_auc", "age_only_auc", "fused_auc")}
    summary["weights_chosen"] = results["weights_chosen"]
    args.output_json.write_text(json.dumps({"summary": summary, **results}, indent=2, default=float))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
