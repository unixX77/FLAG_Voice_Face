"""Does the voice carry face age within a gender? Identity-disjoint linear probe.

Each identity's age label is the median InsightFace age over its faces. A ridge
regressor maps voice features to that label; predictions are always made for
identities held out of training (gender-stratified identity folds) and averaged
per identity. Signal is measured within each gender and compared with a null
distribution where ages are shuffled among same-gender identities.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.fold_dataset import load_feature_matrices  # noqa: E402


def identity_folds(ids, gender, n_folds, rng):
    folds = {}
    for g in ("m", "f"):
        members = rng.permutation([i for i in ids if gender[i] == g])
        for k, identity in enumerate(members):
            folds[identity] = k % n_folds
    return folds


def cross_val_predict(X, labels, target, ids, gender, n_folds, rng):
    folds = identity_folds(ids, gender, n_folds, rng)
    sample_fold = np.array([folds[l] for l in labels])
    pred = np.empty(len(labels))
    for k in range(n_folds):
        test = sample_fold == k
        scaler = StandardScaler().fit(X[~test])
        model = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(scaler.transform(X[~test]), target[~test])
        pred[test] = model.predict(scaler.transform(X[test]))
    return {i: pred[labels == i].mean() for i in ids}


def within_gender_stats(pred, age, ids, gender, min_gap):
    out = {}
    for g in ("m", "f"):
        members = [i for i in ids if gender[i] == g]
        p = np.array([pred[i] for i in members]); a = np.array([age[i] for i in members])
        pairs = [(x, y) for x, y in combinations(range(len(members)), 2) if abs(a[x] - a[y]) >= min_gap]
        concordant = np.mean([np.sign(p[x] - p[y]) == np.sign(a[x] - a[y]) for x, y in pairs])
        out[g] = {"n_ids": len(members), "pearson_r": float(np.corrcoef(p, a)[0, 1]),
                  "mae_years": float(np.abs(p - a).mean()), "mae_predict_mean": float(np.abs(a - a.mean()).mean()),
                  "pair_concordance": float(concordant), "n_pairs": len(pairs)}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-csv", type=Path, required=True)
    parser.add_argument("--voice-csv", type=Path, required=True)
    parser.add_argument("--age-csv", type=Path, required=True)
    parser.add_argument("--meta-csv", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--min-gap", type=float, default=10.0)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    F, V, labels = load_feature_matrices(str(args.face_csv), str(args.voice_csv))
    gender = dict(row for row in csv.reader(args.meta_csv.open()))
    age_rows = list(csv.DictReader(args.age_csv.open()))
    names = np.array([r["identity"] for r in age_rows])
    # Feature CSVs carry numeric class labels; they must map one-to-one onto identity names.
    assert len(names) == len(labels) and len(set(zip(labels.tolist(), names))) == len(set(names)) == len(set(labels.tolist()))
    labels = names
    ids = sorted(set(labels))

    sample_age = np.array([float(r["age"]) if r["age"] else np.nan for r in age_rows])
    age = {i: float(np.nanmedian(sample_age[labels == i])) for i in ids}
    within_id_sd = np.mean([np.nanstd(sample_age[labels == i]) for i in ids])
    pred_gender = {i: max("mf", key=lambda g: sum(r["gender_pred"] == g for r, l in zip(age_rows, labels) if l == i)) for i in ids}
    report = {
        "age_labels": {
            "gender_agreement_with_meta": float(np.mean([pred_gender[i] == gender[i] for i in ids])),
            "mean_within_identity_age_sd": float(within_id_sd),
            **{f"{g}_identity_ages": sorted(round(age[i]) for i in ids if gender[i] == g) for g in ("m", "f")},
        }
    }
    # Each sample's target is its identity's median age; fill one missing face with it as well.
    target = np.array([age[l] for l in labels])

    rng = np.random.default_rng(0)
    results = {}
    for name, X in (("voice", V), ("face_sanity", F)):
        preds = [cross_val_predict(X, labels, target, ids, gender, args.folds, rng) for _ in range(args.repeats)]
        mean_pred = {i: np.mean([p[i] for p in preds]) for i in ids}
        results[name] = within_gender_stats(mean_pred, age, ids, gender, args.min_gap)
    report["probe"] = results

    null = {g: {"pearson_r": [], "pair_concordance": []} for g in ("m", "f")}
    for _ in range(args.permutations):
        shuffled = dict(age)
        for g in ("m", "f"):
            members = [i for i in ids if gender[i] == g]
            for i, v in zip(members, rng.permutation([age[i] for i in members])):
                shuffled[i] = v
        pred = cross_val_predict(V, labels, np.array([shuffled[l] for l in labels]), ids, gender, args.folds, rng)
        stats = within_gender_stats(pred, shuffled, ids, gender, args.min_gap)
        for g in ("m", "f"):
            for k in null[g]:
                null[g][k].append(stats[g][k])
    report["voice_null"] = {
        g: {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
                "p_value": float((1 + sum(x >= results["voice"][g][k] for x in v)) / (1 + len(v)))}
            for k, v in null[g].items()} for g in ("m", "f")}
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
