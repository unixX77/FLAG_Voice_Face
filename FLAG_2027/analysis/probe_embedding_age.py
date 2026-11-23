"""Does a trained cross-modal model keep and use age? Fold-honest analysis.

For each validation fold, only identities held out of that fold's training are
scored.
A. Age decodability: ridge regressors fitted on the fold's training rows map raw
   features or learned embeddings to identity age (median InsightFace face age),
   then predict held-out identities. Reported as within-gender Pearson r.
B. Age use in matching: same-gender verification AUC on held-out rows, with
   impostor pairs split by the age gap between the two identities.
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
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cross_modal_model import CrossModalIdentityNetwork  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices  # noqa: E402

GAP_BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 200))


def load_fold_model(path: Path, face_dim: int, voice_dim: int) -> CrossModalIdentityNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "model_kwargs" in checkpoint:
        model = CrossModalIdentityNetwork(**checkpoint["model_kwargs"])
    else:
        c = checkpoint["config"]
        num_classes = checkpoint["state_dict"]["identity_classifier.weight"].shape[0]
        model = CrossModalIdentityNetwork(
            face_dim=face_dim, voice_dim=voice_dim, embedding_dim=c["embedding_dim"], num_classes=num_classes,
            arc_scale=c["arc_scale"], arc_margin=c["arc_margin"], prototype_momentum=c["prototype_momentum"],
            face_hidden_dims=tuple(c["face_hidden_dims"]), voice_hidden_dims=tuple(c["voice_hidden_dims"]),
            hidden_dropout=c["hidden_dropout"], standardize_inputs=c["standardize_inputs"],
            input_dropout=c["input_dropout"], input_noise_std=c["input_noise_std"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.eval()


def ridge_identity_predictions(X, train, test, target, labels):
    scaler = StandardScaler().fit(X[train])
    model = RidgeCV(alphas=np.logspace(-2, 5, 15)).fit(scaler.transform(X[train]), target[train])
    pred = model.predict(scaler.transform(X[test]))
    test_labels = labels[test]
    return {i: float(pred[test_labels == i].mean()) for i in np.unique(test_labels)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-csv", type=Path, required=True)
    parser.add_argument("--voice-csv", type=Path, required=True)
    parser.add_argument("--age-csv", type=Path, required=True)
    parser.add_argument("--meta-csv", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=str, required=True, help="pattern with {fold}")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    F, V, numeric = load_feature_matrices(args.face_csv, args.voice_csv)
    labels = np.array([r["identity"] for r in csv.DictReader(args.age_csv.open())])
    assert len(set(zip(numeric.tolist(), labels))) == len(set(labels))
    sample_age = np.array([float(r["age"]) if r["age"] else np.nan for r in csv.DictReader(args.age_csv.open())])
    age = {i: float(np.nanmedian(sample_age[labels == i])) for i in np.unique(labels)}
    gender = dict(row for row in csv.reader(args.meta_csv.open()))
    target = np.array([age[l] for l in labels])
    sample_gender = np.array([gender[l] for l in labels])

    preds = {name: {} for name in ("raw_voice", "v2_voice", "raw_face", "v2_face")}
    auc_rows = {b: [] for b in GAP_BUCKETS}
    auc_all = []
    for fold in range(args.folds):
        fold_dir = args.validation_root / f"fold_{fold}"
        train = np.load(fold_dir / "train_indices.npy"); test = np.load(fold_dir / "validation_indices.npy")
        model = load_fold_model(Path(args.checkpoints.format(fold=fold)), F.shape[1], V.shape[1])
        with torch.no_grad():
            Fe = model.face_branch(torch.from_numpy(F)).numpy()
            Ve = model.voice_branch(torch.from_numpy(V)).numpy()
        for name, X in (("raw_voice", V), ("v2_voice", Ve), ("raw_face", F), ("v2_face", Fe)):
            preds[name].update(ridge_identity_predictions(X, train, test, target, labels))

        # B: every held-out voice against every held-out face of the same gender.
        scores = Ve[test] @ Fe[test].T
        tl = labels[test]; tg = sample_gender[test]; ta = np.array([age[l] for l in tl])
        same_id = tl[:, None] == tl[None, :]
        same_gender = tg[:, None] == tg[None, :]
        gap = np.abs(ta[:, None] - ta[None, :])
        genuine = scores[same_id]
        impostor_mask = same_gender & ~same_id
        auc_all.append(roc_auc_score(np.r_[np.ones(len(genuine)), np.zeros(impostor_mask.sum())],
                                     np.r_[genuine, scores[impostor_mask]]))
        for lo, hi in GAP_BUCKETS:
            mask = impostor_mask & (gap >= lo) & (gap < hi)
            if mask.sum() >= 100:
                auc_rows[(lo, hi)].append((roc_auc_score(np.r_[np.ones(len(genuine)), np.zeros(mask.sum())],
                                                         np.r_[genuine, scores[mask]]), int(mask.sum())))

    decodability = {}
    for name, p in preds.items():
        decodability[name] = {}
        for g in ("m", "f"):
            ids = [i for i in p if gender[i] == g]
            decodability[name][g] = round(float(np.corrcoef([p[i] for i in ids], [age[i] for i in ids])[0, 1]), 3)
    matching = {"same_gender_auc_all": round(float(np.mean(auc_all)), 3)}
    for (lo, hi), rows in auc_rows.items():
        if rows:
            weights = np.array([n for _, n in rows])
            matching[f"impostor_age_gap_{lo}-{hi}"] = {
                "auc": round(float(np.average([a for a, _ in rows], weights=weights)), 3),
                "impostor_pairs": int(weights.sum()), "folds": len(rows)}
    report = {"age_decodability_within_gender_r": decodability, "same_gender_matching": matching}
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
