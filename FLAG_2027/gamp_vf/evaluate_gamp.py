"""Verification diagnostics beyond EER for one or more five-fold configurations.

For each configuration (checkpoint pattern with {fold} and {seed}) on the fixed
official fold protocols (deterministic trial files on disk):
  * EER, ROC AUC per track (standard / same-gender), averaged over folds x seeds
  * FAR and FRR at the training-side-free operating point of each fold's own EER
    threshold, plus FRR at FAR = 5 % and FAR at FRR = 5 %
  * same-gender genuine / impostor score mean, std and d'
  * hard-negative violation rate: share of voice clips whose highest-scoring
    same-gender impostor face outranks its lowest-scoring genuine face
  * pooled score histograms (raw cosine) for plots
Statistics with uncertainty come from compare_fold_runs.py (paired bootstrap);
this tool only describes behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.compare_fold_runs import load_model, voice_inputs, weighted_eer  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices  # noqa: E402
from local_validation.model_evaluation import load_protocol  # noqa: E402

BINS = np.linspace(-1, 1, 81)


def operating_points(labels, scores):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    j = np.searchsorted(fpr, 0.05, side="right") - 1
    k = np.flatnonzero(fnr <= 0.05)
    return {"far_at_eer": float(fpr[i]), "frr_at_eer": float(fnr[i]),
            "frr_at_far5": float(fnr[max(j, 0)]), "far_at_frr5": float(fpr[k[0]]) if len(k) else 1.0}


def violation_rate(labels, scores, voice_idx):
    rates = []
    for v in np.unique(voice_idx):
        m = voice_idx == v
        g, i = scores[m & (labels == 1)], scores[m & (labels == 0)]
        if len(g) and len(i):
            rates.append(float(i.max() > g.min()))
    return float(np.mean(rates))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="name=pattern pairs")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--face-csv", type=Path, default=Path("features/face_v2_farl/train_English_faces.csv"))
    p.add_argument("--voice-csv", type=Path, default=Path("features/voice_v8_wavlm_egemaps/train_English_voices.csv"))
    p.add_argument("--validation-root", type=Path, default=Path("validation"))
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    F, V, _ = load_feature_matrices(args.face_csv, args.voice_csv)
    Ft, Vt = torch.from_numpy(F), torch.from_numpy(V)
    protocols = {(k, t): load_protocol(args.validation_root / f"fold_{k}" / f"{t}_pairs.csv",
                                       args.validation_root / f"fold_{k}" / f"{t}_ground_truth.csv")
                 for k in range(5) for t in ("standard", "gender")}
    report = {"seeds": args.seeds, "score_type": "raw_cosine", "configs": {}}
    for spec in args.runs:
        name, pattern = spec.split("=", 1)
        per = {t: {"eer": [], "auc": [], "ops": [], "violation": [], "gen": [], "imp": []} for t in ("standard", "gender")}
        for fold in range(5):
            for seed in args.seeds:
                path = Path(pattern.format(fold=fold, seed=seed))
                voice = voice_inputs(torch.load(path, map_location="cpu", weights_only=False)["config"], Vt, args.voice_csv)
                model = load_model(path, F.shape[1], voice)
                with torch.no_grad():
                    fe, ve = model.face_branch(Ft), model.voice_branch(voice)
                for t in ("standard", "gender"):
                    pr = protocols[(fold, t)]
                    fi, vi = np.asarray(pr.face_indices), np.asarray(pr.voice_indices)
                    labels = np.asarray(pr.labels)
                    with torch.no_grad():
                        s = model.pair_similarity(fe[fi], ve[vi]).numpy()
                    d = per[t]
                    d["eer"].append(weighted_eer(labels, s, np.ones(len(s))))
                    d["auc"].append(roc_auc_score(labels, s))
                    d["ops"].append(operating_points(labels, s))
                    d["violation"].append(violation_rate(labels, s, vi))
                    d["gen"].append(s[labels == 1])
                    d["imp"].append(s[labels == 0])
        out = {}
        for t, d in per.items():
            gen, imp = np.concatenate(d["gen"]), np.concatenate(d["imp"])
            out[t] = {"eer": float(np.mean(d["eer"])), "auc": float(np.mean(d["auc"])),
                      **{k: float(np.mean([o[k] for o in d["ops"]])) for k in d["ops"][0]},
                      "violation_rate": float(np.mean(d["violation"])),
                      "genuine_mean": float(gen.mean()), "genuine_std": float(gen.std()),
                      "impostor_mean": float(imp.mean()), "impostor_std": float(imp.std()),
                      "d_prime": float((gen.mean() - imp.mean()) / np.sqrt(0.5 * (gen.var() + imp.var()))),
                      "hist_bins": BINS.tolist(),
                      "hist_genuine": np.histogram(gen, BINS)[0].tolist(),
                      "hist_impostor": np.histogram(imp, BINS)[0].tolist(),
                      "per_fold_seed_eer": [round(x, 4) for x in d["eer"]]}
        report["configs"][name] = out
        print(name, {t: {k: round(v, 4) for k, v in o.items() if isinstance(v, float)} for t, o in out.items()},
              flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
