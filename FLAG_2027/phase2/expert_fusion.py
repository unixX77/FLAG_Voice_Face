"""Phase 4: expert diversity and low-capacity score fusion on the fixed fold protocols.

Experts are five-fold configurations (checkpoint patterns with {fold}/{seed}). For each fold and
seed, every expert scores the same trials (raw cosine). Pairing rule: B0 seed s is fused with
expert seed s; the control "B0 + B0" fuses B0 seed s with B0 seed (s mod 3)+1, so a fusion gain
is only credited to an expert if it beats a same-architecture seed ensemble.
  F1 equal-weight mean of per-fold z-scored scores (z-statistics from the *other* folds' trials)
  F2 logistic regression on [z-scored expert scores], fitted on the other four folds (out-of-fold;
     identity-disjoint), applied to the held-out fold
Diversity vs B0: Pearson / Spearman score correlation, error overlap at each model's own EER
threshold, share of B0 errors corrected by the expert.
Uncertainty: paired identity-level bootstrap (resample held-out voice identities within fold).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.compare_fold_runs import face_inputs, load_model, voice_inputs, weighted_eer  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices, load_manifest_identities  # noqa: E402
from local_validation.model_evaluation import load_protocol  # noqa: E402

TRACKS = ("standard", "gender")


def eer_thr(lab, s):
    fpr, tpr, thr = roc_curve(lab, s)
    i = np.nanargmin(np.abs(1 - tpr - fpr))
    return thr[i]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b0", required=True)
    p.add_argument("--experts", nargs="+", required=True, help="name=pattern")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    F, V, _ = load_feature_matrices(Path("features/face_v2_farl/train_English_faces.csv"),
                                    Path("features/voice_v8_wavlm_egemaps/train_English_voices.csv"))
    Ft, Vt = torch.from_numpy(F), torch.from_numpy(V)
    ident = load_manifest_identities(Path("train_set/train_set/train_English.txt"))
    experts = dict(s.split("=", 1) for s in args.experts)
    if {"b0", "b0x2", "b0x3"} & set(experts):
        raise ValueError("expert names b0/b0x2/b0x3 are reserved for the reference model")
    specs = {"b0": args.b0, **experts}
    # scores[name][(fold, seed, track)] = array
    trials, scores = {}, {n: {} for n in specs}
    for fold in range(5):
        fd = Path(f"validation/fold_{fold}")
        for t in TRACKS:
            pr = load_protocol(fd / f"{t}_pairs.csv", fd / f"{t}_ground_truth.csv")
            trials[(fold, t)] = (np.asarray(pr.face_indices), np.asarray(pr.voice_indices), np.asarray(pr.labels))
        for name, pattern in specs.items():
            for seed in args.seeds:
                path = Path(pattern.format(fold=fold, seed=seed))
                cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
                voice = voice_inputs(cfg, Vt, Path("features/voice_v8_wavlm_egemaps/train_English_voices.csv"))
                face = face_inputs(cfg, Ft, Path("features/face_v2_farl/train_English_faces.csv"))
                model = load_model(path, face.shape[1], voice)
                with torch.no_grad():
                    fe, ve = model.face_branch(face), model.voice_branch(voice)
                    for t in TRACKS:
                        fi, vi, _ = trials[(fold, t)]
                        scores[name][(fold, seed, t)] = model.pair_similarity(fe[fi], ve[vi]).numpy().astype(np.float64)
        print("scored fold", fold, flush=True)

    def zstats(name, fold, seed, t):
        other = np.concatenate([scores[name][(k, seed, t)] for k in range(5) if k != fold])
        return other.mean(), other.std()

    def fused(names_seeds, fold, t, method):
        """names_seeds: list of (name, seed); returns fused scores for this fold/track."""
        z = []
        for n, s in names_seeds:
            mu, sd = zstats(n, fold, s, t)
            z.append((scores[n][(fold, s, t)] - mu) / sd)
        z = np.stack(z, 1)
        if method == "mean":
            return z.mean(1)
        X, y = [], []
        for k in range(5):
            if k == fold:
                continue
            zk = []
            for n, s in names_seeds:
                mu, sd = zstats(n, fold, s, t)  # one transform, computed without the held-out fold
                zk.append((scores[n][(k, s, t)] - mu) / sd)
            X.append(np.stack(zk, 1)), y.append(trials[(k, t)][2])
        clf = LogisticRegression(C=1.0).fit(np.concatenate(X), np.concatenate(y))
        return z @ clf.coef_[0]

    systems = {"b0": lambda fold, seed, t: scores["b0"][(fold, seed, t)]}
    # Every fusion has a same-size B0 seed ensemble as its control (matched number of models).
    pairs = {"b0x2": lambda seed: [("b0", seed), ("b0", seed % 3 + 1)],
             "b0x3": lambda seed: [("b0", 1), ("b0", 2), ("b0", 3)]}
    control_of = {}
    for name in specs:
        if name != "b0":
            systems[name] = (lambda n: lambda fold, seed, t: scores[n][(fold, seed, t)])(name)
            pairs[f"b0+{name}"] = (lambda n: lambda seed: [("b0", seed), (n, seed)])(name)
            pairs[f"b0x2+{name}"] = (lambda n: lambda seed: [("b0", seed), ("b0", seed % 3 + 1), (n, seed)])(name)
            control_of[f"b0+{name}"], control_of[f"b0x2+{name}"] = "b0x2", "b0x3"
    for pname, sel in pairs.items():
        for method in ("mean", "logreg"):
            systems[f"{pname}|{method}"] = (lambda sel, m: lambda fold, seed, t: fused(sel(seed), fold, t, m))(sel, method)
    # Point estimates and bootstrap vs B0.
    out_scores = {s: {(f, sd, t): fn(f, sd, t) for f in range(5) for sd in args.seeds for t in TRACKS}
                  for s, fn in systems.items()}
    voice_id = {(f, t): ident[trials[(f, t)][1]] for f in range(5) for t in TRACKS}

    def summary(sys_name, weights):
        return {t: float(np.mean([weighted_eer(trials[(f, t)][2], out_scores[sys_name][(f, sd, t)], weights[(f, t)])
                                  for f in range(5) for sd in args.seeds])) for t in TRACKS}
    ones = {(f, t): np.ones(len(trials[(f, t)][2])) for f in range(5) for t in TRACKS}
    point = {s: summary(s, ones) for s in systems}
    rng = np.random.default_rng(0)
    boots = {s: [] for s in systems}
    boots_ctrl = {s: [] for s in systems}
    for _ in range(args.bootstrap):
        w = {}
        for f in range(5):
            ids = np.unique(voice_id[(f, "standard")])
            drawn, cnt = np.unique(rng.choice(ids, len(ids)), return_counts=True)
            c = dict(zip(drawn, cnt))
            for t in TRACKS:
                w[(f, t)] = np.array([c.get(i, 0) for i in voice_id[(f, t)]], float)
        b0 = summary("b0", w)
        cache = {}
        for s in systems:
            cache[s] = summary(s, w)
            boots[s].append(np.mean([cache[s][t] - b0[t] for t in TRACKS]))
        for s in systems:
            base, method = (s.split("|") + [""])[:2]
            if base in control_of:
                c = cache[f"{control_of[base]}|{method}"]
                boots_ctrl[s].append(np.mean([cache[s][t] - c[t] for t in TRACKS]))
    report = {"seeds": args.seeds, "systems": {}}
    for s in systems:
        d = np.asarray(boots[s])
        report["systems"][s] = {"standard": point[s]["standard"], "gender": point[s]["gender"],
                                "mean": float(np.mean(list(point[s].values()))),
                                "mean_minus_b0": float(np.mean(list(point[s].values())) - np.mean(list(point["b0"].values()))),
                                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                                "p_better_than_b0": float((d < 0).mean())}
        if boots_ctrl[s]:
            dc = np.asarray(boots_ctrl[s])
            base, method = (s.split("|") + [""])[:2]
            ctrl = f"{control_of[base]}|{method}"
            report["systems"][s] |= {"control": ctrl,
                                     "mean_minus_control": report["systems"][s]["mean"] - float(np.mean(list(point[ctrl].values()))),
                                     "ci95_vs_control": [float(np.percentile(dc, 2.5)), float(np.percentile(dc, 97.5))],
                                     "p_better_than_control": float((dc < 0).mean())}
    # Diversity vs B0 (pooled over folds/seeds, per track).
    div = {}
    for name in specs:
        if name == "b0":
            continue
        dd = {}
        for t in TRACKS:
            a = np.concatenate([scores["b0"][(f, s, t)] for f in range(5) for s in args.seeds])
            b = np.concatenate([scores[name][(f, s, t)] for f in range(5) for s in args.seeds])
            lab = np.concatenate([trials[(f, t)][2] for f in range(5) for s in args.seeds])
            ea = (a >= eer_thr(lab, a)) != lab.astype(bool)
            eb = (b >= eer_thr(lab, b)) != lab.astype(bool)
            dd[t] = {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(spearmanr(a, b).statistic),
                     "b0_errors": int(ea.sum()), "expert_errors": int(eb.sum()), "both_wrong": int((ea & eb).sum()),
                     "share_b0_errors_corrected_by_expert": float((ea & ~eb).sum() / ea.sum())}
        div[name] = dd
    # Seed-vs-seed diversity of B0 itself (control).
    dd = {}
    for t in TRACKS:
        a = np.concatenate([scores["b0"][(f, s, t)] for f in range(5) for s in args.seeds])
        b = np.concatenate([scores["b0"][(f, s % 3 + 1, t)] for f in range(5) for s in args.seeds])
        dd[t] = {"pearson": float(np.corrcoef(a, b)[0, 1])}
    div["b0_other_seed(control)"] = dd
    report["diversity_vs_b0"] = div
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=1))
    for s, r in report["systems"].items():
        extra = (f" | vs {r['control']}: {r['mean_minus_control']*100:+.2f} [{r['ci95_vs_control'][0]*100:+.2f},"
                 f"{r['ci95_vs_control'][1]*100:+.2f}] P={r['p_better_than_control']:.2f}") if "control" in r else ""
        print(f"{s:34s} std {r['standard']*100:.2f} gen {r['gender']*100:.2f} mean {r['mean']*100:.2f} "
              f"d {r['mean_minus_b0']*100:+.2f} [{r['ci95'][0]*100:+.2f},{r['ci95'][1]*100:+.2f}] P={r['p_better_than_b0']:.2f}{extra}")
    for n, d in div.items():
        print(n, {t: {k: round(v, 3) for k, v in x.items()} for t, x in d.items()})


if __name__ == "__main__":
    main()
