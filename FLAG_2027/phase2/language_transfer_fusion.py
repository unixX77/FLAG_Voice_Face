"""Phase 5: unseen-language (Urdu-holdout proxy) evaluation of single models and score fusions.

Same trials as evaluate_language_transfer.py (identical builder and RNG seeds): MAV-Celeb v1
people held out of each P2 fold; P2 models never heard Urdu. Unlike that tool, each seed is
scored separately and fusions have a matched-size control:
  b0           P2 B0 model, seed s
  b0x2         z-mean of B0 seeds s and s%3+1        (control for 2-model fusions)
  b0+<expert>  z-mean of B0 seed s and expert seed s
z-statistics per fold come from the other four folds' trials (no held-out statistics).
EER per track (standard / same-gender) x voice language (English heard, Urdu unheard),
mean over folds and seeds; paired identity bootstrap vs b0 and vs the control.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.compare_fold_runs import weighted_eer  # noqa: E402
from local_validation.evaluate_language_transfer import build_trials, embed  # noqa: E402
from train_cross_modal import holdout_external_identities  # noqa: E402

TRACKS = ("standard", "gender")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--b0", required=True)
    p.add_argument("--experts", nargs="+", default=[], help="name=pattern")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--prefix", default="mav1", help="external source whose people are held out (mav1 / mav3)")
    p.add_argument("--unheard-language", default="Urdu", help="language excluded from training (Urdu / German)")
    p.add_argument("--save-scores", type=Path, help="optional .npz with every per-trial score")
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    LANGS = ("English", args.unheard_language)
    experts = dict(s.split("=", 1) for s in args.experts)
    if {"b0", "b0x2", "b0x3"} & set(experts):
        raise ValueError("expert names b0/b0x2/b0x3 are reserved for the reference model")
    specs = {"b0": args.b0, **experts}
    first = torch.load(args.b0.format(fold=0, seed=args.seeds[0]), map_location="cpu", weights_only=False)["config"]
    rows = [l.split() for l in Path(first["extra_manifest"]).open() if l.strip()]
    gender_of = dict(l.strip().split(",") for l in Path(first["extra_gender_metadata"]).open() if l.strip())
    names = [r[4] for r in rows]
    cache = {}

    def feats(path):
        if path not in cache:
            cache[path] = np.loadtxt(path, delimiter=",", dtype=np.float32)[:, :-1]
        return cache[path]

    trials, raw = {}, {n: {} for n in specs}
    for fold in range(5):
        held = holdout_external_identities(names, args.prefix, fold)
        trials[fold] = build_trials(rows, gender_of, held, np.random.default_rng(100 + fold), 3, 3)
        for n, pattern in specs.items():
            for seed in args.seeds:
                path = pattern.format(fold=fold, seed=seed)
                cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
                if sorted(cfg.get("held_out_external", [])) != sorted(held):
                    raise ValueError(f"{path}: not trained with the {args.prefix} hold-out of fold {fold}")
                if args.unheard_language not in cfg.get("exclude_train_languages", []):
                    raise ValueError(f"{path}: {args.unheard_language} was not excluded from training")
                fe, ve = embed(path, feats(cfg["extra_face_csv"]), feats(cfg["extra_voice_csv"]))
                for t in TRACKS:
                    v = np.array([x[0] for x in trials[fold][t]]); f = np.array([x[1] for x in trials[fold][t]])
                    raw[n][(fold, seed, t)] = (fe[f] * ve[v]).sum(1).astype(np.float64)
        print("fold", fold, "scored", flush=True)
    meta = {}
    for fold in range(5):
        people = sorted({x[3] for x in trials[fold]["standard"]})
        pos = {q: i for i, q in enumerate(people)}
        meta[fold] = {t: (np.array([pos[x[3]] for x in trials[fold][t]]), np.array([x[4] for x in trials[fold][t]]),
                          np.array([x[2] for x in trials[fold][t]])) for t in TRACKS}
        meta[fold]["n_people"] = len(people)

    def z(n, fold, seed, t):
        other = np.concatenate([raw[n][(k, seed, t)] for k in range(5) if k != fold])
        return (raw[n][(fold, seed, t)] - other.mean()) / other.std()

    # The 2-model control pairs each seed with the next listed seed (= s % 3 + 1 for seeds 1 2 3);
    # with a single seed there is no control and no fusion is evaluated.
    nxt = {s: args.seeds[(i + 1) % len(args.seeds)] for i, s in enumerate(args.seeds)}
    multi = len(args.seeds) > 1
    systems = {"b0": lambda f, s, t: raw["b0"][(f, s, t)]}
    if multi:
        systems["b0x2"] = lambda f, s, t: 0.5 * (z("b0", f, s, t) + z("b0", f, nxt[s], t))
    control = {}
    for n in specs:
        if n != "b0":
            systems[n] = (lambda n: lambda f, s, t: raw[n][(f, s, t)])(n)
            if multi:
                systems[f"b0+{n}"] = (lambda n: lambda f, s, t: 0.5 * (z("b0", f, s, t) + z(n, f, s, t)))(n)
                control[f"b0+{n}"] = "b0x2"
    scored = {k: {(f, s, t): fn(f, s, t) for f in range(5) for s in args.seeds for t in TRACKS} for k, fn in systems.items()}

    def summary(k, sel=None):
        out = {}
        for t in TRACKS:
            for lang in LANGS:
                e = []
                for f in range(5):
                    idv, lg, lab = meta[f][t]
                    w = np.ones(len(idv)) if sel is None else sel[f][idv]
                    m = (lg == lang) & (w > 0)
                    e += [weighted_eer(lab[m], scored[k][(f, s, t)][m], w[m]) for s in args.seeds]
                out[f"{t}_{lang}"] = float(np.mean(e))
            out[f"{t}_gap"] = out[f"{t}_{LANGS[1]}"] - out[f"{t}_English"]
        return out
    point = {k: summary(k) for k in systems}
    rng = np.random.default_rng(0)
    d_b0 = {k: [] for k in systems}
    d_ctrl = {k: [] for k in control}
    for _ in range(args.bootstrap):
        sel = {f: np.bincount(rng.integers(0, meta[f]["n_people"], meta[f]["n_people"]),
                              minlength=meta[f]["n_people"]).astype(float) for f in range(5)}
        r = {k: summary(k, sel) for k in systems}
        for k in systems:
            d_b0[k].append({m: r[k][m] - r["b0"][m] for m in r[k]})
        for k, c in control.items():
            d_ctrl[k].append({m: r[k][m] - r[c][m] for m in r[k]})

    def stats(ds):
        return {m: {"mean": float(np.mean([d[m] for d in ds])),
                    "ci95": [float(np.percentile([d[m] for d in ds], q)) for q in (2.5, 97.5)],
                    "p_better": float(np.mean([d[m] < 0 for d in ds]))} for m in ds[0]}
    report = {"seeds": args.seeds, "point": point, "vs_b0": {k: stats(v) for k, v in d_b0.items() if k != "b0"},
              "vs_control": {k: stats(v) for k, v in d_ctrl.items() if v}, "control_of": control}
    report |= {"prefix": args.prefix, "unheard_language": args.unheard_language,
               "trials_per_fold": {f: {t: int(len(meta[f][t][2])) for t in TRACKS} for f in range(5)},
               "people_per_fold": {f: int(meta[f]["n_people"]) for f in range(5)}}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=1))
    if args.save_scores:
        np.savez_compressed(args.save_scores, **{f"{k}__f{f}_s{s}_{t}": v for k, d in scored.items()
                                                 for (f, s, t), v in d.items()},
                            **{f"meta__f{f}_{t}_{n}": meta[f][t][i] for f in range(5) for t in TRACKS
                               for i, n in enumerate(("identity", "language", "label"))})
    for k, v in point.items():
        print(f"{k:20s}", {m: round(x * 100, 2) for m, x in v.items()})
    for k, v in report["vs_control"].items():
        print(f"{k} vs {control[k]}:", {m: (round(x['mean'] * 100, 2), [round(c * 100, 2) for c in x['ci95']], x['p_better'])
                                       for m, x in v.items() if m in ("gender_English", f"gender_{LANGS[1]}",
                                                                       f"standard_{LANGS[1]}", "gender_gap")})


if __name__ == "__main__":
    main()
