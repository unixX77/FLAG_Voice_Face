"""E6: seed-ensemble evaluation on a language-holdout proxy from saved per-trial scores.

Input: the --save-scores .npz of phase2/language_transfer_fusion.py (keys
"<system>__f<fold>_s<seed>_<track>" and "meta__f<fold>_<track>_{identity,language,label}").
For each requested system: single seeds, 2-seed (s + next) and 3-seed z-mean ensembles
(z-statistics from the other folds). EER per track x language, mean over folds (and seeds for
single / 2-seed), and a paired identity bootstrap of every ensemble vs that system's single models.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.compare_fold_runs import weighted_eer  # noqa: E402

TRACKS = ("standard", "gender")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores", type=Path, required=True)
    p.add_argument("--systems", nargs="+", default=["b0"])
    p.add_argument("--languages", nargs=2, default=["English", "Urdu"])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    d = np.load(args.scores)
    seeds = args.seeds
    nxt = {s: seeds[(i + 1) % len(seeds)] for i, s in enumerate(seeds)}
    meta = {(f, t): tuple(d[f"meta__f{f}_{t}_{n}"] for n in ("identity", "language", "label"))
            for f in range(5) for t in TRACKS}
    n_people = {f: int(meta[(f, "standard")][0].max()) + 1 for f in range(5)}

    def raw(sys_, f, s, t):
        return d[f"{sys_}__f{f}_s{s}_{t}"].astype(np.float64)

    def z(sys_, f, s, t):
        o = np.concatenate([raw(sys_, k, s, t) for k in range(5) if k != f])
        return (raw(sys_, f, s, t) - o.mean()) / o.std()

    variants = {}
    for sy in args.systems:
        variants[f"{sy}|single"] = [(lambda sy, s: lambda f, t: raw(sy, f, s, t))(sy, s) for s in seeds]
        variants[f"{sy}|x2"] = [(lambda sy, s: lambda f, t: 0.5 * (z(sy, f, s, t) + z(sy, f, nxt[s], t)))(sy, s) for s in seeds]
        variants[f"{sy}|x3"] = [(lambda sy: lambda f, t: np.mean([z(sy, f, s, t) for s in seeds], 0))(sy)]
    scored = {k: [{(f, t): fn(f, t) for f in range(5) for t in TRACKS} for fn in fns] for k, fns in variants.items()}

    def summary(k, sel=None):
        out = {}
        for t in TRACKS:
            for lang in args.languages:
                e = []
                for f in range(5):
                    idv, lg, lab = meta[(f, t)]
                    w = np.ones(len(idv)) if sel is None else sel[f][idv]
                    m = (lg == lang) & (w > 0)
                    e += [weighted_eer(lab[m], sc[(f, t)][m], w[m]) for sc in scored[k]]
                out[f"{t}_{lang}"] = float(np.mean(e))
        return out
    point = {k: summary(k) for k in scored}
    rng = np.random.default_rng(0)
    diffs = {k: [] for k in scored if not k.endswith("|single")}
    for _ in range(args.bootstrap):
        sel = {f: np.bincount(rng.integers(0, n_people[f], n_people[f]), minlength=n_people[f]).astype(float)
               for f in range(5)}
        r = {k: summary(k, sel) for k in scored}
        for k in diffs:
            base = r[k.split("|")[0] + "|single"]
            diffs[k].append({m: r[k][m] - base[m] for m in base})
    stats = {k: {m: {"mean": float(np.mean([x[m] for x in v])),
                     "ci95": [float(np.percentile([x[m] for x in v], q)) for q in (2.5, 97.5)],
                     "p_better": float(np.mean([x[m] < 0 for x in v]))} for m in v[0]} for k, v in diffs.items()}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"point": point, "ensemble_vs_single": stats, "seeds": seeds}, indent=1))
    for k, v in point.items():
        print(f"{k:16s}", {m: round(x * 100, 2) for m, x in v.items()})
    for k, v in stats.items():
        print(k, "vs single:", {m: (round(x["mean"] * 100, 2), [round(c * 100, 2) for c in x["ci95"]]) for m, x in v.items()})


if __name__ == "__main__":
    main()
