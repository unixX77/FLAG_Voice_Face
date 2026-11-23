#!/usr/bin/env python3
"""Local proxy for the heard / unheard-language gap (Bengali itself is never used).

Runs trained with ``--holdout-external-prefix mav1`` never see one fifth of the
MAV-Celeb v1 identities (index % 5 == fold). Those people speak English and
Urdu. Runs trained with ``--exclude-train-languages Urdu`` additionally never
hear any Urdu, so Urdu plays the role of the unheard test language.

For every held-out voice clip, fixed trials are drawn once per fold (identical
for every run): ``--targets`` faces of the same person (other images) and
``--impostors`` faces of other held-out people -- any gender (standard track)
or same gender (gender track). EER per track and voice language, averaged over
folds and seeds, with a paired identity bootstrap when two run sets are given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.compare_fold_runs import load_model, weighted_eer  # noqa: E402
from train_cross_modal import holdout_external_identities  # noqa: E402


def build_trials(rows, gender_of, held_out, rng, targets, impostors):
    idx = {i: np.flatnonzero(np.array([r[4] == i for r in rows])) for i in held_out}
    trials = {"standard": [], "gender": []}
    for identity in sorted(held_out):
        others_any = [o for o in sorted(held_out) if o != identity]
        others_same = [o for o in others_any if gender_of[o] == gender_of[identity]]
        for v in idx[identity]:
            language = rows[v][2].split("/")[1]
            own = idx[identity][idx[identity] != v]
            for f in rng.choice(own, min(targets, len(own)), replace=False):
                for track in trials:
                    trials[track].append((v, f, 1, identity, language))
            for track, pool in (("standard", others_any), ("gender", others_same)):
                for o in rng.choice(pool, impostors, replace=True):
                    trials[track].append((v, rng.choice(idx[o]), 0, identity, language))
    return trials


def embed(path, faces, voices):
    model = load_model(path, faces.shape[1], torch.from_numpy(voices[:1]))
    with torch.inference_mode():
        f, v = model(torch.from_numpy(faces), torch.from_numpy(voices))
    return f.numpy(), v.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="one or two checkpoint patterns with {fold}, {seed}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--prefix", default="mav1")
    parser.add_argument("--targets", type=int, default=3)
    parser.add_argument("--impostors", type=int, default=3)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    first = torch.load(args.runs[0].format(fold=0, seed=args.seeds[0]), map_location="cpu", weights_only=False)["config"]
    manifest = Path(first["extra_manifest"])
    rows = [line.split() for line in manifest.open() if line.strip()]
    gender_of = dict(line.strip().split(",") for line in Path(first["extra_gender_metadata"]).open() if line.strip())
    cache = {}

    def features(path):
        if path not in cache:
            cache[path] = np.loadtxt(path, delimiter=",", dtype=np.float32)[:, :-1]
        return cache[path]

    # scores[run][fold][track] = array over that fold's trials (averaged over seeds)
    scores, trial_sets = [], {}
    for fold in range(5):
        held_out = holdout_external_identities([r[4] for r in rows], args.prefix, fold)
        trial_sets[fold] = build_trials(rows, gender_of, held_out, np.random.default_rng(100 + fold),
                                        args.targets, args.impostors)
    for pattern in args.runs:
        per_fold = {}
        for fold in range(5):
            sums = {t: 0.0 for t in ("standard", "gender")}
            for seed in args.seeds:
                path = pattern.format(fold=fold, seed=seed)
                config = torch.load(path, map_location="cpu", weights_only=False)["config"]
                if sorted(config.get("held_out_external", [])) != sorted(
                        holdout_external_identities([r[4] for r in rows], args.prefix, fold)):
                    raise ValueError(f"{path} was not trained with the {args.prefix} hold-out of fold {fold}")
                fe, ve = embed(path, features(config["extra_face_csv"]), features(config["extra_voice_csv"]))
                for track, trials in trial_sets[fold].items():
                    v, f = np.array([t[0] for t in trials]), np.array([t[1] for t in trials])
                    sums[track] = sums[track] + (fe[f] * ve[v]).sum(1)
            per_fold[fold] = {t: s / len(args.seeds) for t, s in sums.items()}
        scores.append(per_fold)

    # Per fold and track: identity index, language and label of every trial.
    arrays, names = {}, {}
    for fold, sets in trial_sets.items():
        names[fold] = sorted({t[3] for t in sets["standard"]})
        pos = {n: i for i, n in enumerate(names[fold])}
        arrays[fold] = {track: (np.array([pos[t[3]] for t in tr]), np.array([t[4] for t in tr]),
                                np.array([t[2] for t in tr])) for track, tr in sets.items()}

    def summarize(run_scores, select=None):
        """EER per track and language; ``select`` = {fold: bootstrap count per identity} or None."""
        out = {}
        for track in ("standard", "gender"):
            for language in ("English", "Urdu"):
                eers = []
                for fold in range(5):
                    ident, lang, labels = arrays[fold][track]
                    w = np.ones(len(ident)) if select is None else select[fold][ident]
                    m = (lang == language) & (w > 0)
                    eers.append(weighted_eer(labels[m], run_scores[fold][track][m], w[m]))
                out[f"{track}_{language}"] = float(np.mean(eers))
            out[f"{track}_gap"] = out[f"{track}_Urdu"] - out[f"{track}_English"]
        out["mean_English"] = (out["standard_English"] + out["gender_English"]) / 2
        out["mean_Urdu"] = (out["standard_Urdu"] + out["gender_Urdu"]) / 2
        return out

    report = {"runs": args.runs, "seeds": args.seeds, "prefix": args.prefix,
              "trials_per_fold": {f: {t: len(v) for t, v in s.items()} for f, s in trial_sets.items()},
              "results": [summarize(s) for s in scores]}
    if len(scores) == 2:
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(args.bootstrap):
            select = {fold: np.bincount(rng.integers(0, len(names[fold]), len(names[fold])),
                                        minlength=len(names[fold])).astype(float) for fold in range(5)}
            a, b = summarize(scores[0], select), summarize(scores[1], select)
            diffs.append({k: b[k] - a[k] for k in a})
        report["b_minus_a"] = {k: {"mean": report["results"][1][k] - report["results"][0][k],
                                   "ci95": [float(np.percentile([d[k] for d in diffs], q)) for q in (2.5, 97.5)],
                                   "p_b_better": float(np.mean([d[k] < 0 for d in diffs]))}
                               for k in diffs[0]}
    text = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.write_text(text + "\n")
    print(json.dumps({"results": report["results"], "b_minus_a": report.get("b_minus_a")}, indent=1))


if __name__ == "__main__":
    main()
