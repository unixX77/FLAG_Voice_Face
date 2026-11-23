"""Approach B: fixed age/gender-consistency terms on top of a trained model's score.

Per trial, using only that trial's own face image and voice clip:

    score = similarity(face, voice)
            - w_age    * |voice_age - face_age| / 10
            - w_gender * |p_male(voice) - male(face)|

Voice age and p_male come from the frozen audeering age-gender model; face age
and gender come from frozen InsightFace (``estimate_face_age.py``). The weights
are chosen per track on the other four folds (leave-one-fold-out), so no fold is
scored with weights tuned on itself. The paired identity-level bootstrap is the
one used by ``compare_fold_runs.py``.

Step 1 (``--check-only``) reports how well the voice-age estimates agree with
the face ages within each gender, against the earlier 70-identity probe (r ~ 0.6).
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.compare_fold_runs import load_model, weighted_eer  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices, load_manifest_identities  # noqa: E402
from local_validation.model_evaluation import load_protocol  # noqa: E402

AGE_GRID = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12)
GENDER_GRID = (0.0, 0.05, 0.1, 0.2, 0.3)
TRACKS = ("standard", "gender")


def read_attributes(voice_csv: Path, face_csv: Path):
    voice = list(csv.DictReader(voice_csv.open()))
    face = list(csv.DictReader(face_csv.open()))
    voice_age = np.array([float(r["age_years"]) for r in voice])
    voice_male = np.array([float(r["p_male"]) / max(float(r["p_male"]) + float(r["p_female"]), 1e-6) for r in voice])
    face_age = np.array([float(r["age"]) if r["age"] else np.nan for r in face])
    face_male = np.array([1.0 if r["gender_pred"] == "m" else 0.0 if r["gender_pred"] == "f" else np.nan for r in face])
    return voice_age, voice_male, face_age, face_male


def check(identities, gender, voice_age, voice_male, face_age):
    ids = np.unique(identities)
    report = {"voice_gender_accuracy_vs_metadata": float(np.mean(
        [(voice_male[identities == i].mean() > 0.5) == (gender[i] == "m") for i in ids]))}
    for g in ("m", "f"):
        members = [i for i in ids if gender[i] == g]
        v = np.array([np.median(voice_age[identities == i]) for i in members])
        f = np.array([np.nanmedian(face_age[identities == i]) for i in members])
        rows = np.isin(identities, members) & np.isfinite(face_age)
        report[g] = {
            "identities": len(members),
            "identity_level_r": round(float(np.corrcoef(v, f)[0, 1]), 3),
            "clip_level_r": round(float(np.corrcoef(voice_age[rows], face_age[rows])[0, 1]), 3),
            "mean_voice_age": round(float(v.mean()), 1), "mean_face_age": round(float(f.mean()), 1),
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("face-csv", "voice-csv", "manifest", "meta-csv", "voice-attributes", "face-attributes"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoints", help="pattern with {fold} and {seed}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--validation-root", type=Path, default=Path("validation"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--correct-voice-age",
        action="store_true",
        help="Map voice age onto the face-age scale with a linear fit per predicted voice gender, "
             "fitted on each fold's training rows only (a fixed model component)",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=1.0,
        help="Multiply the weight grids, for models whose scores have a different spread "
             "(fixed factor measured on training pairs, e.g. 1.36 for the hyperbolic model)",
    )
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    identities = load_manifest_identities(args.manifest)
    gender = dict(row for row in csv.reader(args.meta_csv.open()))
    voice_age, voice_male, face_age, face_male = read_attributes(args.voice_attributes, args.face_attributes)
    report = {"check": check(identities, gender, voice_age, voice_male, face_age)}
    if args.check_only:
        print(json.dumps(report, indent=2))
        return

    F, V, _ = load_feature_matrices(args.face_csv, args.voice_csv)
    Ft, Vt = torch.from_numpy(F), torch.from_numpy(V)
    # trials[fold][track]: labels, voice identity, base scores per seed, age gap, gender gap
    trials = []
    corrections = []
    for fold in range(5):
        d = args.validation_root / f"fold_{fold}"
        fold_voice_age = voice_age
        if args.correct_voice_age:
            # Linear map voice age -> face age, per predicted voice gender, from this
            # fold's training rows only. Selection uses the voice model's own gender
            # prediction, so it needs no labels at test time.
            train = np.load(d / "train_indices.npy")
            fold_voice_age = voice_age.copy()
            fits = {}
            for name, rows in (("male", voice_male >= 0.5), ("female", voice_male < 0.5)):
                fit_rows = np.intersect1d(np.nonzero(rows & np.isfinite(face_age))[0], train)
                slope, intercept = np.polyfit(voice_age[fit_rows], face_age[fit_rows], 1)
                fold_voice_age[rows] = intercept + slope * voice_age[rows]
                fits[name] = [round(float(slope), 3), round(float(intercept), 2)]
            corrections.append(fits)
        per_track = {}
        for track in TRACKS:
            p = load_protocol(d / f"{track}_pairs.csv", d / f"{track}_ground_truth.csv")
            fi, vi = np.asarray(p.face_indices, int), np.asarray(p.voice_indices, int)
            age_gap = np.abs(fold_voice_age[vi] - face_age[fi]) / 10.0
            gender_gap = np.abs(voice_male[vi] - face_male[fi])
            # A face without an InsightFace detection contributes no penalty.
            per_track[track] = {"y": np.asarray(p.labels), "vid": identities[vi], "fi": fi, "vi": vi,
                                "age_gap": np.nan_to_num(age_gap, nan=0.0),
                                "gender_gap": np.nan_to_num(gender_gap, nan=0.0), "base": []}
        for seed in args.seeds:
            model = load_model(Path(args.checkpoints.format(fold=fold, seed=seed)), F.shape[1], Vt)
            with torch.no_grad():
                fe, ve = model.face_branch(Ft), model.voice_branch(Vt)
                for t in per_track.values():
                    t["base"].append(model.pair_similarity(fe[t["fi"]], ve[t["vi"]]).numpy())
        trials.append(per_track)

    def fused(t, w_age, w_gender, seed_index):
        return t["base"][seed_index] - w_age * t["age_gap"] - w_gender * t["gender_gap"]

    def fold_eer(k, track, w, weights=None):
        t = trials[k][track]
        sw = np.ones(len(t["y"])) if weights is None else weights
        return float(np.mean([weighted_eer(t["y"], fused(t, *w, s), sw) for s in range(len(t["base"]))]))

    grid = [(a * args.weight_scale, g * args.weight_scale)
            for a, g in itertools.product(AGE_GRID, GENDER_GRID)]
    chosen = {track: [] for track in TRACKS}
    for track in TRACKS:
        for k in range(5):
            others = [j for j in range(5) if j != k]
            chosen[track].append(min(grid, key=lambda w: np.mean([fold_eer(j, track, w) for j in others])))

    def summary(weights_for):
        out = {}
        for track in TRACKS:
            out[("a", track)] = np.mean([fold_eer(k, track, (0.0, 0.0), weights_for(k, track)) for k in range(5)])
            out[("b", track)] = np.mean([fold_eer(k, track, chosen[track][k], weights_for(k, track)) for k in range(5)])
        return out

    point = summary(lambda k, track: None)
    rng = np.random.default_rng(0)
    diffs = {t: [] for t in (*TRACKS, "mean")}
    for _ in range(args.bootstrap):
        counts = []
        for f in trials:
            ids = np.unique(f["standard"]["vid"])
            drawn, n = np.unique(rng.choice(ids, len(ids)), return_counts=True)
            counts.append(dict(zip(drawn, n)))
        boot = summary(lambda k, track: np.array([counts[k].get(i, 0) for i in trials[k][track]["vid"]], float))
        for t in TRACKS:
            diffs[t].append(boot[("b", t)] - boot[("a", t)])
        diffs["mean"].append(np.mean([boot[("b", t)] - boot[("a", t)] for t in TRACKS]))
    for t in (*TRACKS, "mean"):
        a = np.mean([point[("a", x)] for x in TRACKS]) if t == "mean" else point[("a", t)]
        b = np.mean([point[("b", x)] for x in TRACKS]) if t == "mean" else point[("b", t)]
        lo, hi = np.percentile(diffs[t], [2.5, 97.5])
        report[t] = {"base": round(float(a), 4), "with_terms": round(float(b), 4), "diff": round(float(b - a), 4),
                     "ci95": [round(float(lo), 4), round(float(hi), 4)],
                     "p_better": round(float(np.mean(np.array(diffs[t]) < 0)), 3)}
    if corrections:
        report["voice_age_correction_per_fold_[slope,intercept]"] = corrections
    report["weights_per_fold"] = {t: [list(w) for w in chosen[t]] for t in TRACKS}
    report["per_fold_eer"] = {t: {"base": [round(fold_eer(k, t, (0.0, 0.0)), 4) for k in range(5)],
                                  "with_terms": [round(fold_eer(k, t, chosen[t][k]), 4) for k in range(5)]}
                              for t in TRACKS}
    text = json.dumps(report, indent=2)
    if args.output_json:
        args.output_json.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
