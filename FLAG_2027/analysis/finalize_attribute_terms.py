"""Fix the age/gender-consistency scoring terms before any evaluation data is used.

Writes a JSON scoring configuration from training data only:

* the voice-age correction (linear map voice age -> face age per predicted voice
  gender), fitted on all training rows;
* one (w_age, w_gender) pair per track: the grid point with the lowest mean EER
  over all five folds and the given seeds, using the fold models.

The submission script applies this file unchanged to every evaluation trial.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from age_consistency_test import AGE_GRID, GENDER_GRID, TRACKS, read_attributes  # noqa: E402
from local_validation.compare_fold_runs import load_model, voice_inputs, weighted_eer  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices  # noqa: E402
from local_validation.model_evaluation import load_protocol  # noqa: E402


def fit_correction(voice_age, voice_male, face_age, rows):
    fits = {}
    for name, mask in (("male", voice_male >= 0.5), ("female", voice_male < 0.5)):
        fit_rows = np.intersect1d(np.nonzero(mask & np.isfinite(face_age))[0], rows)
        slope, intercept = np.polyfit(voice_age[fit_rows], face_age[fit_rows], 1)
        fits[name] = {"slope": float(slope), "intercept": float(intercept)}
    return fits


def apply_correction(voice_age, voice_male, fits):
    out = voice_age.copy()
    for name, mask in (("male", voice_male >= 0.5), ("female", voice_male < 0.5)):
        out[mask] = fits[name]["intercept"] + fits[name]["slope"] * voice_age[mask]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("face-csv", "voice-csv", "voice-attributes", "face-attributes", "output-json"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--checkpoints", required=True, help="fold model pattern with {fold} and {seed}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--validation-root", type=Path, default=Path("validation"))
    parser.add_argument("--ensemble", action="store_true",
                        help="fit the terms on the per-trial mean cosine of all --seeds (a fixed seed ensemble)")
    args = parser.parse_args()

    F, V, _ = load_feature_matrices(args.face_csv, args.voice_csv)
    Ft, Vt = torch.from_numpy(F), torch.from_numpy(V)
    voice_age, voice_male, face_age, face_male = read_attributes(args.voice_attributes, args.face_attributes)

    grid = [(a * args.weight_scale, g * args.weight_scale) for a, g in itertools.product(AGE_GRID, GENDER_GRID)]
    eers = {track: {w: [] for w in grid} for track in TRACKS}
    for fold in range(5):
        d = args.validation_root / f"fold_{fold}"
        # Fold-honest correction, as in the cross-validated test.
        corrected = apply_correction(voice_age, voice_male,
                                     fit_correction(voice_age, voice_male, face_age, np.load(d / "train_indices.npy")))
        pooled = {track: [] for track in TRACKS}
        for seed in args.seeds:
            path = Path(args.checkpoints.format(fold=fold, seed=seed))
            voice = voice_inputs(torch.load(path, map_location="cpu", weights_only=False)["config"], Vt, args.voice_csv)
            model = load_model(path, F.shape[1], voice)
            with torch.no_grad():
                fe, ve = model.face_branch(Ft), model.voice_branch(voice)
            for track in TRACKS:
                p = load_protocol(d / f"{track}_pairs.csv", d / f"{track}_ground_truth.csv")
                fi, vi = np.asarray(p.face_indices, int), np.asarray(p.voice_indices, int)
                with torch.no_grad():
                    base = model.pair_similarity(fe[fi], ve[vi]).numpy()
                age_gap = np.nan_to_num(np.abs(corrected[vi] - face_age[fi]) / 10.0)
                gender_gap = np.nan_to_num(np.abs(voice_male[vi] - face_male[fi]))
                y = np.asarray(p.labels)
                if args.ensemble:
                    pooled[track].append((base, age_gap, gender_gap, y))
                    continue
                for w in grid:
                    eers[track][w].append(weighted_eer(y, base - w[0] * age_gap - w[1] * gender_gap, np.ones(len(y))))
        if args.ensemble:
            for track, parts in pooled.items():
                base = np.mean([q[0] for q in parts], axis=0)  # same trials for every seed
                _, age_gap, gender_gap, y = parts[0]
                for w in grid:
                    eers[track][w].append(weighted_eer(y, base - w[0] * age_gap - w[1] * gender_gap, np.ones(len(y))))

    weights = {}
    for track in TRACKS:
        best = min(grid, key=lambda w: np.mean(eers[track][w]))
        weights[track] = {"w_age_per_10_years": best[0], "w_gender": best[1],
                          "cv_mean_eer": float(np.mean(eers[track][best])),
                          "cv_mean_eer_without_terms": float(np.mean(eers[track][(0.0, 0.0)]))}
    all_rows = np.arange(len(voice_age))
    config = {
        "description": "Fixed age/gender-consistency terms; distance = base_distance + w_age*|corr_voice_age - face_age|/10 + w_gender*|p_male_voice - male_face|",
        "voice_age_correction": fit_correction(voice_age, voice_male, face_age, all_rows),
        "weights": {"no_gender": weights["standard"], "gender": weights["gender"]},
        "fitted_on": "official training set only (all 70 identities for the correction; 5 folds x seeds for the weights)",
        "fold_models": args.checkpoints, "seeds": args.seeds, "weight_scale": args.weight_scale,
        "ensemble": bool(args.ensemble),
    }
    args.output_json.write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
