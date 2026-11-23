"""Does the voice carry face build within a gender, beyond age? Identity-disjoint probe.

Build label = per-identity median FaRL zero-shot build score. With --residualize-age,
the label is first replaced by its residual after regressing on identity age within
each gender, so only build information independent of age is tested. The probe and
permutation null are those of probe_voice_age.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_voice_age import cross_val_predict, within_gender_stats  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("face-csv", "voice-csv", "age-csv", "attribute-csv", "meta-csv", "output-json"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--attribute", default="build")
    p.add_argument("--residualize-age", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--permutations", type=int, default=100)
    args = p.parse_args()

    F, V, _ = load_feature_matrices(args.face_csv, args.voice_csv)
    age_rows = list(csv.DictReader(args.age_csv.open()))
    labels = np.array([r["identity"] for r in age_rows])
    sample_age = np.array([float(r["age"]) if r["age"] else np.nan for r in age_rows])
    attr = np.array([float(r[args.attribute]) for r in csv.DictReader(args.attribute_csv.open())])
    assert len(attr) == len(labels)
    gender = dict(row for row in csv.reader(args.meta_csv.open()))
    ids = sorted(set(labels))
    value = {i: float(np.median(attr[labels == i])) for i in ids}
    age = {i: float(np.nanmedian(sample_age[labels == i])) for i in ids}
    if args.residualize_age:
        for g in ("m", "f"):
            members = [i for i in ids if gender[i] == g]
            slope, intercept = np.polyfit([age[i] for i in members], [value[i] for i in members], 1)
            for i in members:
                value[i] -= slope * age[i] + intercept
    # Concordance is counted on pairs at least one within-gender SD apart.
    gap = float(np.mean([np.std([value[i] for i in ids if gender[i] == g]) for g in ("m", "f")]))
    target = np.array([value[l] for l in labels])

    rng = np.random.default_rng(0)
    report = {"attribute": args.attribute, "residualize_age": args.residualize_age, "pair_gap": gap, "probe": {}}
    for name, X in (("voice", V), ("face_sanity", F)):
        preds = [cross_val_predict(X, labels, target, ids, gender, 5, rng) for _ in range(args.repeats)]
        report["probe"][name] = within_gender_stats({i: np.mean([q[i] for q in preds]) for i in ids}, value, ids, gender, gap)
    null = {g: [] for g in ("m", "f")}
    for _ in range(args.permutations):
        shuffled = dict(value)
        for g in ("m", "f"):
            members = [i for i in ids if gender[i] == g]
            shuffled.update(zip(members, rng.permutation([value[i] for i in members])))
        pred = cross_val_predict(V, labels, np.array([shuffled[l] for l in labels]), ids, gender, 5, rng)
        stats = within_gender_stats(pred, shuffled, ids, gender, gap)
        for g in null:
            null[g].append(stats[g]["pearson_r"])
    report["voice_null_pearson_r"] = {
        g: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
            "p_value": float((1 + sum(x >= report["probe"]["voice"][g]["pearson_r"] for x in v)) / (1 + len(v)))}
        for g, v in null.items()}
    args.output_json.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
