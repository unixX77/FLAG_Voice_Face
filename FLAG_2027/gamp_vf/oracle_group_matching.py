"""Stage F diagnostic: identity matching with *oracle* groups (true identities).

Separates the two failure sources of pseudo-labelling: if voice and face sets
were grouped perfectly, how well does the teacher associate them? Reports, for
the hidden pool H (77 people) and the calibration pool C (39):
  * top-1 accuracy of voice prototype -> face prototype (any gender / same gender)
  * Hungarian one-to-one accuracy
  * precision when accepting only the most confident fraction (by margin)
Ground truth defines the groups here, so this is an upper bound, not a method.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamp_vf.matching import hungarian_partial  # noqa: E402
from gamp_vf.prototypes import identity_prototypes  # noqa: E402
from local_validation.compare_fold_runs import load_model  # noqa: E402

LANG = {"English": 0, "Urdu": 1, "German": 2}


def analyse(S, gender_v, gender_f):
    n = len(S)
    out = {"people": n}
    for name, M in (("any_gender", S), ("same_gender", np.where(gender_v[:, None] == gender_f[None, :], S, -2))):
        top = M.argmax(1)
        out[f"top1_{name}"] = float((top == np.arange(n)).mean())
        match = hungarian_partial(M, -1.5)
        out[f"hungarian_{name}"] = float((match == np.arange(n)).mean())
        srt = np.sort(M, 1)
        margin = srt[:, -1] - srt[:, -2]
        order = np.argsort(-margin)
        for frac in (0.1, 0.25, 0.5):
            k = max(1, int(round(frac * n)))
            out[f"precision_top{int(frac*100)}pct_margin_{name}"] = float((top[order[:k]] == order[:k]).mean())
        out[f"chance_top1_{name}"] = float(np.mean([1 / (M[i] > -1.5).sum() for i in range(n)]))
    return out


def main() -> None:
    ext = Path("features/external/combined_all")
    lists = Path("results/gamp_vf/stage_E")
    rows = [l.split() for l in (ext / "train_manifest.txt").open() if l.strip()]
    names = np.array([r[4] for r in rows])
    langs = np.array([LANG.get(r[2].split("/")[1], -1) for r in rows])
    g = dict(l.strip().split(",") for l in (ext / "gender.csv").open())
    faces = np.loadtxt(ext / "train_faces.csv", delimiter=",", dtype=np.float32)[:, :-1]
    voices = np.loadtxt(ext / "train_voices_egemaps.csv", delimiter=",", dtype=np.float32)[:, :-1]
    report = {"note": "oracle groups = true identities (upper bound); teacher = sim_teacher seed 1", "folds": []}
    for fold in range(5):
        path = Path(f"runs/gamp_vf/sim_teacher/fold_{fold}_seed_1/checkpoint_last.pth.tar")
        model = load_model(path, faces.shape[1], torch.from_numpy(voices))
        per = {}
        for pool in ("hidden", "calibration"):
            ids = [l.strip() for l in (lists / f"{pool}_identities.txt").open() if l.strip()]
            r = np.flatnonzero(np.isin(names, ids))
            with torch.no_grad():
                fe = model.face_branch(torch.from_numpy(faces[r])).numpy()
                ve = model.voice_branch(torch.from_numpy(voices[r])).numpy()
            vp = identity_prototypes(ve, names[r], langs[r], "language_balanced")
            fp = identity_prototypes(fe, names[r])
            assert np.array_equal(vp.identities, fp.identities)
            gen = np.array([g[n] for n in vp.identities])
            per[pool] = analyse(vp.prototypes @ fp.prototypes.T, gen, gen)
        report["folds"].append(per)
        print(fold, json.dumps({k: {m: round(v, 3) for m, v in d.items()} for k, d in per.items()}), flush=True)
    keys = report["folds"][0]["hidden"].keys()
    report["mean"] = {pool: {k: float(np.mean([f[pool][k] for f in report["folds"]])) for k in keys}
                      for pool in ("hidden", "calibration")}
    out = Path("results/gamp_vf/stage_F/oracle_group_matching.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report["mean"], indent=1))


if __name__ == "__main__":
    main()
