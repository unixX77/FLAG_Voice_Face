"""Phase 3 diagnostic: does the cross-modal score exploit a shared recording session?

On people unseen by the model (P2 fold models: MAV-Celeb v1 people held out of that fold), build
voice-clip x face-image trials from the clip/frame video IDs:
  genuine_same_video   voice and face from the same source video
  genuine_other_video  same person, different videos
  impostor             same-gender other person (always another video)
If same-video genuine pairs score much higher than other-video ones, the model uses session/
channel cues shared by the audio and the frame, not only the person. Deterministic sampling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_validation.compare_fold_runs import load_model  # noqa: E402
from phase2 import unimodal_eval as ue  # noqa: E402
from phase2.audio_diagnostics import EXT, read_csv_matrix  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="name=pattern with {fold}")
    p.add_argument("--per-person", type=int, default=60)
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    rows = [l.split() for l in open(EXT / "train_manifest.txt") if l.strip()]
    ident = np.array([r[4] for r in rows])
    vvid = np.array([r[2].split("/")[-1].rsplit("_", 1)[0] for r in rows])
    fvid = np.array([r[3].split("/")[-1].rsplit("_", 1)[0] for r in rows])
    gmap = dict(l.strip().split(",") for l in open(EXT / "gender.csv"))
    cache = {}
    report = {"configs": {}}
    for spec in args.runs:
        name, pattern = spec.split("=", 1)
        per = {"genuine_same_video": [], "genuine_other_video": [], "impostor_same_gender": []}
        eers = {"same_video": [], "other_video": []}
        for fold in range(5):
            path = Path(pattern.format(fold=fold))
            cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
            for key in ("extra_voice_csv", "extra_face_csv"):
                if cfg[key] not in cache:
                    cache[cfg[key]] = torch.from_numpy(read_csv_matrix(cfg[key]))
            V, F = cache[cfg["extra_voice_csv"]], cache[cfg["extra_face_csv"]]
            held = sorted(cfg["held_out_external"])
            m = np.flatnonzero(np.isin(ident, held))
            model = load_model(path, F.shape[1], V)
            with torch.no_grad():
                ve, fe = ue.unit(model.voice_branch(V[m]).numpy()), ue.unit(model.face_branch(F[m]).numpy())
            rng = np.random.default_rng(fold)
            idm = ident[m]
            gs, go, im = [], [], []
            for person in held:
                rows_p = np.flatnonzero(idm == person)
                others = np.flatnonzero((idm != person) & np.array([gmap[x] == gmap[person] for x in idm]))
                v_choice = rng.choice(rows_p, min(args.per_person, len(rows_p)), replace=False)
                for vi in v_choice:
                    same = rows_p[fvid[m][rows_p] == vvid[m][vi]]
                    diff = rows_p[fvid[m][rows_p] != vvid[m][vi]]
                    if len(same) and len(diff):  # only voices that have both kinds of genuine faces
                        gs.append(float(ve[vi] @ fe[rng.choice(same)]))
                        go.append(float(ve[vi] @ fe[rng.choice(diff)]))
                        im.append(float(ve[vi] @ fe[rng.choice(others)]))
            gs, go, im = map(np.asarray, (gs, go, im))
            per["genuine_same_video"] += gs.tolist()
            per["genuine_other_video"] += go.tolist()
            per["impostor_same_gender"] += im.tolist()
            eers["same_video"].append(ue.eer(np.r_[np.ones(len(gs)), np.zeros(len(im))], np.r_[gs, im]))
            eers["other_video"].append(ue.eer(np.r_[np.ones(len(go)), np.zeros(len(im))], np.r_[go, im]))
        out = {k: {"n": len(v), "mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in per.items()}
        out["eer_same_video_genuine"] = float(np.mean(eers["same_video"]))
        out["eer_other_video_genuine"] = float(np.mean(eers["other_video"]))
        # Paired: same voice clip, same-video face vs other-video face.
        d = np.asarray(per["genuine_same_video"]) - np.asarray(per["genuine_other_video"])
        out["paired_same_minus_other_mean"] = float(d.mean())
        out["paired_share_same_video_higher"] = float((d > 0).mean())
        report["configs"][name] = out
        print(name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)
    args.output_json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
