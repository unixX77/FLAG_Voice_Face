"""Phase 2: face-only identity verification, retrieval and quality diagnostics.

Same protocol as Phase 1 (phase2.unimodal_eval), with faces:
  official - 70 people in the 5 official validation folds (14 each). Encoders: InsightFace
             ArcFace (buffalo_l), FaRL (+ flip), DINOv2 (L12 patch, L1-12 CLS), FaRL+DINOv2,
             and B0's projected face embedding (fold models, unseen people).
  external - all 231 external people (FaRL only; sessions = source video), in 10 groups.
Genuine pairs never share an image hash. "Similar" impostors = same-gender person with the
closest median estimated age. Quality: InsightFace detection score and face-box size bins.
Outputs: results/phase2/phase2_face/.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamp_vf.clustering import agglomerative, cluster_metrics  # noqa: E402
from local_validation.compare_fold_runs import load_model  # noqa: E402
from phase2 import unimodal_eval as ue  # noqa: E402
from phase2.audio_diagnostics import Backend, evaluate_groups, read_csv_matrix, subset  # noqa: E402

OUT = Path("results/phase2/phase2_face")
EXT = Path("features/external/combined_all")
CONDITIONS = {
    "all": (ue.GENUINE, ue.IMPOSTOR[:2]),
    "same_gender": (ue.GENUINE, ("imp_same_gender_same_group",)),
    "diff_gender": (ue.GENUINE, ("imp_diff_gender_same_group",)),
    "similar_age": (ue.GENUINE, ("imp_similar",)),
    "diff_session_same_gender": (("same_group_diff_session",), ("imp_same_gender_same_group",)),
    "same_session_same_gender": (("same_group_same_session",), ("imp_same_gender_same_group",)),
}


def quality_bins(trials, s, q, edges, name):
    out = {}
    v = np.minimum(q[trials.a], q[trials.b])
    gen, imp = trials.label == 1, trials.condition == "imp_same_gender_same_group"
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (v >= lo) & (v < hi)
        if (gen & b).sum() >= 20 and (imp & b).sum() >= 20:
            m = (gen | imp) & b
            out[f"{name}_{lo:g}-{hi:g}"] = ue.verification_metrics(trials.label[m], s[m])
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    rows = [l.split() for l in open("train_set/train_set/train_English.txt") if l.strip()]
    ident = np.array([r[4] for r in rows])
    g = dict(l.strip().split(",") for l in open("meta_file_train_set.csv"))
    gender = np.array([1 if g[i] == "m" else 0 for i in ident])
    rm = list(csv.DictReader(open("features/face_v1_insightface/train_English_faces_rowmap.csv")))
    assert [r["face_path"] for r in rm] == [r[3] for r in rows]
    sha = np.array([r["sha256"] for r in rm])
    det = np.array([float(r["detection_score"] or "nan") for r in rm])
    box = np.array([np.sqrt(max(float(r["bbox_x2"] or 0) - float(r["bbox_x1"] or 0), 0) *
                            max(float(r["bbox_y2"] or 0) - float(r["bbox_y1"] or 0), 0)) for r in rm])
    age = pd.read_csv("features/face_age/train_English_face_age.csv")["age"].to_numpy(float)
    pool = ue.Pool(ident, gender, np.zeros(len(rows), int), np.full(len(rows), ""), sha, age, det)
    folds = np.full(len(rows), -1)
    for k in range(5):
        folds[np.isin(ident, json.load(open(f"validation/fold_{k}/metadata.json"))["validation_identities"])] = k
    reps = {"arcface_insightface": read_csv_matrix("features/face_v1_insightface/train_English_faces.csv"),
            "farl": read_csv_matrix("features/face_v2_farl/train_English_faces.csv"),
            "farl_flip": read_csv_matrix("features/face_v3_farl_final_flip/train_English_faces.csv"),
            "dinov2_l12_patch": read_csv_matrix("features/face_v5_dinov2_l12_patch/train_English_faces.csv"),
            "dinov2_l1_12_cls": read_csv_matrix("features/face_v6_dinov2_l1_12_cls/train_English_faces.csv"),
            "farl_flip_dinov2": read_csv_matrix("features/face_v7_farl_flip_dinov2/train_English_faces.csv")}
    report = {"official": {}, "external": {}, "duplicates": {}}
    raw = {}
    # Exact duplicate images in the official set (same hash, different rows).
    _, counts = np.unique(sha, return_counts=True)
    report["duplicates"]["official_rows"] = len(sha)
    report["duplicates"]["official_unique_images"] = int(len(counts))
    for name, x in reps.items():
        for kind in ("raw", "centred"):
            def emb_for(gid, m, x=x, kind=kind):
                return x[m] - x[folds != gid].mean(0) if kind == "centred" else x[m]
            r = evaluate_groups(pool, folds, emb_for, CONDITIONS, 40)
            t, s = r["trials"], r["scores"]
            r["metrics"].update(quality_bins(t, s, det, [0, 0.5, 0.65, 0.75, 1.01], "detection_score"))
            r["metrics"].update(quality_bins(t, s, box, [0, 100, 130, 160, 1e9], "face_box_px"))
            report["official"][f"{name}|{kind}"] = {"metrics": r["metrics"], "retrieval": r["retrieval"]}
            raw[f"official__{name}__{kind}"] = s
            print("official", name, kind, {k: round(v["eer"] * 100, 2) for k, v in r["metrics"].items()
                                           if v and not k.startswith(("detection", "face_box"))},
                  "R@1", round(r["retrieval"]["recall_at_1"], 3), flush=True)
    # B0 projected faces on unseen people.
    v8 = torch.from_numpy(read_csv_matrix("features/voice_v8_wavlm_egemaps/train_English_voices.csv"))
    ff = torch.from_numpy(reps["farl"])
    models = {k: load_model(Path(f"runs/traits/full/fold_{k}_seed_1/checkpoint_last.pth.tar"), 512, v8) for k in range(5)}
    r = evaluate_groups(pool, folds, lambda gid, m: models[gid].face_branch(ff[m]).numpy(), CONDITIONS, 40)
    report["official"]["b0_full_projected|raw"] = {"metrics": r["metrics"], "retrieval": r["retrieval"]}
    raw["official__b0_full_projected__raw"] = r["scores"]
    trials_off = r["trials"]
    print("official b0 projected", {k: round(v["eer"] * 100, 2) for k, v in r["metrics"].items() if v},
          "R@1", round(r["retrieval"]["recall_at_1"], 3), flush=True)

    # External faces (FaRL), sessions = source video of the frame.
    erows = [l.split() for l in open(EXT / "train_manifest.txt") if l.strip()]
    eid = np.array([r[4] for r in erows])
    eg = dict(l.strip().split(",") for l in open(EXT / "gender.csv"))
    egender = np.array([1 if eg[i] == "m" else 0 for i in eid])
    evideo = np.array([r[3].split("/")[-1].rsplit("_", 1)[0] for r in erows])
    epath = np.array([r[3] for r in erows])  # face images reused across rows share a path
    ages = pd.concat([pd.read_csv(f"features/external/{s}/train_face_age.csv") for s in ("mav1", "mav3", "vox2test")])
    ages = ages.set_index("key").loc[[r[0] for r in erows]]
    epool = ue.Pool(eid, egender, np.zeros(len(erows), int), evideo, epath, ages["age"].to_numpy(float),
                    ages["det_score"].to_numpy(float))
    ids = sorted(set(eid))
    egroups = np.array([ids.index(i) % 10 for i in eid])
    ex = read_csv_matrix(EXT / "train_faces.csv")
    for kind in ("raw", "centred"):
        b = Backend(reps["farl"], ident, kind if kind != "raw" else "raw")  # official mean: disjoint people
        r = evaluate_groups(epool, egroups, lambda gid, m, b=b: b(ex[m]), CONDITIONS, 40, retrieval_session=True)
        r["metrics"].update(quality_bins(r["trials"], r["scores"], epool.quality, [0, 0.6, 0.75, 0.85, 1.01],
                                         "detection_score"))
        entry = {"metrics": r["metrics"], "retrieval": r["retrieval"],
                 "retrieval_including_same_video": ue.retrieval(b(ex[egroups == 0]), subset(epool, egroups == 0))}
        aris = {t: float(np.mean([cluster_metrics(agglomerative(b(ex[egroups == k]), t, 3), eid[egroups == k])["ari"]
                                  for k in range(10)])) for t in (0.2, 0.3, 0.4, 0.5)}
        entry["clustering_ari_by_threshold"] = aris
        entry["fisher_identity"] = ue.fisher_ratio(b(ex), eid)
        report["external"][f"farl|{kind}"] = entry
        raw[f"external__farl__{kind}"] = r["scores"]
        print("external farl", kind, {k: round(v["eer"] * 100, 2) for k, v in r["metrics"].items()
                                      if v and not k.startswith("detection")},
              "R@1 (other videos)", round(r["retrieval"]["recall_at_1"], 3), flush=True)
    trials_ext = r["trials"]
    np.savez_compressed(OUT / "trials.npz", official_a=trials_off.a, official_b=trials_off.b,
                        official_label=trials_off.label, official_condition=trials_off.condition,
                        external_a=trials_ext.a, external_b=trials_ext.b, external_label=trials_ext.label,
                        external_condition=trials_ext.condition)
    np.savez_compressed(OUT / "scores.npz", **raw)
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1, default=float))
    print("saved", OUT)


if __name__ == "__main__":
    main()
