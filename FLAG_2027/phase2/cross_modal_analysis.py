"""Phase 3 information and ambiguity analysis of cross-modal models (held-out official people).

For each configuration (checkpoint pattern with {fold}/{seed}) and each fold's 14 unseen people:
  verification  - the fixed fold protocols (standard / same-gender): EER, AUC, genuine/impostor
                  overlap (share of genuine scores below the impostor median)
  retrieval     - voice clip -> face-identity prototypes of the 14 people (any gender / same
                  gender): top-1, top-5, MRR, chance; reciprocal top-1 at the identity level
                  (voice prototype <-> face prototype)
  calibration   - among same-gender genuine trials whose voice ranks its own face identity first,
                  the share scored below the fold's same-gender EER threshold ("ranked right,
                  scored low"), and genuine trials whose identity is outside the top 3
Retrieval and verification are reported separately; they answer different questions.
Output JSON with per-fold-seed values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_validation.compare_fold_runs import face_inputs, load_model, voice_inputs  # noqa: E402
from local_validation.fold_dataset import load_feature_matrices, load_manifest_identities  # noqa: E402
from local_validation.model_evaluation import load_protocol  # noqa: E402


def eer_threshold(labels, s):
    fpr, tpr, thr = roc_curve(labels, s)
    i = np.nanargmin(np.abs((1 - tpr) - fpr))
    return float((fpr[i] + 1 - tpr[i]) / 2), float(thr[i])


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="name=pattern")
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--face-csv", type=Path, default=Path("features/face_v2_farl/train_English_faces.csv"))
    p.add_argument("--voice-csv", type=Path, default=Path("features/voice_v8_wavlm_egemaps/train_English_voices.csv"))
    p.add_argument("--output-json", type=Path, required=True)
    args = p.parse_args()
    F, V, _ = load_feature_matrices(args.face_csv, args.voice_csv)
    Ft, Vt = torch.from_numpy(F), torch.from_numpy(V)
    ident = load_manifest_identities(Path("train_set/train_set/train_English.txt"))
    g = dict(l.strip().split(",") for l in open("meta_file_train_set.csv"))
    report = {"seeds": args.seeds, "score_type": "raw_cosine", "configs": {}}
    for spec in args.runs:
        name, pattern = spec.split("=", 1)
        rows = []
        for fold in range(5):
            fd = Path(f"validation/fold_{fold}")
            val_idx = np.load(fd / "validation_indices.npy").astype(np.int64)
            protos = {t: load_protocol(fd / f"{t}_pairs.csv", fd / f"{t}_ground_truth.csv") for t in ("standard", "gender")}
            for seed in args.seeds:
                path = Path(pattern.format(fold=fold, seed=seed))
                cfg = torch.load(path, map_location="cpu", weights_only=False)["config"]
                voice, face = voice_inputs(cfg, Vt, args.voice_csv), face_inputs(cfg, Ft, args.face_csv)
                model = load_model(path, face.shape[1], voice)
                with torch.no_grad():
                    fe = model.face_branch(face[val_idx]).numpy()
                    ve = model.voice_branch(voice[val_idx]).numpy()
                emb_f, emb_v = {i: e for i, e in zip(val_idx, fe)}, {i: e for i, e in zip(val_idx, ve)}
                r = {"fold": fold, "seed": seed}
                for t, pr in protos.items():
                    lab = np.asarray(pr.labels)
                    s = np.array([emb_f[f] @ emb_v[v] for f, v in zip(pr.face_indices, pr.voice_indices)])
                    e, thr = eer_threshold(lab, s)
                    r[f"{t}_eer"], r[f"{t}_auc"] = e, float(roc_auc_score(lab, s))
                    r[f"{t}_genuine_below_impostor_median"] = float((s[lab == 1] < np.median(s[lab == 0])).mean())
                    if t == "gender":
                        thr_gender = thr
                # Retrieval over the 14 people's face prototypes.
                ids = ident[val_idx]
                people = np.unique(ids)
                fp = unit(np.stack([fe[ids == q].mean(0) for q in people]))
                vp = unit(np.stack([ve[ids == q].mean(0) for q in people]))
                gen = np.array([g[q] for q in people])
                S = ve @ fp.T  # (clips, people)
                truth = np.searchsorted(people, ids)
                for gal, mask in (("any_gender", np.ones_like(S, bool)),
                                  ("same_gender", gen[None, :] == np.array([g[q] for q in ids])[:, None])):
                    Sm = np.where(mask, S, -np.inf)
                    true_s = Sm[np.arange(len(Sm)), truth]
                    rank = (Sm > true_s[:, None]).sum(1) + 1
                    r[f"retrieval_{gal}_top1"] = float((rank == 1).mean())
                    r[f"retrieval_{gal}_top5"] = float((rank <= 5).mean())
                    r[f"retrieval_{gal}_mrr"] = float((1 / rank).mean())
                    r[f"retrieval_{gal}_chance_top1"] = float((1 / mask.sum(1)).mean())
                    if gal == "same_gender":
                        top1 = rank == 1
                        # "ranked right, scored low": correct top-1 but cosine under the same-gender EER threshold
                        r["ranked_right_scored_low"] = float((true_s[top1] < thr_gender).mean()) if top1.any() else np.nan
                        r["correct_outside_top3_same_gender"] = float((rank > 3).mean())
                P = vp @ fp.T
                r["identity_reciprocal_top1"] = float(np.mean([(P[i].argmax() == i) and (P[:, i].argmax() == i)
                                                              for i in range(len(people))]))
                rows.append(r)
        keys = [k for k in rows[0] if k not in ("fold", "seed")]
        report["configs"][name] = {"mean": {k: float(np.nanmean([r[k] for r in rows])) for k in keys}, "per_run": rows}
        m = report["configs"][name]["mean"]
        print(name, {k: round(m[k], 3) for k in ("standard_eer", "gender_eer", "retrieval_any_gender_top1",
                                               "retrieval_same_gender_top1", "retrieval_same_gender_chance_top1",
                                               "identity_reciprocal_top1", "ranked_right_scored_low",
                                               "correct_outside_top3_same_gender")}, flush=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
