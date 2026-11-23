"""Stage A/B diagnostics: speaker/face prototypes, cross-language consistency, clustering.

Uses the P2 fold models (runs/traits/p2_full): in fold k, 1/5 of the MAV-Celeb v1
people (English + Urdu) are held out of training and no Urdu is ever trained on.
Those held-out people are unseen identities speaking a seen and an unseen
language, so every statistic here is on identities the model never saw.

Outputs (results/gamp_vf/stage_A_B/):
  speaker_similarity.json   same/different speaker x same/different language cosines
  clustering.json           ARI / NMI / purity / fragmentation (voices and faces)
  prototype_retrieval.json  voice->face identity retrieval: recordings vs prototypes
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
from gamp_vf.clustering import agglomerative, cluster_metrics, hdbscan  # noqa: E402
from gamp_vf.prototypes import identity_prototypes, speaker_similarity_statistics  # noqa: E402
from local_validation.compare_fold_runs import load_model  # noqa: E402

LANG = {"English": 0, "Urdu": 1, "German": 2}


def retrieval(voice_queries, query_ids, face_protos, face_ids, genders_q=None, genders_f=None):
    """Top-1 accuracy and mean rank of the true identity (all pools / same-gender pools)."""
    s = voice_queries @ face_protos.T
    out = {}
    for name, mask in (("any_gender", None), ("same_gender", "g")):
        if mask is not None:
            s_use = np.where(genders_q[:, None] == genders_f[None, :], s, -np.inf)
        else:
            s_use = s
        true_col = np.searchsorted(face_ids, query_ids)
        true_score = s_use[np.arange(len(s_use)), true_col]
        rank = (s_use > true_score[:, None]).sum(1) + 1
        pool = np.isfinite(s_use).sum(1)
        out[name] = {"top1": float((rank == 1).mean()), "mean_rank": float(rank.mean()),
                     "mean_pool": float(pool.mean()), "chance_top1": float((1 / pool).mean())}
    return out


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="runs/traits/p2_full/fold_{fold}_seed_1/checkpoint_last.pth.tar")
    p.add_argument("--ext", type=Path, default=Path("features/external/combined_all"))
    p.add_argument("--out", type=Path, default=Path("results/gamp_vf/stage_A_B"))
    p.add_argument("--voice-file", default="train_voices_egemaps.csv", help="external voice CSV the models use")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows = [l.split() for l in (args.ext / "train_manifest.txt").open() if l.strip()]
    names = np.array([r[4] for r in rows])
    lang = np.array([LANG.get(r[2].split("/")[1], -1) for r in rows])
    gender_of = dict(l.strip().split(",") for l in (args.ext / "gender.csv").open())
    gender = np.array([0 if gender_of[n] == "f" else 1 for n in names])
    faces = np.loadtxt(args.ext / "train_faces.csv", delimiter=",", dtype=np.float32)[:, :-1]
    voices = np.loadtxt(args.ext / args.voice_file, delimiter=",", dtype=np.float32)[:, :-1]
    wavlm = voices[:, :768]

    sim_report, clus_report, retr_report = {}, {"voice": {}, "face": {}}, {}
    pooled = {"raw_wavlm": [], "projected": []}
    per_fold_retrieval = []
    for fold in range(5):
        path = Path(args.runs.format(fold=fold))
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        held = sorted(ckpt["config"]["held_out_external"])
        rows_h = np.flatnonzero(np.isin(names, held))
        model = load_model(path, faces.shape[1], torch.from_numpy(voices))
        with torch.no_grad():
            fe = model.face_branch(torch.from_numpy(faces[rows_h])).numpy()
            ve = model.voice_branch(torch.from_numpy(voices[rows_h])).numpy()
        ids_h, lang_h, g_h = names[rows_h], lang[rows_h], gender[rows_h]
        pooled["raw_wavlm"].append(speaker_similarity_statistics(wavlm[rows_h], ids_h, lang_h, seed=fold))
        pooled["projected"].append(speaker_similarity_statistics(ve, ids_h, lang_h, seed=fold))

        # Clustering of the held-out people (unknown identities); ground truth for metrics only.
        for modality, emb in (("voice", ve), ("face", fe)):
            res = {}
            for t in (0.3, 0.4, 0.5, 0.6, 0.7):
                res[f"agglomerative_t{t}"] = cluster_metrics(agglomerative(emb, t, min_size=3), ids_h)
            res["hdbscan_m10"] = cluster_metrics(hdbscan(emb, 10), ids_h)
            clus_report[modality][f"fold_{fold}"] = res
        raw_face_res = {f"agglomerative_t{t}": cluster_metrics(agglomerative(faces[rows_h], t, 3), ids_h)
                        for t in (0.1, 0.15, 0.2, 0.3)}
        clus_report.setdefault("face_raw_farl", {})[f"fold_{fold}"] = raw_face_res
        raw_voice_res = {f"agglomerative_t{t}": cluster_metrics(agglomerative(wavlm[rows_h], t, 3), ids_h)
                         for t in (0.02, 0.05, 0.1, 0.2)}
        clus_report.setdefault("voice_raw_wavlm", {})[f"fold_{fold}"] = raw_voice_res

        # Association recovery among held-out people: voice query -> face identity prototype.
        fp = identity_prototypes(fe, ids_h)
        face_g = np.array([0 if gender_of[n] == "f" else 1 for n in fp.identities])
        r = {"recording": retrieval(unit(ve), ids_h, fp.prototypes, fp.identities, g_h, face_g)}
        for mode in ("uniform", "language_balanced"):
            vp = identity_prototypes(ve, ids_h, lang_h, mode)
            vg = np.array([0 if gender_of[n] == "f" else 1 for n in vp.identities])
            r[f"prototype_{mode}"] = retrieval(vp.prototypes, vp.identities, fp.prototypes, fp.identities, vg, face_g)
        for lname, lid in (("English", 0), ("Urdu", 1)):
            keep = lang_h == lid
            vp = identity_prototypes(ve[keep], ids_h[keep])
            vg = np.array([0 if gender_of[n] == "f" else 1 for n in vp.identities])
            r[f"prototype_{lname}_only"] = retrieval(vp.prototypes, vp.identities, fp.prototypes, fp.identities, vg, face_g)
            r[f"recording_{lname}_only"] = retrieval(unit(ve[keep]), ids_h[keep], fp.prototypes, fp.identities,
                                                     g_h[keep], face_g)
        r["held_out_identities"] = len(held)
        r["held_out_rows"] = int(len(rows_h))
        per_fold_retrieval.append(r)
        print(f"fold {fold}: {len(held)} held-out people, {len(rows_h)} rows", flush=True)

    def mean_dicts(ds):
        if isinstance(ds[0], dict):
            return {k: mean_dicts([d[k] for d in ds]) for k in ds[0]}
        return float(np.mean(ds))

    sim_report = {k: {"mean_over_folds": mean_dicts(v), "per_fold": v} for k, v in pooled.items()}
    retr_report = {"mean_over_folds": mean_dicts(per_fold_retrieval), "per_fold": per_fold_retrieval}
    clus_summary = {m: mean_dicts(list(v.values())) for m, v in clus_report.items()}
    meta = {"models": args.runs, "note": "held-out MAV-Celeb v1 identities (English+Urdu); Urdu never trained on",
            "score_type": "raw_cosine"}
    (args.out / "speaker_similarity.json").write_text(json.dumps({**meta, **sim_report}, indent=1))
    (args.out / "clustering.json").write_text(json.dumps({**meta, "mean_over_folds": clus_summary,
                                                          "per_fold": clus_report}, indent=1))
    (args.out / "prototype_retrieval.json").write_text(json.dumps({**meta, **retr_report}, indent=1))
    print(json.dumps({"speaker_similarity": {k: v["mean_over_folds"] for k, v in sim_report.items()},
                      "retrieval": retr_report["mean_over_folds"]}, indent=1))


if __name__ == "__main__":
    main()
