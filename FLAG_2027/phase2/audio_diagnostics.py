"""Phase 1: audio-only speaker verification, retrieval and representation diagnostics.

Pools (identity groups are evaluated separately; impostors come from the same group):
  official  - 70 official people (English only), split into the 5 official validation folds
              (14 people each), so the B0 fold models' projected voices are on unseen people.
  bilingual - MAV-Celeb v1 (English+Urdu, 64) and v3 (English+German, 49) people, split into
              5 groups by sorted index % 5 per source (= the P2 Urdu-holdout split for v1).
Representations: frozen encoders (raw cosine, centred cosine, LDA backend fitted on
identity-disjoint data) and the B0/P2 cross-modal voice embeddings where valid.
Outputs: results/phase2/phase1_audio/ (trials, raw scores, metrics, retrieval, probes).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gamp_vf.clustering import agglomerative, cluster_metrics  # noqa: E402
from local_validation.compare_fold_runs import load_model  # noqa: E402
from phase2 import unimodal_eval as ue  # noqa: E402

LANG = {"English": 0, "Urdu": 1, "German": 2}
EXT = Path("features/external/combined_all")
OUT = Path("results/phase2/phase1_audio")


def read_csv_matrix(path, drop_last=True):
    m = pd.read_csv(path, header=None, dtype=np.float32).to_numpy()
    return m[:, :-1] if drop_last else m


def layer_means(npy, rows):
    """(len(rows), 13, 768) mean-pooled hidden states for each WavLM layer."""
    arr = np.load(npy, mmap_mode="r")
    return np.asarray(arr[rows, :, :768], dtype=np.float32)


# ---------------------------------------------------------------- data ---------------------------------------
def load_official():
    rows = [l.split() for l in open("train_set/train_set/train_English.txt") if l.strip()]
    ident = np.array([r[4] for r in rows])
    g = dict(l.strip().split(",") for l in open("meta_file_train_set.csv"))
    gender = np.array([1 if g[i] == "m" else 0 for i in ident])
    rowmap = list(csv.DictReader(open("features/voice_v1_wavlm/train_English_voices_rowmap.csv")))
    sha = np.array([r["sha256"] for r in rowmap])
    assert [r["voice_path"] for r in rowmap] == [r[2] for r in rows]
    dur = np.array([sf.info(f"train_set/train_set/{r[2]}").duration for r in rows])
    eg = pd.read_csv("features/egemaps/train_English.csv")
    assert list(eg["key"]) == [r[0] for r in rows]
    f0 = eg["F0semitoneFrom27.5Hz_sma3nz_amean"].to_numpy()
    pool = ue.Pool(ident, gender, np.zeros(len(rows), int), np.full(len(rows), ""), sha, f0, dur)
    reps = {"wavlm_allmean": read_csv_matrix("features/voice_v4_wavlm_allmean/train_English_voices.csv"),
            "wavlm_egemaps": read_csv_matrix("features/voice_v8_wavlm_egemaps/train_English_voices.csv"),
            "ecapa_vox2": read_csv_matrix("features/voice_v6_ecapa/train_English_voices.csv"),
            "wavlm_sv_xvector": read_csv_matrix("features/voice_v1_wavlm/train_English_voices.csv"),
            "w2v2_agegender_hidden": read_csv_matrix("features/voice_v7_agegender/train_English_voices.csv")}
    reps["egemaps"] = reps["wavlm_egemaps"][:, 768:]  # imputed + clipped eGeMAPS block (no NaN)
    layers = layer_means("features/voice_v2_wavlm_layers/train_English_layers.npy", np.arange(len(rows)))
    folds = np.full(len(rows), -1)
    for k in range(5):
        meta = json.load(open(f"validation/fold_{k}/metadata.json"))
        folds[np.isin(ident, meta["validation_identities"])] = k
    assert (folds >= 0).all()
    return pool, reps, layers, folds, [r[0] for r in rows]


def load_external_all():
    rows = [l.split() for l in open(EXT / "train_manifest.txt") if l.strip()]
    ident = np.array([r[4] for r in rows])
    g = dict(l.strip().split(",") for l in open(EXT / "gender.csv"))
    gender = np.array([1 if g[i] == "m" else 0 for i in ident])
    lang = np.array([LANG.get(r[2].split("/")[1], -1) for r in rows])
    video = np.array([r[2].split("/")[-1].rsplit("_", 1)[0] for r in rows])
    path = np.array([r[2] for r in rows])
    v8 = read_csv_matrix(EXT / "train_voices_egemaps.csv")
    eg = pd.concat([pd.read_csv(f"features/egemaps/{s}_train.csv") for s in ("mav1", "mav3", "vox2test")])
    eg = eg.set_index("key").loc[[r[0] for r in rows]]
    f0 = eg["F0semitoneFrom27.5Hz_sma3nz_amean"].to_numpy()
    reps = {"wavlm_allmean": read_csv_matrix(EXT / "train_voices.csv"), "egemaps": v8[:, 768:],
            "wavlm_egemaps": v8}
    # Per-layer WavLM from the per-source arrays, matched by key.
    layers = np.zeros((len(rows), 13, 768), np.float32)
    keys = np.array([r[0] for r in rows])
    for src, folder in (("mav1", "mav1"), ("mav3", "mav3"), ("vox2", "vox2test")):
        man = [l.split()[0] for l in open(f"features/external/manifests/{folder}_train.txt") if l.strip()]
        pos = {k: i for i, k in enumerate(man)}
        sel = np.flatnonzero(np.char.startswith(keys, src + "_"))
        src_rows = np.array([pos[k] for k in keys[sel]])
        order = np.argsort(src_rows)
        layers[sel[order]] = layer_means(f"features/external/{folder}/train_layers.npy", src_rows[order])
    pool = ue.Pool(ident, gender, lang, video, path, f0)
    return pool, reps, layers, keys


def subset(pool: ue.Pool, m: np.ndarray) -> ue.Pool:
    return ue.Pool(*(None if v is None else v[m] for v in (pool.identity, pool.gender, pool.group, pool.session,
                                                           pool.content, pool.attribute, pool.quality)))


# ---------------------------------------------------------------- backends -----------------------------------
class Backend:
    """Centring / LDA fitted on identity-disjoint data only."""

    def __init__(self, train_x, train_ids, kind, n_components=150):
        self.kind = kind
        self.mean = train_x.mean(0)
        self.std = train_x.std(0) + 1e-6
        if kind == "lda":
            n = min(n_components, len(np.unique(train_ids)) - 1, train_x.shape[1])
            self.lda = LinearDiscriminantAnalysis(n_components=n, solver="eigen", shrinkage="auto")
            self.lda.fit((train_x - self.mean) / self.std, train_ids)

    def __call__(self, x):
        if self.kind == "raw":
            return x
        if self.kind == "centred":
            return x - self.mean
        if self.kind == "zscore":
            return (x - self.mean) / self.std
        return self.lda.transform((x - self.mean) / self.std)


# ---------------------------------------------------------------- evaluation ---------------------------------
CONDITIONS_OFFICIAL = {
    "all": (ue.GENUINE, ue.IMPOSTOR[:2]),
    "same_gender": (ue.GENUINE, ("imp_same_gender_same_group",)),
    "diff_gender": (ue.GENUINE, ("imp_diff_gender_same_group",)),
    "similar_f0": (ue.GENUINE, ("imp_similar",)),
}
CONDITIONS_BILINGUAL = {
    "same_lang_same_gender": (("same_group_diff_session",), ("imp_same_gender_same_group",)),
    "cross_lang_same_gender": (("diff_group_diff_session",), ("imp_same_gender_diff_group",)),
    "same_session_same_gender": (("same_group_same_session",), ("imp_same_gender_same_group",)),
    "same_lang_diff_gender": (("same_group_diff_session",), ("imp_diff_gender_same_group",)),
    "similar_f0": (("same_group_diff_session",), ("imp_similar",)),
}


def duration_bins(trials, s, dur):
    out = {}
    d = np.minimum(dur[trials.a], dur[trials.b])
    m_gen = trials.label == 1
    m_imp = trials.condition == "imp_same_gender_same_group"
    for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 1e9)):
        b = (d >= lo) & (d < hi)
        g, i = m_gen & b, m_imp & b
        if g.sum() >= 20 and i.sum() >= 20:
            m = g | i
            out[f"{lo}-{hi if hi < 1e9 else 'inf'}s"] = {**ue.verification_metrics(trials.label[m], s[m])}
    return out


def evaluate_groups(pool, groups, emb_for_group, conditions, per_identity, retrieval_session=False, dur=None):
    """Concatenate per-group trials; returns metrics, retrieval (mean over groups), raw arrays."""
    all_t, all_s = [], []
    retr = []
    for gid in np.unique(groups[groups >= 0]):
        m = np.flatnonzero(groups == gid)
        sub = subset(pool, np.isin(np.arange(len(groups)), m))
        emb = emb_for_group(gid, m)
        if emb is None:
            continue
        t = ue.build_trials(sub, per_identity, seed=int(gid))
        s = ue.scores(emb, t)
        t.a, t.b = m[t.a], m[t.b]  # back to pool row indices
        all_t.append(t), all_s.append(s)
        r = ue.retrieval(emb, sub, exclude_same_session=retrieval_session)
        if pool.group[m].max() > 0:  # cross-language retrieval: non-English queries, English gallery
            r["cross_language"] = ue.retrieval(emb, sub, query_mask=sub.group > 0, gallery_mask=sub.group == 0,
                                               exclude_same_session=True)
        retr.append(r)
    if not all_t:
        return None
    t = ue.Trials(*(np.concatenate([getattr(x, f) for x in all_t]) for f in ("a", "b", "label", "condition")))
    s = np.concatenate(all_s)
    metrics = {name: ue.condition_metrics(t, s, gen, imp) for name, (gen, imp) in conditions.items()}
    if dur is not None:
        metrics["duration_same_gender"] = duration_bins(t, s, dur)

    def mean_dict(ds):
        if isinstance(ds[0], dict):
            return {k: mean_dict([d[k] for d in ds if k in d]) for k in ds[0]}
        return float(np.mean(ds))
    return {"metrics": metrics, "retrieval": mean_dict(retr), "trials": t, "scores": s}


def probes(emb, pool, groups):
    """Identity-disjoint gender/language probes; cross-language identity probe; separation stats."""
    e = ue.unit(emb)
    out = {"fisher_identity": ue.fisher_ratio(e, pool.identity),
           "variance_explained_identity": ue.variance_explained(e, pool.identity),
           "variance_explained_language": ue.variance_explained(e, pool.group),
           "variance_explained_gender": ue.variance_explained(e, pool.gender)}
    gkf = GroupKFold(5)
    for target, y in (("gender", pool.gender), ("language", (pool.group > 0).astype(int))):
        accs = []
        for tr, te in gkf.split(e, y, pool.identity):
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(e[tr], y[tr])
            p = clf.predict(e[te])
            accs.append(np.mean([np.mean(p[y[te] == c] == c) for c in np.unique(y[te])]))  # balanced acc
        out[f"probe_{target}_balanced_accuracy_unseen_people"] = float(np.mean(accs))
    # Cross-language identity probe within each group: train on English clips, test on the
    # other language (closed set), vs English held-out by video.
    xl, wl = [], []
    for gid in np.unique(groups[groups >= 0]):
        m = groups == gid
        en, other = m & (pool.group == 0), m & (pool.group > 0)
        vids = np.unique(pool.session[en])
        rng = np.random.default_rng(int(gid))
        test_v = set(rng.choice(vids, len(vids) // 4, replace=False))
        en_test = en & np.isin(pool.session, list(test_v))
        en_train = en & ~en_test
        ids_ok = np.intersect1d(np.unique(pool.identity[en_train]), np.unique(pool.identity[other]))
        tr = en_train & np.isin(pool.identity, ids_ok)
        clf = LogisticRegression(max_iter=3000, C=10.0).fit(e[tr], pool.identity[tr])
        for target_mask, store in ((other & np.isin(pool.identity, ids_ok), xl),
                                   (en_test & np.isin(pool.identity, ids_ok), wl)):
            if target_mask.any():
                store.append(float((clf.predict(e[target_mask]) == pool.identity[target_mask]).mean()))
    out["identity_probe_english_trained_acc_on_other_language"] = float(np.mean(xl))
    out["identity_probe_english_trained_acc_on_unseen_english_videos"] = float(np.mean(wl))
    return out


def clustering_check(emb, pool, groups):
    """Best-threshold ARI (oracle threshold) vs the tuned-on-other-data setting: is clustering the problem?"""
    res = {}
    for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        aris = []
        for gid in np.unique(groups[groups >= 0]):
            m = groups == gid
            aris.append(cluster_metrics(agglomerative(emb[m], t, 3), pool.identity[m])["ari"])
        res[f"t{t}"] = float(np.mean(aris))
    return {"ari_by_threshold": res, "best_ari": max(res.values())}


def summarise(name, r):
    m = r["metrics"]
    keys = [k for k in m if m[k] is not None and k != "duration_same_gender"]
    return {name: {k: round(m[k]["eer"] * 100, 2) for k in keys} | {
        "R@1": round(r["retrieval"]["recall_at_1"], 3), "MRR": round(r["retrieval"]["mrr"], 3)}}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-identity", type=int, default=40)
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)

    # ------------------------------ official pool ------------------------------
    opool, oreps, olayers, ofolds, _ = load_official()
    xpool, xreps, xlayers, xkeys = load_external_all()
    # Backends for the official pool are fitted on all external people (disjoint).
    report = {"official": {}, "bilingual": {}, "layer_sweep": {"official": {}, "bilingual": {}}}
    raw = {}
    fit_ext = {"wavlm_allmean": xreps["wavlm_allmean"], "egemaps": xreps["egemaps"],
               "wavlm_egemaps": xreps["wavlm_egemaps"]}
    rep_specs = []
    for name, x in oreps.items():
        kinds = ["raw", "centred"] + (["zscore"] if name in ("egemaps", "wavlm_egemaps") else []) \
            + (["lda"] if name in fit_ext else [])
        for kind in kinds:
            if kind in ("centred", "zscore", "lda") and name not in fit_ext:
                if kind != "centred":
                    continue
                # no external copy: centre with the official mean of the *other* folds per group
                rep_specs.append((name, kind, None))
                continue
            rep_specs.append((name, kind, Backend(fit_ext[name], xpool.identity, kind) if name in fit_ext else None))
    for name, kind, backend in rep_specs:
        x = oreps[name]

        def emb_for(gid, m, x=x, backend=backend, kind=kind):
            if backend is not None:
                return backend(x[m])
            if kind == "centred":  # fold-external mean: other official folds
                return x[m] - x[ofolds != gid].mean(0)
            return x[m]
        r = evaluate_groups(opool, ofolds, emb_for, CONDITIONS_OFFICIAL, args.per_identity, dur=opool.quality)
        key = f"{name}|{kind}"
        report["official"][key] = {"metrics": r["metrics"], "retrieval": r["retrieval"]}
        raw[f"official|{key}"] = r["scores"]
        print("official", summarise(key, r), flush=True)
    # B0 cross-modal voice embeddings (fold models on their own held-out people).
    B0 = "runs/traits/full/fold_{k}_seed_1/checkpoint_last.pth.tar"
    v8 = torch.from_numpy(oreps["wavlm_egemaps"])
    models = {k: load_model(Path(B0.format(k=k)), 512, v8) for k in range(5)}

    def proj_official(gid, m):
        return models[gid].voice_branch(v8[m]).numpy()
    r = evaluate_groups(opool, ofolds, proj_official, CONDITIONS_OFFICIAL, args.per_identity, dur=opool.quality)
    report["official"]["b0_full_projected|raw"] = {"metrics": r["metrics"], "retrieval": r["retrieval"]}
    raw["official|b0_full_projected|raw"] = r["scores"]
    trials_off = r["trials"]
    print("official", summarise("b0_full_projected", r), flush=True)
    # Layer sweep (centred with fold-external mean).
    for L in range(13):
        def emb_for(gid, m, L=L):
            return olayers[m, L] - olayers[ofolds != gid, L].mean(0)
        r = evaluate_groups(opool, ofolds, emb_for, {"same_gender": CONDITIONS_OFFICIAL["same_gender"]}, args.per_identity)
        report["layer_sweep"]["official"][f"L{L}"] = {"same_gender_eer": r["metrics"]["same_gender"]["eer"],
                                                      "R@1": r["retrieval"]["recall_at_1"]}

    # ------------------------------ bilingual pool ------------------------------
    names = xpool.identity
    groups = np.full(len(names), -1)
    for src in ("mav1", "mav3"):
        ids = sorted({n for n in names if n.startswith(src + "_")})
        for i, n in enumerate(ids):
            groups[names == n] = i % 5 + (0 if src == "mav1" else 5)  # mav1 groups 0-4 (= P2 split), mav3 5-9
    bil = groups >= 0
    bpool, bgroups = subset(xpool, bil), groups[bil]
    # Backends for the bilingual pool: fitted on official + VoxCeleb2 people (disjoint from MAV-Celeb).
    fit_mask = ~bil
    fit_b = {n: (np.concatenate([oreps[n], xreps[n][fit_mask]]),
                 np.concatenate([opool.identity, xpool.identity[fit_mask]])) for n in fit_ext}
    for name in fit_ext:
        x = xreps[name][bil]
        kinds = ["raw", "centred", "lda"] + (["zscore"] if name in ("egemaps", "wavlm_egemaps") else [])
        for kind in kinds:
            backend = Backend(*fit_b[name], kind)
            r = evaluate_groups(bpool, bgroups, lambda gid, m, x=x, b=backend: b(x[m]), CONDITIONS_BILINGUAL,
                                args.per_identity, retrieval_session=True)
            key = f"{name}|{kind}"
            entry = {"metrics": r["metrics"], "retrieval": r["retrieval"]}
            if kind in ("centred", "lda"):
                entry["probes"] = probes(backend(x), bpool, bgroups)
                entry["clustering"] = clustering_check(backend(x), bpool, bgroups)
            report["bilingual"][key] = entry
            raw[f"bilingual|{key}"] = r["scores"]
            print("bilingual", summarise(key, r), flush=True)
    trials_bil = r["trials"]
    # P2 cross-modal voice embeddings: mav1 groups 0-4 only (people held out of that P2 fold, Urdu unseen).
    P2 = "runs/traits/p2_full/fold_{k}_seed_1/checkpoint_last.pth.tar"
    xv8 = torch.from_numpy(xreps["wavlm_egemaps"][bil])
    p2 = {k: load_model(Path(P2.format(k=k)), 512, xv8) for k in range(5)}
    for k in range(5):
        held = set(torch.load(P2.format(k=k), map_location="cpu", weights_only=False)["config"]["held_out_external"])
        assert set(bpool.identity[bgroups == k]) == held, "group split must equal the P2 hold-out"

    def proj_bil(gid, m):
        return p2[gid].voice_branch(xv8[m]).numpy() if gid < 5 else None
    proj_full = np.zeros((len(bgroups), 128), np.float32)  # row-aligned with bpool
    for k in range(5):
        proj_full[bgroups == k] = proj_bil(k, np.flatnonzero(bgroups == k))
    r = evaluate_groups(bpool, bgroups, proj_bil, CONDITIONS_BILINGUAL, args.per_identity, retrieval_session=True)
    report["bilingual"]["p2_full_projected|raw(mav1 only)"] = {
        "metrics": r["metrics"], "retrieval": r["retrieval"],
        "probes": probes(proj_full[bgroups < 5], subset(bpool, bgroups < 5), bgroups[bgroups < 5])}
    print("bilingual", summarise("p2_full_projected (mav1)", r), flush=True)
    # Same representations restricted to the mav1 groups, for a like-for-like comparison.
    for name in ("wavlm_allmean", "wavlm_egemaps"):
        backend = Backend(*fit_b[name], "lda")
        x = xreps[name][bil]
        r = evaluate_groups(bpool, np.where(bgroups < 5, bgroups, -1), lambda gid, m, x=x, b=backend: b(x[m]),
                            CONDITIONS_BILINGUAL, args.per_identity, retrieval_session=True)
        report["bilingual"][f"{name}|lda(mav1 only)"] = {"metrics": r["metrics"], "retrieval": r["retrieval"]}
    for L in range(13):
        b = Backend(np.concatenate([olayers[:, L], xlayers[fit_mask, L]]), None, "centred")
        r = evaluate_groups(bpool, bgroups, lambda gid, m, L=L, b=b: b(xlayers[bil][m, L]),
                            {k: CONDITIONS_BILINGUAL[k] for k in ("same_lang_same_gender", "cross_lang_same_gender")},
                            args.per_identity, retrieval_session=True)
        report["layer_sweep"]["bilingual"][f"L{L}"] = {k: v["eer"] for k, v in r["metrics"].items()} | {
            "R@1": r["retrieval"]["recall_at_1"], "cross_language_R@1": r["retrieval"]["cross_language"]["recall_at_1"]}

    # ------------------------------ save ------------------------------
    np.savez_compressed(OUT / "trials.npz", official_a=trials_off.a, official_b=trials_off.b,
                        official_label=trials_off.label, official_condition=trials_off.condition,
                        bilingual_a=trials_bil.a, bilingual_b=trials_bil.b, bilingual_label=trials_bil.label,
                        bilingual_condition=trials_bil.condition)
    np.savez_compressed(OUT / "scores.npz", **{k.replace("|", "__"): v for k, v in raw.items()})
    (OUT / "metrics.json").write_text(json.dumps(report, indent=1, default=float))
    print("saved", OUT)


if __name__ == "__main__":
    main()
