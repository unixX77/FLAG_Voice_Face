"""E1a: frozen-feature complementarity of B0's voice input (WavLM 768 + eGeMAPS 88) and ECAPA (192).

All fitting is identity-disjoint: ridge maps and probes are fitted on one set of people and
evaluated on others (official + VoxCeleb2 people -> MAV-Celeb people, as in Phase 1).
Reports (results/phase3/e1_features/complementarity.json):
  * linear CKA (clip level and person-mean level) between the blocks; effective rank and
    PCA spectrum; norms; per-language shift of the same speaker (bilingual MAV-Celeb people)
  * redundancy: R^2 of ridge predicting ECAPA from B0 features and vice versa (unseen people)
  * the ECAPA *residual* (ECAPA minus its ridge prediction from B0 features): speaker
    verification on the Phase 1 bilingual trials (same / cross language, same gender), and
    person-level face-trait probes - does the part of ECAPA that B0 lacks carry identity /
    language-stable / face-related information?
CKA measures representational similarity, not redundancy for a downstream task or causal use.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phase2 import unimodal_eval as ue  # noqa: E402
from phase2.audio_diagnostics import (CONDITIONS_BILINGUAL, EXT, evaluate_groups, load_external_all,  # noqa: E402
                                      read_csv_matrix, subset)

OUT = Path("results/phase3/e1_features")


def zscore(train, *others):
    mu, sd = train.mean(0), train.std(0) + 1e-6
    return [(x - mu) / sd for x in (train, *others)]


def linear_cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    hsic = np.linalg.norm(x.T @ y, "fro") ** 2
    return float(hsic / (np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")))


def spectrum(x):
    ev = np.linalg.eigvalsh(np.cov(x - x.mean(0), rowvar=False))[::-1].clip(min=0)
    p = ev / ev.sum()
    return {"effective_rank": float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum())),
            "var_top10": float(p[:10].sum()), "var_top50": float(p[:50].sum()), "dims": int(len(ev))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    off = [l.split() for l in open("train_set/train_set/train_English.txt") if l.strip()]
    xpool, xreps, _, _ = load_external_all()
    n_off = len(off)
    ident = np.concatenate([np.array([r[4] for r in off]), xpool.identity])
    g = dict(l.strip().split(",") for l in open("meta_file_train_set.csv"))
    g.update(dict(l.strip().split(",") for l in open(EXT / "gender.csv")))
    gender = np.array([1 if g[i] == "m" else 0 for i in ident])
    lang = np.concatenate([np.zeros(n_off, int), xpool.group])
    B = np.vstack([read_csv_matrix("features/voice_v8_wavlm_egemaps/train_English_voices.csv"), xreps["wavlm_egemaps"]])
    E = np.vstack([read_csv_matrix("features/voice_v6_ecapa/train_English_voices.csv"),
                   read_csv_matrix(EXT / "train_voices_ecapa.csv")])
    bil = np.char.startswith(ident.astype(str), "mav1_") | np.char.startswith(ident.astype(str), "mav3_")
    fit = ~bil  # official + VoxCeleb2 people: fitting set; MAV-Celeb people: evaluation set
    Bz_fit, Bz_all = zscore(B[fit], B)
    Ez_fit, Ez_all = zscore(E[fit], E)
    rng = np.random.default_rng(0)
    sub = rng.choice(np.flatnonzero(bil), 6000, replace=False)
    people = np.unique(ident[bil])
    pm = lambda X: np.stack([X[ident == p].mean(0) for p in people])  # noqa: E731
    rep = {"rows": {"fit": int(fit.sum()), "eval": int(bil.sum())}}
    rep["cka"] = {"clip_level": {"b0_vs_ecapa": linear_cka(Bz_all[sub], Ez_all[sub]),
                                 "wavlm_vs_ecapa": linear_cka(Bz_all[sub, :768], Ez_all[sub]),
                                 "egemaps_vs_ecapa": linear_cka(Bz_all[sub, 768:], Ez_all[sub]),
                                 "wavlm_vs_egemaps": linear_cka(Bz_all[sub, :768], Bz_all[sub, 768:])},
                  "person_mean_level": {"b0_vs_ecapa": linear_cka(pm(Bz_all), pm(Ez_all))}}
    rep["spectrum"] = {"b0_voice": spectrum(Bz_all[sub]), "wavlm": spectrum(Bz_all[sub, :768]),
                       "egemaps": spectrum(Bz_all[sub, 768:]), "ecapa": spectrum(Ez_all[sub])}
    rep["raw_norms"] = {"b0_voice_mean": float(np.linalg.norm(B[bil], axis=1).mean()),
                        "ecapa_mean": float(np.linalg.norm(E[bil], axis=1).mean()),
                        "ecapa_cv": float(np.linalg.norm(E[bil], axis=1).std() / np.linalg.norm(E[bil], axis=1).mean())}
    # Same-speaker language shift (bilingual people with >= 5 clips in each language), relative to within spread.
    shifts = {}
    for name, X in (("b0_voice", Bz_all), ("ecapa", Ez_all)):
        rel = []
        for p in people:
            m = ident == p
            ls = [l for l in np.unique(lang[m]) if l >= 0 and (m & (lang == l)).sum() >= 5]
            if len(ls) == 2:
                a, b = X[m & (lang == ls[0])], X[m & (lang == ls[1])]
                within = 0.5 * (np.linalg.norm(a - a.mean(0), axis=1).mean() + np.linalg.norm(b - b.mean(0), axis=1).mean())
                rel.append(np.linalg.norm(a.mean(0) - b.mean(0)) / within)
        shifts[name] = {"people": len(rel), "language_shift_over_within_spread_median": float(np.median(rel))}
    rep["same_speaker_language_shift"] = shifts
    # Redundancy: ridge predictions on unseen people.
    def r2(model, Xtr, Ytr, Xte, Yte):
        model.fit(Xtr, Ytr)
        P = model.predict(Xte)
        return float(1 - ((Yte - P) ** 2).sum() / ((Yte - Yte.mean(0)) ** 2).sum()), model
    r2_e_from_b, ridge = r2(RidgeCV(alphas=np.logspace(0, 4, 9)), Bz_fit, Ez_fit, Bz_all[bil], Ez_all[bil])
    r2_b_from_e, _ = r2(RidgeCV(alphas=np.logspace(0, 4, 9)), Ez_fit, Bz_fit, Ez_all[bil], Bz_all[bil])
    rep["redundancy_r2_unseen_people"] = {"ecapa_from_b0": r2_e_from_b, "b0_from_ecapa": r2_b_from_e,
                                          "ridge_alpha_ecapa_from_b0": float(ridge.alpha_)}
    resid = Ez_all - ridge.predict(Bz_all)     # ECAPA part not linearly predictable from B0 features
    pred = ridge.predict(Bz_all)               # ECAPA part that B0 already carries
    # Speaker verification of each part on the Phase 1 bilingual groups (same trials as Phase 1).
    names = xpool.identity
    groups = np.full(len(names), -1)
    for src in ("mav1", "mav3"):
        ids = sorted({n for n in names if n.startswith(src + "_")})
        for i, n in enumerate(ids):
            groups[names == n] = i % 5 + (0 if src == "mav1" else 5)
    xb = groups >= 0
    bpool, bgroups = subset(xpool, xb), groups[xb]
    rows_x = np.flatnonzero(bil[n_off:]) + n_off  # bilingual rows in the stacked arrays (same order as xb)
    assert np.array_equal(np.flatnonzero(xb), rows_x - n_off)
    sv = {}
    for name, X in (("ecapa_full", Ez_all), ("ecapa_predictable_from_b0", pred), ("ecapa_residual", resid),
                    ("b0_voice_zscore", Bz_all)):
        Xb = X[rows_x]
        r = evaluate_groups(bpool, bgroups, lambda gid, m, Xb=Xb: Xb[m] - Xb.mean(0), CONDITIONS_BILINGUAL, 40,
                            retrieval_session=True)
        sv[name] = {k: v["eer"] for k, v in r["metrics"].items() if v} | {
            "R@1": r["retrieval"]["recall_at_1"], "cross_language_R@1": r["retrieval"]["cross_language"]["recall_at_1"]}
        print(name, {k: round(v * 100 if k not in ("R@1", "cross_language_R@1") else v, 2) for k, v in sv[name].items()},
              flush=True)
    rep["speaker_verification_bilingual_centred"] = sv
    # Person-level probes on all 301 people (GroupKFold): face age / build within gender, gender, language.
    tr = pd.concat([pd.read_csv("features/traits/train_English_traits.csv"),
                    pd.read_csv("features/traits/combined_all_traits.csv")])
    keys = [r[0] for r in off] + [l.split()[0] for l in open(EXT / "train_manifest.txt") if l.strip()]
    tr = tr.set_index("key").loc[keys]
    allp = np.unique(ident)
    gp = np.array([g[p] for p in allp])
    tgt = {t: np.array([np.nanmedian(tr[t].to_numpy()[ident == p]) for p in allp]) for t in ("age", "build")}
    probes = {}
    for name, X in (("b0_voice", Bz_all), ("ecapa", Ez_all), ("ecapa_residual", resid), ("b0+ecapa", np.hstack([Bz_all, Ez_all]))):
        P = np.stack([X[ident == p].mean(0) for p in allp])
        out = {}
        for t, y in tgt.items():
            pr = np.full(len(y), np.nan)
            for a, b in GroupKFold(5).split(P, y, np.arange(len(y))):
                mu, sd = P[a].mean(0), P[a].std(0) + 1e-6
                pr[b] = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit((P[a] - mu) / sd, y[a]).predict((P[b] - mu) / sd)
            for gg in ("m", "f"):
                mm = gp == gg
                out[f"face_{t}_r_within_{gg}"] = float(np.corrcoef(pr[mm], y[mm])[0, 1])
        # Clip-level, identity-disjoint gender / language probes on the bilingual people.
        Xb, yb_g, yb_l, idb = X[rows_x], gender[rows_x], (lang[rows_x] > 0).astype(int), ident[rows_x]
        for t, y in (("gender", yb_g), ("language", yb_l)):
            accs = []
            for a, b in GroupKFold(5).split(Xb, y, idb):
                c = LogisticRegression(max_iter=2000).fit(Xb[a], y[a]).predict(Xb[b])
                accs.append(np.mean([np.mean(c[y[b] == k] == k) for k in np.unique(y[b])]))
            out[f"{t}_balanced_acc_unseen_people"] = float(np.mean(accs))
        probes[name] = out
        print(name, {k: round(v, 3) for k, v in out.items()}, flush=True)
    rep["probes"] = probes
    rep["note"] = ("fit set = official + VoxCeleb2 people; evaluation = MAV-Celeb v1/v3 people (unseen by all "
                   "ridge maps); CKA = representational similarity, not task redundancy")
    (OUT / "complementarity.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("cka", "redundancy_r2_unseen_people", "same_speaker_language_shift")}, indent=1))


if __name__ == "__main__":
    main()
