#!/usr/bin/env python3
"""Publication figures generated from public_results/experiments.json (SVG + PNG in figures/).

No value is typed here: every bar/point is looked up in the registry by experiment id.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REG = json.loads((ROOT / "public_results/experiments.json").read_text())
E = {e["id"]: e for e in REG["experiments"]}
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
C = {"teal": "#14b8a6", "violet": "#8b5cf6", "amber": "#f59e0b", "rose": "#f43f5e", "grey": "#64748b", "sky": "#0ea5e9"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "figure.dpi": 150, "svg.fonttype": "none"})


def sysm(i, name, key, field="systems"):
    return E[i][field][name][key]


def save(fig, name, source_ids):
    fig.text(0.01, 0.005, "Source: public_results/experiments.json · " + ", ".join(source_ids), fontsize=6.5, color="#64748b")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for ext in ("svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def progression():
    steps = [("V2", "v2_reference", "V2", "age_gender_terms_v2", "V2 + corrected terms"),
             ("+ external data", "ext_all_share50", "+ v1 + v3 + VoxCeleb2 (50% share) = eval #3 model", "eval3_terms", "V2 + all external (eval #3) + terms"),
             ("B0", "b0_full", "B0 = + eGeMAPS + traits + language", "b0_terms", "B0 + terms"),
             ("C1 (+ ECAPA)", "g3_cat_farl", "C1 = B0 voice + ECAPA (concat) + FaRL", "c1_terms", "C1 + terms"),
             ("C1 × 3", "seed_ensemble_official", "C1 x 3", "final_c1x3_terms", "C1 x 3 (raw-mean ensemble) + terms")]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(steps))
    raw = [sysm(i, n, "mean") for _, i, n, _, _ in steps]
    terms = [sysm(i, n, "mean") for _, _, _, i, n in steps]
    ax.plot(x, raw, "o-", color=C["violet"], label="cosine only")
    ax.plot(x, terms, "s-", color=C["teal"], label="+ fixed age/gender terms")
    for xi, (r, t) in enumerate(zip(raw, terms)):
        ax.annotate(f"{t:.2f}", (xi, t), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8, color=C["teal"])
        ax.annotate(f"{r:.2f}", (xi, r), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8, color=C["violet"])
    ax.set_xticks(x, [s[0] for s in steps])
    ax.set_ylim(min(terms) - 0.35, max(raw) + 0.3)
    ax.set_ylabel("Local mean EER (%) ↓")
    ax.set_title("Baseline-to-final progression (local 5-fold, 3 seeds; not official scores)")
    ax.legend(frameon=False)
    save(fig, "fig_progression", [s[1] for s in steps] + [s[3] for s in steps])


def encoder_2x2():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    uni = E["p2_audio_unimodal"]["unimodal"]
    names = [("WavLM + eGeMAPS (B0 input), LDA", "WavLM+eGeMAPS"), ("WavLM-SV x-vector (per clip), LDA", "x-vector"), ("ECAPA-TDNN, centred", "ECAPA")]
    vals = [uni[n]["same_language_other_video"] for n, _ in names]
    face = E["p2_face_unimodal"]["unimodal"]
    fnames = [("farl", "FaRL"), ("arcface_insightface", "ArcFace"), ("dinov2_l12_patch", "DINOv2")]
    fvals = [face[n]["same_gender"] for n, _ in fnames]
    ax = axes[0]
    ax.bar([s for _, s in names] + [s for _, s in fnames], vals + fvals, color=[C["violet"]] * 3 + [C["teal"]] * 3)
    ax.set_ylabel("Unimodal same-gender EER (%) ↓")
    ax.set_title("Unimodal identity (unseen people)")
    ax.tick_params(axis="x", rotation=30)
    ax = axes[1]
    rows = [("B0 (WavLM+eGeMAPS / FaRL)", 0.0, None), ("ECAPA / FaRL", E["g3_ecapa_farl"]["comparison"]["delta"]["mean"], None),
            ("B0 voice / ArcFace", E["g3_b0_arcface"]["comparison"]["delta"]["mean"], None), ("ECAPA / ArcFace", E["g3_ecapa_arcface"]["comparison"]["delta"]["mean"], None),
            ("B0 voice + ECAPA / FaRL (C1)", E["g3_cat_farl"]["comparison"]["delta"]["mean"], None)]
    y = np.arange(len(rows))[::-1]
    for yi, (lbl, d, _) in zip(y, rows):
        if isinstance(d, dict):
            v, ci = d["value"], d["ci95"]
            ax.errorbar(v, yi, xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=C["rose"] if ci[0] > 0 else C["grey"], capsize=3)
        else:
            ax.plot(0, yi, "o", color=C["grey"])
    ax.axvline(0, color="#334155", lw=1)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Cross-modal mean EER vs B0 (points, 95% CI) ↓")
    ax.set_title("Cross-modal (official folds)")
    save(fig, "fig_unimodal_vs_crossmodal", ["p2_audio_unimodal", "p2_face_unimodal", "g3_ecapa_farl", "g3_b0_arcface", "g3_ecapa_arcface", "g3_cat_farl"])


def learning_curve():
    pts = E["learning_curve"]["points"]
    fig, ax = plt.subplots(figsize=(5, 3.4))
    n = [p["identities"] for p in pts]
    for k, col in (("standard", C["teal"]), ("gender", C["violet"])):
        ax.plot(n, [p[k] for p in pts], "o-", color=col, label="standard" if k == "standard" else "same gender")
    ax.set_xscale("log")
    ax.set_xticks(n, [str(v) for v in n])
    ax.set_xlabel("Training identities per fold (log scale)")
    ax.set_ylabel("EER (%) ↓")
    ax.set_title("Learning curve (V2, 3 seeds × 5 folds)")
    ax.legend(frameon=False)
    save(fig, "fig_learning_curve", ["learning_curve"])


def gamp():
    r = E["gamp_pseudo_labels"]["recovery"]
    items = [("Pairwise top-1\n(learned groups)", r["pairwise_top1_precision"]), ("Hungarian\n(learned groups)", r["hungarian_precision"]),
             ("Top-1\n(oracle groups)", r["oracle_groups_top1"]), ("Hungarian\n(oracle groups)", r["oracle_groups_hungarian"]),
             ("Most confident 10%\n(oracle groups)", r["oracle_groups_precision_top10pct"])]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.bar([i[0] for i in items], [100 * i[1] for i in items], color=[C["grey"], C["grey"], C["violet"], C["violet"], C["violet"]])
    ax.axhline(100 * r["target_precision"], color=C["rose"], ls="--", lw=1.2)
    ax.text(4.4, 100 * r["target_precision"] + 1.5, "precision needed for training labels", color=C["rose"], ha="right", fontsize=8)
    ax.set_ylabel("Correct associations (%) ↑")
    ax.set_ylim(0, 100)
    ax.set_title("GAMP-VF: pseudo-label recovery on 77 hidden identities")
    save(fig, "fig_gamp_pseudo_labels", ["gamp_pseudo_labels"])


def unseen_language():
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=False)
    for ax, (eid, lang, field) in zip(axes, (("seed_ensembles_holdouts", "Urdu", "systems_urdu"), ("seed_ensembles_holdouts", "German", "systems_german"))):
        tab = E[eid][field]
        names = ["B0", "C1", "C1 x 3"]
        keys = [("standard_English", "std English"), (f"standard_{lang}", f"std {lang}"), ("gender_English", "same-gender English"), (f"gender_{lang}", f"same-gender {lang}")]
        x = np.arange(len(keys))
        w = 0.26
        for k, (nm, col) in enumerate(zip(names, (C["grey"], C["violet"], C["teal"]))):
            ax.bar(x + (k - 1) * w, [tab[nm][kk] for kk, _ in keys], w, color=col, label=nm.replace("x", "×"))
        ax.set_xticks(x, [k[1] for k in keys], rotation=20)
        ax.set_ylim(15, None)
        ax.set_ylabel("EER (%) ↓")
        ax.set_title(f"{lang} holdout (unseen people + language)")
    axes[0].legend(frameon=False, fontsize=8)
    save(fig, "fig_unseen_language", ["seed_ensembles_holdouts", "urdu_holdout_phase2", "german_holdout"])


def ensemble():
    e = E["seed_ensemble_official"]["systems"]
    names = ["C1", "C1 x 2", "C1 x 3"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.bar([n.replace("x", "×") for n in names], [e[n]["mean"] for n in names], color=[C["violet"], C["sky"], C["teal"]])
    for i, n in enumerate(names):
        ax.text(i, e[n]["mean"] + 0.03, f"{e[n]['mean']:.2f}", ha="center", fontsize=8)
    lo = min(e[n]["mean"] for n in names)
    ax.set_ylim(lo - 1.0, lo + 1.2)
    d = E["seed_ensemble_official"]["comparison"]["delta"]["mean"]
    ax.set_title(f"Seed ensemble (official folds): C1×3 − C1 = {d['value']:+.2f} [{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]", fontsize=9)
    ax.set_ylabel("Local mean EER (%) ↓")
    save(fig, "fig_seed_ensemble", ["seed_ensemble_official"])


def same_gender_scores():
    h = REG["score_histograms"]["tracks"]["gender"]
    b = np.array(h["bins"])
    mid = (b[1:] + b[:-1]) / 2
    g, i = np.array(h["genuine"], float), np.array(h["impostor"], float)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.step(mid, g / g.sum(), where="mid", color=C["teal"], label="same person")
    ax.step(mid, i / i.sum(), where="mid", color=C["rose"], label="same-gender impostor")
    ax.fill_between(mid, g / g.sum(), step="mid", color=C["teal"], alpha=0.15)
    ax.fill_between(mid, i / i.sum(), step="mid", color=C["rose"], alpha=0.12)
    ax.set_xlim(-0.2, 1)
    ax.set_xlabel("Cosine score")
    ax.set_ylabel("Share of trials")
    ax.set_title(f"B0 same-gender trials overlap heavily (EER {h['eer']:.2f}%)")
    ax.legend(frameon=False)
    save(fig, "fig_same_gender_scores", ["score_histograms"])


def ablation_forest():
    rows = [(i, E[i]["title"]) for i in ("aug25", "hyperbolic_t010", "layer_attention", "ecapa_only_v2", "wavlm_ecapa_v2", "ext_all_share50",
                                         "b0_full", "gamp_b3_sg", "g3_b0_arcface", "g3_ecapa_farl", "g3_cat_farl", "pj_linear", "ls_multipos", "p3_bl2", "p3_k64")]
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    y = np.arange(len(rows))[::-1]
    for yi, (i, t) in zip(y, rows):
        d = E[i]["comparison"]["delta"]["mean"]
        v, ci = d["value"], d["ci95"]
        col = C["teal"] if ci[1] < 0 else C["rose"] if ci[0] > 0 else C["grey"]
        ax.errorbar(v, yi, xerr=[[v - ci[0]], [ci[1] - v]], fmt="o", color=col, capsize=2.5)
        # baseline label in a column just right of the axes (axes-x, data-y coordinates)
        ax.text(1.01, yi, f"vs {E[i]['comparison']['a']}", va="center", fontsize=6.5, color="#64748b",
                transform=ax.get_yaxis_transform())
    ax.axvline(0, color="#334155", lw=1)
    ax.set_yticks(y, [f"{t[:46]}{'…' if len(t) > 46 else ''}" + (" (seed 1)" if E[i].get("seeds") == [1] else "") for i, t in rows], fontsize=8)
    ax.set_xlabel("Mean EER difference vs its baseline (points, 95% CI); negative = better")
    ax.set_xlim(-4.5, 13.5)
    ax.set_title("Ablation summary (local 5-fold, paired identity bootstrap)")
    save(fig, "fig_ablation_forest", [i for i, _ in rows])


if __name__ == "__main__":
    for f in (progression, encoder_2x2, learning_curve, gamp, unseen_language, ensemble, same_gender_scores, ablation_forest):
        f()
    print("figures:", sorted(p.name for p in OUT.iterdir()))
