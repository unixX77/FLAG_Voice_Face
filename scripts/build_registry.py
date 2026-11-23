#!/usr/bin/env python3
"""Build the canonical experiment registry (public_results/experiments.json) from the research tree.

Every number is read from a saved result file of the research project, never typed by hand,
except for entries whose only record is the progress log; those carry source_type
"progress_log" with exact line ranges in docs/research/progress_log.md (a verbatim copy).
Every cited JSON source is copied into public_results/sources/ (same relative path), so the
public repository can re-verify the registry (tests/test_registry.py).

Usage: python scripts/build_registry.py --research-root /path/to/Face-to-Voice
EER values are percentages (lower is better). Deltas are B - A in EER points.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PCT = 100.0
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "public_results"
LOG = "docs/research/progress_log.md"

PROTOCOLS = {
    "codabench_dev": "Official CodaBench dev (progress) phase: English + Bangla (unheard) cells, mean of 4 cells.",
    "codabench_eval": "Official CodaBench evaluation phase 1: English (heard) + Bangla (unheard), mean of 4 cells.",
    "local_5fold": "Local identity-disjoint 5-fold cross-validation on the 70 official (English) training identities; fixed trial files; 3 seeds x 5 folds x 90 epochs unless 'seeds' says otherwise; paired identity-level bootstrap.",
    "local_5fold_terms": "As local_5fold, scores include the fixed age/gender terms; term weights chosen on the same folds (small in-sample optimism).",
    "urdu_holdout": "Unseen-language proxy: 1/5 of MAV-Celeb v1 people held out per fold, no Urdu in training; per-seed scoring (phase2/language_transfer_fusion.py).",
    "urdu_holdout_seedavg": "Same Urdu holdout, but scores averaged over the 3 seeds before EER (local_validation/evaluate_language_transfer.py); not directly comparable with urdu_holdout.",
    "german_holdout": "Unseen-language confirmation: 1/5 of MAV-Celeb v3 people held out per fold, no German in training; per-seed scoring; never used for model selection.",
    "unimodal_diagnostic": "Voice-only or face-only verification/retrieval on unseen people (deterministic trials; Phase II).",
    "simulation": "Hidden-association simulation (GAMP-VF): labels of 77 external training people hidden; association recovery measured against the hidden truth.",
}


class Builder:
    def __init__(self, research: Path):
        self.research = research
        self.entries: list[dict] = []
        self.copied: set[str] = set()

    def load(self, rel: str) -> dict:
        src = self.research / rel
        dst = OUT / "sources" / rel
        if rel not in self.copied:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            self.copied.add(rel)
        return json.loads(src.read_text())

    def add(self, **e) -> None:
        required = ("id", "phase", "family", "title", "question", "protocol", "outcome", "interpretation")
        missing = [k for k in required if k not in e]
        if missing:
            raise ValueError(f"{e.get('id')}: missing {missing}")
        if e["protocol"] not in PROTOCOLS:
            raise ValueError(f"{e['id']}: unknown protocol {e['protocol']}")
        if e["outcome"] not in ("adopted", "rejected", "diagnostic", "candidate", "baseline", "official"):
            raise ValueError(f"{e['id']}: bad outcome")
        if any(x["id"] == e["id"] for x in self.entries):
            raise ValueError(f"duplicate id {e['id']}")
        self.entries.append(e)

    # ---- helpers ---------------------------------------------------------------------------
    def compare(self, rel: str, a: str, b: str) -> dict:
        """Paired comparison JSON from local_validation/compare_fold_runs.py."""
        d = self.load(rel)
        out = {"systems": {a: {t: round(d[t]["a"] * PCT, 2) for t in ("standard", "gender", "mean")},
                           b: {t: round(d[t]["b"] * PCT, 2) for t in ("standard", "gender", "mean")}},
               "comparison": {"a": a, "b": b, "seeds": d["seeds"],
                              "delta": {t: {"value": round(d[t]["b_minus_a"] * PCT, 2),
                                            "ci95": [round(x * PCT, 2) for x in d[t]["ci95"]],
                                            "p_b_better": d[t]["p_b_better"]} for t in ("standard", "gender", "mean")}},
               "sources": [{"path": f"public_results/sources/{rel}", "type": "json",
                            "extract": "compare_fold_runs: standard/gender/mean.{a,b,b_minus_a,ci95,p_b_better}"}]}
        if "per_fold_eer" in d:
            out["per_fold"] = {lbl: {t: [round(x * PCT, 2) for x in d["per_fold_eer"][k][t]] for t in ("standard", "gender")}
                               for k, lbl in (("a", a), ("b", b))}
        return out

    def terms(self, rel: str, label: str) -> dict:
        d = self.load(rel)["weights"]
        s, g = d["no_gender"], d["gender"]
        # Means are computed from the unrounded track values, then rounded once.
        sys_ = {"standard": round(s["cv_mean_eer"] * PCT, 2), "gender": round(g["cv_mean_eer"] * PCT, 2),
                "mean": round((s["cv_mean_eer"] + g["cv_mean_eer"]) / 2 * PCT, 2)}
        raw = {"standard": round(s["cv_mean_eer_without_terms"] * PCT, 2), "gender": round(g["cv_mean_eer_without_terms"] * PCT, 2),
               "mean": round((s["cv_mean_eer_without_terms"] + g["cv_mean_eer_without_terms"]) / 2 * PCT, 2)}
        return {"systems": {f"{label} + terms": sys_, f"{label} (terms fitter, no terms)": raw},
                "term_weights": {"standard_track": {"w_age_per_10y": s["w_age_per_10_years"], "w_gender": s["w_gender"]},
                                 "gender_track": {"w_age_per_10y": g["w_age_per_10_years"], "w_gender": g["w_gender"]}},
                "sources": [{"path": f"public_results/sources/{rel}", "type": "json",
                             "extract": "weights.{no_gender,gender}.cv_mean_eer[_without_terms]"}]}

    def transfer(self, rel: str, names: dict[str, str], lang: str, ref_label: str) -> dict:
        """language_transfer_fusion.py output: per-seed point estimates + paired deltas vs its reference."""
        d = self.load(rel)
        systems = {}
        for key, label in names.items():
            p = d["point"][key]
            systems[label] = {f"standard_{c}": round(p[f"standard_{c}"] * PCT, 2) for c in ("English", lang)} | \
                             {f"gender_{c}": round(p[f"gender_{c}"] * PCT, 2) for c in ("English", lang)}
        deltas = {}
        for key, label in names.items():
            if key in d.get("vs_b0", {}):
                deltas[f"{label} vs {ref_label}"] = {m: {"value": round(v["mean"] * PCT, 2),
                                                         "ci95": [round(x * PCT, 2) for x in v["ci95"]],
                                                         "p_b_better": round(v["p_better"], 3)}
                                                     for m, v in d["vs_b0"][key].items() if not m.endswith("_gap")}
        return {"systems": systems, "deltas": deltas, "languages": ["English", lang],
                "sources": [{"path": f"public_results/sources/{rel}", "type": "json", "extract": "point.*, vs_b0.*"}]}

    @staticmethod
    def log(lines: str, what: str) -> list[dict]:
        return [{"path": LOG, "type": "progress_log", "lines": lines, "extract": what}]


def build(b: Builder) -> None:
    # ================= Stage 1-2: baselines and V2 =================
    b.add(id="codabench_dev_v2", phase="1-2 Baseline and V2", family="Official results",
          title="V2 on the CodaBench dev (progress) leaderboard", question="How does the improved V2 system score officially?",
          protocol="codabench_dev", outcome="official", languages=["English", "Bangla"],
          systems={"FOP baseline (organisers)": {"mean": 36.92}, "V2": {"mean": 29.11}},
          cells={"V2": {"standard_English": 23.61, "standard_Bangla": 23.19, "gender_English": 35.85, "gender_Bangla": 33.79},
                 "FOP baseline (organisers)": {"standard_English": 32.54, "standard_Bangla": 38.12, "gender_English": 32.99, "gender_Bangla": 44.01}},
          interpretation="V2 is 7.8 points better than the official baseline on the dev leaderboard.",
          sources=[{"path": "docs/research/codabench/codabench_dev_v2.txt", "type": "official_result"}] + b.log("57-63", "baseline cells"))
    b.add(id="codabench_eval_submissions", phase="1-2 Baseline and V2", family="Official results",
          title="Evaluation-phase submissions #1-#3", question="Official scores of the submitted systems",
          protocol="codabench_eval", outcome="official", languages=["English", "Bangla"],
          systems={"FOP baseline (organisers)": {"mean": 35.32}, "Eval #1: V2": {"mean": 31.28},
                   "Eval #2: V2 + age/gender terms": {"mean": 30.17}, "Eval #3: V2 + external data + terms": {"mean": 28.84}},
          cells={"Eval #1: V2": {"standard_English": 26.53, "standard_Bangla": 25.19, "gender_English": 36.54, "gender_Bangla": 36.87},
                 "Eval #2: V2 + age/gender terms": {"standard_English": 24.49, "standard_Bangla": 22.69, "gender_English": 36.06, "gender_Bangla": 37.43},
                 "Eval #3: V2 + external data + terms": {"standard_English": 22.96, "standard_Bangla": 21.20, "gender_English": 34.62, "gender_Bangla": 36.59}},
          context={"leaderboard_13_nov": {"entries": 34, "median": 22.47, "top": 14.30, "rank_of_eval2": "32 of 34"}},
          interpretation="Local predictions transferred closely (eval #3: local -1.34, CodaBench -1.33). The system remains well behind the leaderboard median (22.47 on 13 Nov). Eval #4-#6 candidates have no submission record; the final challenge ranking is not available.",
          sources=[{"path": "docs/research/codabench/codabench_eval1_v2.txt", "type": "official_result"},
                   {"path": "docs/research/codabench/codabench_eval2_v2_terms.txt", "type": "official_result"}] +
                  b.log("64-71", "baseline + leaderboard context") + b.log("975-987", "eval #3 cells"))
    b.add(id="v1_to_v2", phase="1-2 Baseline and V2", family="Representation", title="V1 -> V2: all-layer WavLM, SupCon, regularised heads",
          question="Do broader WavLM features and a supervised contrastive objective beat the V1 x-vector/ArcFace design?",
          hypothesis="Speaker-verification x-vectors discard traits shared with faces; all-layer WavLM keeps them.",
          protocol="local_5fold", outcome="adopted", seeds=[1],
          systems={"V1 (FaRL + WavLM x-vector, ArcFace heads), best epoch": {"standard": 25.9, "gender": 37.5, "mean": 31.7},
                   "V2, best epoch": {"standard": 22.4, "gender": 33.5, "mean": 28.0},
                   "V2, fixed epoch 90": {"standard": 23.1, "gender": 34.4, "mean": 28.8}},
          comparison={"a": "V1", "b": "V2", "delta": {"standard": {"value": -2.9, "ci95": [-5.5, -0.2]}, "gender": {"value": -3.9, "ci95": [-7.3, -0.5]}}},
          interpretation="Both tracks improve significantly; V2 became the base system.",
          sources=b.log("203-206", "V1/V2 five-fold numbers and paired CI"))
    e = b.compare("runs/voice_aug/compare_base_vs_aug25.json", "V2", "V2 + voice augmentation (clean 25%)")
    b.add(id="v2_reference", phase="1-2 Baseline and V2", family="Baseline", title="V2 reference (3 seeds x 5 folds, epoch 90)",
          question="Reference numbers for every later paired comparison", protocol="local_5fold", outcome="baseline", seeds=[1, 2, 3],
          systems={"V2": e["systems"]["V2"]}, interpretation="All Stage 3-4 comparisons are paired against these runs (runs/voice_aug/base).",
          sources=e["sources"])
    # ================= Stage 4: exploration on 70 identities =================
    for rel, label, eid, title, hyp, outcome, interp in (
        ("runs/voice_aug/compare_base_vs_aug50.json", "Voice augmentation (clean 50%)", "aug50", "Acoustic augmentation (50% clean)",
         "Noise/reverb/codec augmentation regularises the voice branch.", "rejected", "Worse on both tracks; augmented views move features far from their clip."),
        ("runs/voice_aug/compare_base_vs_aug25.json", "Voice augmentation (clean 25%)", "aug25", "Acoustic augmentation (25% clean)",
         "As above, stronger.", "rejected", "Significantly worse (mean CI excludes 0)."),
        ("runs/hyperbolic/compare_base_vs_hyp_t010.json", "Hyperbolic, tau 0.1", "hyperbolic_t010", "Hyperbolic (Poincare) embeddings, tau = 0.1",
         "A hierarchical geometry separates same-gender people better.", "rejected", "Standard track -1.0 (significant), gender track unchanged; overlapped with the age/gender terms, so not carried forward."),
        ("runs/hyperbolic/compare_base_vs_hyp_t020.json", "Hyperbolic, tau 0.2", "hyperbolic_t020", "Hyperbolic (Poincare) embeddings, tau = 0.2",
         "As above.", "rejected", "Standard -1.3 (significant), gender +0.25."),
        ("runs/layer_attention/compare_base_vs_mix_mean.json", "Learned layer mix", "layer_mix", "WavLM learned layer weights",
         "Learned layer weights beat the plain all-layer mean.", "rejected", "No change; learned weights mildly prefer lower layers."),
        ("runs/layer_attention/compare_base_vs_mix_meanstd.json", "Layer mix + temporal std", "layer_mix_std", "WavLM layer mix with temporal spread",
         "Adding each layer's temporal spread adds speaker cues.", "rejected", "Gender worse (CI excludes 0)."),
        ("runs/layer_attention/compare_base_vs_attn_meanstd.json", "Per-clip layer attention", "layer_attention", "Per-clip WavLM layer attention",
         "Per-clip attention over layers selects useful layers.", "rejected", "Significantly worse on both tracks (overfits with 70 identities)."),
        ("runs/speaker_embedding/compare_base_vs_ecapa_only.json", "ECAPA only (70 ids)", "ecapa_only_v2", "ECAPA-TDNN instead of WavLM (70 identities)",
         "A dedicated speaker-verification embedding carries finer person-specific cues.", "rejected", "+3.2 mean: speaker embeddings alone lose the traits that link voice to face."),
        ("runs/speaker_embedding/compare_base_vs_wavlm_ecapa.json", "WavLM + ECAPA (70 ids)", "wavlm_ecapa_v2", "WavLM + ECAPA concatenation (70 identities)",
         "ECAPA complements WavLM.", "rejected", "No gain with 70 identities (seed-1 gain vanished with seeds 2-3). Revisited in Phase II with 301 identities."),
    ):
        e = b.compare(rel, "V2", label)
        b.add(id=eid, phase="4 Architecture exploration", family="Representation" if "ecapa" in eid or "layer" in eid else "Training",
              title=title, question=f"Does '{label}' improve on V2?", hypothesis=hyp, protocol="local_5fold", outcome=outcome,
              seeds=e["comparison"]["seeds"], interpretation=interp, **{k: v for k, v in e.items()})
    d = b.load("features/voice_v7_agegender/age_consistency_B_corrected.json")
    b.add(id="age_gender_terms_v2", phase="4 Architecture exploration", family="Scoring",
          title="Bias-corrected age/gender consistency terms (V2)", question="Do fixed age/gender-consistency score terms help?",
          hypothesis="V2 under-uses age: impostors 20+ years apart are still accepted 20% of the time.",
          protocol="local_5fold", outcome="adopted", seeds=[1, 2, 3],
          systems={"V2": {t: round(d[t]["base"] * PCT, 2) for t in ("standard", "gender", "mean")},
                   "V2 + corrected terms": {t: round(d[t]["with_terms"] * PCT, 2) for t in ("standard", "gender", "mean")}},
          comparison={"a": "V2", "b": "V2 + corrected terms",
                      "delta": {t: {"value": round(d[t]["diff"] * PCT, 2), "ci95": [round(x * PCT, 2) for x in d[t]["ci95"]],
                                    "p_b_better": d[t]["p_better"]} for t in ("standard", "gender", "mean")}},
          interpretation="Significant -0.8 mean, almost all on the standard track; submitted as eval #2 (CodaBench -1.1).",
          sources=[{"path": "public_results/sources/features/voice_v7_agegender/age_consistency_B_corrected.json", "type": "json"}])
    b.add(id="learning_curve", phase="4 Architecture exploration", family="Data", title="Learning curve over training identities",
          question="How much does EER fall with more training identities?", protocol="local_5fold", outcome="diagnostic", seeds=[1, 2, 3],
          points=[{"identities": 20, "standard": 25.09, "gender": 37.55, "mean": 31.32}, {"identities": 28, "standard": 24.55, "gender": 35.86, "mean": 30.20},
                  {"identities": 42, "standard": 23.87, "gender": 35.00, "mean": 29.43}, {"identities": 56, "standard": 22.86, "gender": 34.25, "mean": 28.55}],
          interpretation="About -1.6 EER points per doubling of identities on both tracks; motivated the external-data stage.",
          sources=b.log("657-668", "learning-curve table (15 runs per size)"))
    b.add(id="face_backbones", phase="4 Architecture exploration", family="Representation", title="Alternative face backbones (V2 era)",
          question="Do sharper face features help?", protocol="local_5fold", outcome="rejected",
          interpretation="FaRL variants and FaRL + DINOv2 matched FaRL within noise; DINOv2 alone was worse. No per-variant numbers were logged.",
          sources=b.log("208-210", "qualitative result"))
    b.add(id="losses_misc", phase="4 Architecture exploration", family="Training", title="Same-gender SupCon, hard negatives, prototype loss, OPL, GADH (V2 era)",
          question="Do alternative objectives help the gender track?", protocol="local_5fold", outcome="rejected",
          interpretation="No gains (same-gender SupCon: +1 point on standard). No per-variant numbers were logged.",
          sources=b.log("212-213", "qualitative result"))
    b.add(id="pairwise_verifier", phase="4 Architecture exploration", family="Architecture", title="Learned pairwise verifier",
          question="Does a learned matching head beat cosine?", protocol="local_5fold", outcome="rejected",
          systems={"Cosine": {"standard": 28.9, "gender": 39.3, "mean": 34.1}, "Verifier alone": {"standard": 33.3, "gender": 39.5, "mean": 36.4},
                   "Best blend": {"standard": 28.6, "gender": 38.7, "mean": 33.7}},
          interpretation="Training identities are almost perfectly separated, so the head learns an easy problem unlike unseen people. Numbers are from an earlier, different local setup and are not comparable with the 3-seed tables.",
          sources=b.log("215-224", "verifier table"))
    # ================= Stage 3: data expansion =================
    for rel, label, eid, interp in (
        ("runs/external/compare_base_vs_v3_natural.json", "+ MAV-Celeb v3 (natural sampling)", "ext_v3_natural", "Standard improves; gender unchanged."),
        ("runs/external/compare_base_vs_v3_share50.json", "+ MAV-Celeb v3 (50% official share)", "ext_v3_share50", "As natural sampling."),
        ("runs/external/compare_base_vs_v3vox2_natural.json", "+ v3 + VoxCeleb2 test (natural)", "ext_v3vox2_natural", "Larger gain; wide intervals."),
        ("runs/external/compare_base_vs_v3vox2_share50.json", "+ v3 + VoxCeleb2 test (50% share)", "ext_v3vox2_share50", "Standard significant."),
        ("runs/external/compare_base_vs_all_natural.json", "+ v1 + v3 + VoxCeleb2 (natural)", "ext_all_natural", "Similar to share50, wider CI."),
        ("runs/external/compare_base_vs_all_share50.json", "+ v1 + v3 + VoxCeleb2 (50% share) = eval #3 model", "ext_all_share50",
         "Mean -1.79, significant; adopted (eval #3, CodaBench 28.84)."),
    ):
        e = b.compare(rel, "V2", label)
        b.add(id=eid, phase="3 Data expansion", family="Data", title=label, question="Do external training identities help?",
              hypothesis="More identities reduce EER (learning curve).", protocol="local_5fold",
              outcome="adopted" if eid == "ext_all_share50" else "diagnostic",  # intermediate ablation steps
              seeds=e["comparison"]["seeds"], interpretation=interp, **e)
    e = b.terms("runs/external/all_share50_attribute_terms.json", "V2 + all external (eval #3)")
    b.add(id="eval3_terms", phase="3 Data expansion", family="Scoring", title="Eval #3 model with refitted terms",
          question="Local estimate of the submitted eval #3 system", protocol="local_5fold_terms", outcome="official", seeds=[1, 2, 3],
          interpretation="Local mean with terms; CodaBench eval #3 = 28.84.", **e)
    # ================= B0: traits / eGeMAPS / language =================
    for rel, label, eid, outcome, interp in (
        ("runs/traits/compare_all_share50_vs_lang.json", "+ language adversary + cross-lingual", "b0_lang", "rejected", "No change on its own."),
        ("runs/traits/compare_all_share50_vs_traits.json", "+ shared trait heads (0.2)", "b0_traits", "rejected", "Tiny gain on its own."),
        ("runs/traits/compare_all_share50_vs_traits05.json", "+ shared trait heads (0.5)", "b0_traits05", "rejected", "Tiny gain on its own."),
        ("runs/traits/compare_all_share50_vs_egemaps.json", "+ eGeMAPS voice features", "b0_egemaps", "rejected", "Small, not significant."),
        ("runs/traits/compare_all_share50_vs_full.json", "B0 = + eGeMAPS + traits + language", "b0_full", "adopted", "Best combination (-0.57 n.s.); becomes B0."),
    ):
        e = b.compare(rel, "Eval #3 model (all_share50)", label)
        b.add(id=eid, phase="3 Data expansion", family="Architecture", title=label, question="Do auxiliary voice->traits->face components help?",
              protocol="local_5fold", outcome=outcome, seeds=e["comparison"]["seeds"], interpretation=interp, **e)
    e = b.terms("runs/traits/full_attribute_terms.json", "B0")
    b.add(id="b0_terms", phase="3 Data expansion", family="Scoring", title="B0 with refitted terms (eval #4 zip, not submitted)",
          question="Local estimate of B0 + terms", protocol="local_5fold_terms", outcome="baseline", seeds=[1, 2, 3],
          interpretation="B0 + terms = 25.94 local mean. The eval #4 zip has no submission record.", **e)
    # ================= Stage 5: GAMP-VF =================
    for rel, label, eid, interp in (
        ("results/gamp_vf/stage_C_screening/compare_b0_vs_b1_proto_seed1.json", "B1 prototype InfoNCE", "gamp_b1", "Worse (seed 1)."),
        ("results/gamp_vf/stage_C_screening/compare_b0_vs_b2_protobal_seed1.json", "B2 language-balanced prototypes", "gamp_b2", "No gain (seed 1)."),
        ("results/gamp_vf/stage_C_screening/compare_b0_vs_b3_nn_seed1.json", "B3 most-confusable negatives", "gamp_b3_nn", "No gain (seed 1)."),
        ("results/gamp_vf/stage_C_screening/compare_b0_vs_b3_sgn100_seed1.json", "B3 same-gender nearest, ratio 1.0", "gamp_b3_sgn100", "Worse (seed 1)."),
        ("results/gamp_vf/stage_C_final/compare_b0_vs_b3_sg_3seeds.json", "B3 same-gender partner sampling", "gamp_b3_sg", "Gender -0.41 n.s. (5/5 folds); English-only (Urdu proxy: no gain)."),
        ("results/gamp_vf/stage_C_final/compare_b0_vs_b3_sga_3seeds.json", "B3 same-gender similar-age partner", "gamp_b3_sga", "No change."),
    ):
        e = b.compare(rel, "B0", label)
        b.add(id=eid, phase="5 GAMP-VF", family="GAMP-VF", title=label, question="Do GAMP-VF prototypes / hard negatives help?",
              protocol="local_5fold", outcome="rejected", seeds=e["comparison"]["seeds"], interpretation=interp, **e)
    sims = [b.load(f"results/gamp_vf/stage_E/it1/fold_{k}/summary.json") for k in range(5)]
    def avg(path):
        vals = []
        for s in sims:
            x = s
            for p in path:
                x = x[p]
            vals.append(x)
        vals = [v for v in vals if v == v]
        return round(sum(vals) / len(vals), 3) if vals else None
    orc = b.load("results/gamp_vf/stage_F/oracle_group_matching.json")["mean"]["hidden"]
    b.add(id="gamp_pseudo_labels", phase="5 GAMP-VF", family="GAMP-VF", title="Pseudo-label recovery on 77 hidden identities",
          question="Can clustering + matching recover hidden face-voice associations reliably enough to train on?",
          hypothesis="Speaker clusters + cross-modal matching yield high-precision pseudo-pairs.", protocol="simulation", outcome="rejected",
          recovery={"voice_cluster_ari_hidden": avg(["clustering", "H_voice", "ari"]), "face_cluster_ari_hidden": avg(["clustering", "H_face", "ari"]),
                    "pairwise_top1_precision": avg(["recovery", "H", "pairwise_top1", "pair_precision"]),
                    "hungarian_precision": avg(["recovery", "H", "hungarian", "pair_precision"]),
                    "oracle_groups_top1": round(orc["top1_any_gender"], 3), "oracle_groups_hungarian": round(orc["hungarian_any_gender"], 3),
                    "oracle_groups_precision_top10pct": round(orc["precision_top10pct_margin_any_gender"], 3), "target_precision": 0.9},
          interpretation="Even with perfect grouping only 24% top-1 / 34% Hungarian; the 10% most confident matches are 57% correct. No mode reached usable precision.",
          sources=[{"path": f"public_results/sources/results/gamp_vf/stage_E/it1/fold_{k}/summary.json", "type": "json"} for k in range(5)] +
                  [{"path": "public_results/sources/results/gamp_vf/stage_F/oracle_group_matching.json", "type": "json"}])
    e = b.compare("results/gamp_vf/stage_G/compare_teacher_vs_oracle_seed1.json", "Teacher (77 people hidden)", "Oracle (true labels)")
    b.add(id="gamp_oracle_headroom", phase="5 GAMP-VF", family="GAMP-VF", title="Headroom: true labels of the 77 hidden people",
          question="Would perfect pseudo-labels help?", protocol="local_5fold", outcome="diagnostic", seeds=[1],
          interpretation="No measurable gain (seed 1), so the simulation cannot show a pseudo-label benefit.", **e)
    # ================= Stage 6: Phase II =================
    a1 = b.load("results/phase2/phase1_audio/metrics.json")["bilingual"]
    a2 = b.load("results/phase2/phase1_audio/metrics_speaker_encoders.json")["bilingual"]
    def uni(m):
        mm = m["metrics"]
        return {"same_language_other_video": round(mm["same_lang_same_gender"]["eer"] * PCT, 2),
                "cross_language_other_video": round(mm["cross_lang_same_gender"]["eer"] * PCT, 2),
                "same_video": round(mm["same_session_same_gender"]["eer"] * PCT, 2),
                "retrieval_R1_other_videos": round(m["retrieval"]["recall_at_1"], 3)}
    b.add(id="p2_audio_unimodal", phase="6 Phase II diagnostics", family="Diagnostics", title="Voice-only speaker verification (same-gender impostors)",
          question="Is the voice input speaker-discriminative across sessions and languages?", protocol="unimodal_diagnostic", outcome="diagnostic",
          unimodal={"WavLM + eGeMAPS (B0 input), LDA": uni(a1["wavlm_egemaps|lda"]), "WavLM all-layer mean, raw": uni(a1["wavlm_allmean|raw"]),
                    "WavLM-SV x-vector (per clip), LDA": uni(a2["wavlm_sv_xvector_perclip|lda"]), "ECAPA-TDNN, centred": uni(a2["ecapa_vox2|centred"])},
          interpretation="B0's voice input is session-dominated (30% EER across videos); ECAPA is ~3x better and nearly language-independent.",
          sources=[{"path": "public_results/sources/results/phase2/phase1_audio/metrics.json", "type": "json"},
                   {"path": "public_results/sources/results/phase2/phase1_audio/metrics_speaker_encoders.json", "type": "json"}])
    f = b.load("results/phase2/phase2_face/metrics.json")["official"]
    b.add(id="p2_face_unimodal", phase="6 Phase II diagnostics", family="Diagnostics", title="Face-only identity verification",
          question="Is the face input identity-discriminative?", protocol="unimodal_diagnostic", outcome="diagnostic",
          unimodal={k.split("|")[0]: {"same_gender": round(f[k]["metrics"]["same_gender"]["eer"] * PCT, 2),
                                      "similar_age": round(f[k]["metrics"]["similar_age"]["eer"] * PCT, 2)}
                    for k in ("arcface_insightface|centred", "farl|centred", "dinov2_l12_patch|centred", "b0_full_projected|raw")},
          interpretation="ArcFace ~1.8% vs FaRL 12.3% same-gender EER; FaRL tracks age/appearance strongly.",
          sources=[{"path": "public_results/sources/results/phase2/phase2_face/metrics.json", "type": "json"}])
    for rel, label, eid, outcome, interp in (
        ("results/phase2/phase3_alignment/compare_b0_vs_g3_ecapa_farl.json", "ECAPA + FaRL", "g3_ecapa_farl", "rejected", "Better speaker identity, worse face-voice matching on English."),
        ("results/phase2/phase3_alignment/compare_b0_vs_g3_b0_arcface.json", "B0 voice + ArcFace", "g3_b0_arcface", "rejected", "+10 EER: face-recognition features discard voice-predictable attributes."),
        ("results/phase2/phase3_alignment/compare_b0_vs_g3_ecapa_arcface.json", "ECAPA + ArcFace", "g3_ecapa_arcface", "rejected", "Worst cell of the 2x2."),
        ("results/phase2/phase3_alignment/compare_b0_vs_g3_cat_farl.json", "C1 = B0 voice + ECAPA (concat) + FaRL", "g3_cat_farl", "adopted",
         "Neutral on official English; large gain on unseen languages (see holdouts)."),
        ("results/phase2/phase3_alignment/compare_b0_vs_pj_linear_seed1.json", "Linear heads", "pj_linear", "rejected", "Significantly worse (seed 1)."),
        ("results/phase2/phase3_alignment/compare_b0_vs_pj_deep_seed1.json", "Deeper heads (512-256)", "pj_deep", "rejected", "Worse (seed 1)."),
        ("results/phase2/phase3_alignment/compare_b0_vs_ls_multipos_seed1.json", "Multi-positive InfoNCE", "ls_multipos", "rejected", "Significantly worse (seed 1)."),
        ("results/phase2/phase3_alignment/compare_b0_vs_ls_margin_seed1.json", "+ same-gender margin ranking", "ls_margin", "rejected", "No change (seed 1)."),
    ):
        e = b.compare(rel, "B0", label)
        b.add(id=eid, phase="6 Phase II diagnostics", family="Cross-modal alignment", title=label,
              question="Which factor limits cross-modal verification: encoders, heads or loss?", protocol="local_5fold", outcome=outcome,
              seeds=e["comparison"]["seeds"], interpretation=interp, **e)
    e = b.terms("results/phase2/phase5_language/cat_farl_attribute_terms.json", "C1")
    b.add(id="c1_terms", phase="6 Phase II diagnostics", family="Scoring", title="C1 with refitted terms (eval #5 zip, not submitted)",
          question="Local estimate of C1 + terms", protocol="local_5fold_terms", outcome="candidate", seeds=[1, 2, 3],
          interpretation="25.73 vs B0 + terms 25.94.", **e)
    e = b.transfer("results/phase2/phase5_language/transfer_fusion_round2.json",
                   {"b0": "B0", "ecapa_farl": "ECAPA + FaRL", "cat_farl": "C1", "b0+ecapa_farl": "B0 + ECAPA expert (fusion)"}, "Urdu", "B0")
    b.add(id="urdu_holdout_phase2", phase="6 Phase II diagnostics", family="Unseen language", title="Urdu holdout: B0 vs ECAPA variants",
          question="Does ECAPA help a language never heard in training?", protocol="urdu_holdout", outcome="adopted", seeds=[1, 2, 3],
          interpretation="C1 beats B0 on all four cells; Urdu standard -2.06, Urdu gender -2.49.", **e)
    # ================= Stage 7: Phase III =================
    for rel, label, eid, interp in (
        ("results/phase3/e2_3seeds/official_c1_vs_k64.json", "C1 with ECAPA bottleneck (64)", "p3_k64", "-0.42 n.s. vs C1 on English; no Urdu gain."),
        ("results/phase3/e2_3seeds/official_c1_vs_bl2.json", "C1 with per-block L2", "p3_bl2", "-0.43 n.s. vs C1 on English; Urdu tie; weaker on German."),
        ("results/phase3/screen_seed1/official_c1_vs_rand.json", "Control D: B0 + random 192-d", "p3_random", "English as C1, Urdu gain lost (see Urdu screen)."),
        ("results/phase3/screen_seed1/official_c1_vs_shuf.json", "Control E: speaker-shuffled ECAPA", "p3_shuffled", "Urdu gain lost; English worse."),
        ("results/phase3/screen_seed1/official_c1_vs_res.json", "C1 with residual ECAPA adapter", "p3_residual", "No gain (seed 1)."),
        ("results/phase3/screen_seed1/official_c1_vs_k32.json", "C1 with ECAPA bottleneck (32)", "p3_k32", "Worse (seed 1)."),
        ("results/phase3/screen_seed1/official_c1_vs_tr0.json", "C1 without trait losses", "p3_traits0", "~+0.1: trait heads add little after ECAPA (seed 1)."),
    ):
        e = b.compare(rel, "C1", label)
        b.add(id=eid, phase="7 Phase III complementarity", family="Fusion / controls", title=label,
              question="Why does ECAPA help, and is concatenation the best fusion?", protocol="local_5fold", outcome="rejected",
              seeds=e["comparison"]["seeds"], interpretation=interp, **e)
    e = b.transfer("results/phase3/screen_seed1/p2_vs_c1.json",
                   {"rand": "Control D (random)", "shuf": "Control E (shuffled)", "k64": "Bottleneck 64", "bl2": "Block L2", "res": "Residual adapter"},
                   "Urdu", "C1")
    b.add(id="urdu_controls_seed1", phase="7 Phase III complementarity", family="Fusion / controls", title="Urdu holdout: controls and fusion variants vs C1 (seed 1)",
          question="Is C1's Urdu gain due to dimensionality, regularisation, or speaker-aligned ECAPA content?", protocol="urdu_holdout", outcome="diagnostic", seeds=[1],
          interpretation="Random and shuffled controls are significantly worse than C1 on Urdu (lose the gain); fusion variants tie.", **e)
    e = b.transfer("results/phase3/e3_german/p3_vs_b0.json", {"b0": "B0", "c1": "C1", "bl2": "C1 block L2"}, "German", "B0")
    b.add(id="german_holdout", phase="7 Phase III complementarity", family="Unseen language", title="German holdout confirmation",
          question="Does C1's gain extend beyond Urdu?", protocol="german_holdout", outcome="adopted", seeds=[1, 2, 3],
          interpretation="German gender -2.35 [-4.38, -0.30] vs B0; candidates frozen before this test.", **e)
    def ens(rel, lang, ids):
        d = b.load(rel)
        return {lbl: {m: round(v * PCT, 2) for m, v in d["point"][k].items()} for k, lbl in ids.items()}, d
    u, ud = ens("results/phase3/e6_stabilization/p2_seed_ensembles.json", "Urdu", {"b0|single": "C1", "b0|x3": "C1 x 3", "base_b0|single": "B0", "base_b0|x3": "B0 x 3"})
    g, gd = ens("results/phase3/e3_german/p3_seed_ensembles.json", "German", {"c1|single": "C1", "c1|x3": "C1 x 3", "b0|single": "B0", "b0|x3": "B0 x 3"})
    b.add(id="seed_ensembles_holdouts", phase="7 Phase III complementarity", family="Ensemble", title="Seed ensembles on Urdu and German holdouts",
          question="Does a 3-seed C1 ensemble add robustness on unseen languages?", protocol="urdu_holdout", outcome="adopted", seeds=[1, 2, 3],
          systems_urdu=u, systems_german=g,
          ensemble_vs_single={"urdu_C1x3": {m: {"value": round(v["mean"] * PCT, 2), "ci95": [round(x * PCT, 2) for x in v["ci95"]]} for m, v in ud["ensemble_vs_single"]["b0|x3"].items()},
                              "german_C1x3": {m: {"value": round(v["mean"] * PCT, 2), "ci95": [round(x * PCT, 2) for x in v["ci95"]]} for m, v in gd["ensemble_vs_single"]["c1|x3"].items()}},
          interpretation="Urdu -0.33 / -0.38 (significant); German -0.2 (n.s.). German rows use protocol german_holdout.",
          sources=[{"path": "public_results/sources/results/phase3/e6_stabilization/p2_seed_ensembles.json", "type": "json"},
                   {"path": "public_results/sources/results/phase3/e3_german/p3_seed_ensembles.json", "type": "json"}])
    fu = b.load("results/phase3/e6_stabilization/official_c1_ensembles_and_fusion.json")["systems"]
    b.add(id="seed_ensemble_official", phase="7 Phase III complementarity", family="Ensemble", title="C1 seed ensemble on official folds",
          question="Does seed ensembling stabilise C1?", protocol="local_5fold", outcome="adopted", seeds=[1, 2, 3],
          systems={"C1": {t: round(fu["b0"][t] * PCT, 2) for t in ("standard", "gender", "mean")},
                   "C1 x 2": {t: round(fu["b0x2|mean"][t] * PCT, 2) for t in ("standard", "gender", "mean")},
                   "C1 x 3": {t: round(fu["b0x3|mean"][t] * PCT, 2) for t in ("standard", "gender", "mean")},
                   "C1 + B0 fusion": {t: round(fu["b0+base_b0|mean"][t] * PCT, 2) for t in ("standard", "gender", "mean")}},
          comparison={"a": "C1", "b": "C1 x 3", "delta": {"mean": {"value": round(fu["b0x3|mean"]["mean_minus_b0"] * PCT, 2),
                                                                    "ci95": [round(x * PCT, 2) for x in fu["b0x3|mean"]["ci95"]],
                                                                    "p_b_better": fu["b0x3|mean"]["p_better_than_b0"]}}},
          interpretation="C1 x 3 is -0.66 vs single C1 (significant). Mixing B0 back in does not beat a same-size C1 ensemble.",
          sources=[{"path": "public_results/sources/results/phase3/e6_stabilization/official_c1_ensembles_and_fusion.json", "type": "json"}])
    e = b.terms("results/phase3/e6_stabilization/c1x3_attribute_terms.json", "C1 x 3 (raw-mean ensemble)")
    b.add(id="final_c1x3_terms", phase="8 Final system", family="Final", title="Final system: C1 x 3 + ensemble-fitted terms (eval #6 zip, not submitted)",
          question="Local estimate of the final system", protocol="local_5fold_terms", outcome="candidate", seeds=[1, 2, 3],
          interpretation="25.32 local mean (C1 + terms 25.73, B0 + terms 25.94). No CodaBench score exists for this system.", **e)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research-root", type=Path, required=True)
    args = ap.parse_args()
    b = Builder(args.research_root)
    build(b)
    OUT.mkdir(exist_ok=True)
    h = b.load("results/gamp_vf/stage_C_final/diagnostics_3seeds.json")["configs"]["b0_full"]
    histograms = {"system": "B0", "protocol": "local_5fold", "score_type": "raw cosine (higher = more similar)",
                  "pooled_over": "5 folds x 3 seeds", "source": "public_results/sources/results/gamp_vf/stage_C_final/diagnostics_3seeds.json",
                  "tracks": {t: {"bins": h[t]["hist_bins"], "genuine": h[t]["hist_genuine"], "impostor": h[t]["hist_impostor"],
                                 "eer": round(h[t]["eer"] * PCT, 2)} for t in ("standard", "gender")}}
    reg = {"schema": 1, "units": "EER in percent; lower is better; deltas are B - A in EER points",
           "protocols": PROTOCOLS, "experiments": b.entries, "score_histograms": histograms}
    (OUT / "experiments.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    print(f"{len(b.entries)} experiments, {len(b.copied)} source files copied -> {OUT / 'experiments.json'}")


if __name__ == "__main__":
    main()
