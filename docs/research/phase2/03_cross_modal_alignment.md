# Phase II — 03 Cross-modal alignment (Gate 3, 2026-11-19)

**Diagnosis:** the remaining errors are **not** caused by weak unimodal identity encoders, nor by projection capacity or the loss.
- Replacing either encoder with a much better *identity* encoder makes face–voice verification worse: ArcFace by about +10 EER, ECAPA by +1.8.
- B0's projection heads and SupCon loss are locally optimal: no tested change helped.
- Face–voice association runs through **shared physical attributes**, which attribute-rich encoders (FaRL, WavLM / eGeMAPS) keep and identity encoders discard.
- The one useful change is **adding** ECAPA to B0's voice input. It is neutral on the heard language and strongly helps an unheard language (see `05`).

All runs use B0's full recipe (traits 0.2, language losses 0.1, 90 epochs, 3 seeds × 5 folds unless noted), changing exactly one factor. Runner: `runs/phase2/run_all.sh`. Comparisons: `compare_fold_runs.py` (paired identity bootstrap), outputs in `results/phase2/phase3_alignment/`.

## 3.2 B0 audit (verified in code)

- **Projectors:** standardised input (training statistics) → input dropout 0.2 + Gaussian noise 0.1 → Linear(·, 256) + GELU + dropout 0.3 → Linear(256, 128) → L2 norm, separately for face (FaRL 512) and voice (WavLM ⊕ eGeMAPS 856).
- **Loss:** symmetric supervised contrastive (SupCon, τ = 0.07, all same-identity samples positive), plus pair alignment 0.25, shared trait heads 0.2 (age, build, gender), language adversary 0.1 and cross-lingual alignment 0.1. The ArcFace identity loss is off.
- **Batches:** P16 × K8 identities, 50% official people, 40 batches/epoch.
- **Score:** cosine plus fixed age/gender terms (fitted on fold models).
- **Objective:** it optimises identity-level cross-modal ranking, but in a space that only keeps what both modalities can express. The trait heads make that explicit.

## 3.6 Encoder substitution (2 × 2 + complement)

| Voice input | Face input | Standard | Gender | Mean Δ vs B0 [95% CI] | P(better) | Folds better (std / gen) |
|---|---|---|---|---|---|---|
| WavLM ⊕ eGeMAPS (B0) | FaRL (B0) | 20.34 | 31.97 | — | — | — |
| ECAPA | FaRL | 22.21 | 33.64 | +1.77 [−0.65, +3.90] | 0.08 | 1/5, 3/5 |
| WavLM ⊕ eGeMAPS | **ArcFace** | 30.89 | 41.34 | **+9.96** [+6.62, +13.27] | 0.00 | 0/5, 0/5 |
| ECAPA | ArcFace | 32.91 | 41.80 | **+11.20** [+8.00, +14.34] | 0.00 | 0/5, 0/5 |
| **WavLM ⊕ eGeMAPS ⊕ ECAPA** | FaRL | 20.48 | 31.94 | +0.05 [−1.34, +1.25] | 0.54 | 2/5, 3/5 |
| … with refitted age/gender terms | | 19.94 | 31.52 | B0 + terms 25.94 → **25.73** (−0.21) | — | — |

**Readings:**
- **Face side.** ArcFace (1.8% unimodal EER) is catastrophic cross-modally, against FaRL (12.3% unimodally). Face-recognition training removes the appearance attributes that a voice can predict.
- **Voice side.** ECAPA (9.8% across sessions) loses to WavLM ⊕ eGeMAPS (30.5%) as the *sole* input, and adds nothing on the official English folds when concatenated.
- **Trait information is not the explanation** (`trait_probe.json`; person-level ridge probes, identity-disjoint, within-gender Pearson r):

  | Voice representation | Face age, men / women | Face build, men / women |
  |---|---|---|
  | WavLM ⊕ eGeMAPS | 0.74 / 0.62 | 0.20 / 0.64 |
  | ECAPA | 0.74 / 0.48 | 0.36 / 0.66 |
  | x-vector | 0.74 / 0.63 | 0.39 / 0.59 |

  All voice encoders carry similar face-trait information. (The FaRL build probe is circular, since build labels are FaRL zero-shot outputs.)
- **A shared-session shortcut explains only a little.** On unseen MAV-Celeb people, same-video face–voice genuine pairs score slightly higher than other-video pairs: EER 37.0 vs 39.2, +0.012 cosine, higher in 56% of paired cases (`session_shortcut_p2.json`).

## 3.3 / 3.4 Projection and loss ablations (B0 inputs; seed 1 × 5 folds, paired vs B0 seed 1)

| Change (only this) | Mean Δ [95% CI] | Gender Δ | Folds better (std / gen) |
|---|---|---|---|
| linear heads (no hidden layer) | **+2.45** [+1.18, +3.44] | +2.80 | 1/5, 0/5 |
| two hidden layers (512 → 256) | +1.08 [−0.52, +2.79] | +1.74 | 2/5, 1/5 |
| hidden dropout 0.1 (instead of 0.3) | +0.42 [−0.40, +0.99] | +0.70 | 2/5, 0/5 |
| multi-positive InfoNCE (instead of SupCon) | **+0.96** [+0.15, +1.78] | +0.22 | 0/5, 2/5 |
| + same-gender margin ranking (0.25, m = 0.1, top-5) | +0.03 [−0.10, +0.17] | +0.02 | 1/5, 1/5 |

- B0's capacity, regularisation and loss are a local optimum.
- The residual MLP (D) and a shared compatibility network (F) were **not run**: added depth already hurts, and the learned pairwise verifier was negative before (`progress_readme` §6.4).
- Bidirectional InfoNCE, identity-balanced batches and same-gender hard-negative mixtures are already part of B0 or were tested in GAMP-VF (B3).

## 3.5 Information and ambiguity (held-out official people; `analysis_*.json`)

| | Voice → face top-1, 14 people (chance 7%) | Same-gender gallery top-1 (chance 15%) | Identity reciprocal top-1 | Correct face ranked first but scored below the EER threshold | Correct identity outside the top 3 (same gender) |
|---|---|---|---|---|---|
| B0 | 0.42 | 0.43 | 0.32 | 0.02 | 0.18 |
| ECAPA + FaRL | 0.38 | 0.41 | 0.31 | 0.06 | 0.20 |

- **B0's errors are ranking errors, not calibration errors.** When its ranking is right, the score is almost always above threshold.
- The ambiguity is within gender: among about 7 same-gender candidates the right face ranks first only 43% of the time.

## Gate-3 decision

| Lever | Evidence | Verdict |
|---|---|---|
| better audio identity encoder | ECAPA alone +1.8; as a complement 0 on English | not on heard languages, **but see `05`** |
| better face identity encoder | ArcFace +10 | **no** |
| projection / loss | all tested changes ≥ 0 | **no** |
| attribute-rich features | FaRL ≫ ArcFace; WavLM / eGeMAPS ≥ ECAPA alone | **the information that transfers** |
| fundamental ambiguity | same-gender top-1 is 43% of 7 even with the best model | likely large. The finite-sample evidence does not establish an upper bound |
