# Phase III — 01 ECAPA–WavLM feature complementarity (E1, 2026-11-19)

**Answer to RQ1:** ECAPA adds **speaker-identity information that B0's WavLM ⊕ eGeMAPS input lacks and that is stable across languages**, and that information is what carries C1's unseen-language gain.
- **Not dimensionality:** B0 ⊕ fixed random 192-d features gets no Urdu gain.
- **Not ECAPA as a generic regulariser:** shuffling ECAPA among training speakers removes the gain.
- **Not face traits:** the part of ECAPA that B0 can't predict carries little face-age or build information.

Code: `FLAG_2027/phase3/feature_complementarity.py`. Frozen-feature results: `results/phase3/e1_features/complementarity.json`. Controls: `results/phase3/screen_seed1/`.

## E1a: frozen statistics (identity-disjoint)

Ridge maps and probes are fitted on official + VoxCeleb2 people and evaluated on the 113 MAV-Celeb v1/v3 people.

| Quantity | Value | Reading |
|---|---|---|
| Linear CKA, B0 voice vs ECAPA (clip / person mean) | 0.25 / 0.59 | low similarity per clip; moderate at person level (both encode gender, age, …) |
| Linear CKA, WavLM vs ECAPA / eGeMAPS vs ECAPA / WavLM vs eGeMAPS | 0.24 / 0.19 / 0.37 | ECAPA overlaps least with eGeMAPS |
| R² of ECAPA predicted from B0 voice (unseen people) | **0.18** | 82% of ECAPA variance is not linearly predictable from B0's input |
| R² of B0 voice predicted from ECAPA | 0.13 | |
| Effective rank (z-scored): B0 voice / WavLM / eGeMAPS / ECAPA | see JSON | ECAPA uses most of its 192 dimensions |
| Same speaker's language shift ÷ within-speaker spread, B0 / ECAPA | 0.88 / 0.90 | ECAPA does **not** shift less between languages; it separates speakers more (below) |

CKA is representational similarity, not task redundancy or causal contribution.

**Speaker verification of each part** (Phase 1 bilingual trials, same-gender impostors, centred cosine):

| Representation | Same language | Other language | Same video | R@1 | Cross-language R@1 |
|---|---|---|---|---|---|
| ECAPA (full) | 10.2 | 10.2 | 4.4 | 0.97 | 0.96 |
| ECAPA part predictable from B0 | 23.7 | 23.8 | 8.1 | 0.88 | 0.79 |
| **ECAPA residual (not predictable from B0)** | **14.1** | **14.0** | 6.0 | 0.95 | **0.94** |
| B0 voice (z-scored) | 39.6 | 39.9 | 15.8 | 0.62 | 0.42 |

**Probes** (person-level face traits: within-gender Pearson r; clip-level gender/language: balanced accuracy on unseen people):

| Representation | Face age, men / women | Face build, men / women | Gender | Language |
|---|---|---|---|---|
| B0 voice | 0.74 / 0.62 | 0.20 / 0.64 | 0.95 | 0.81 |
| ECAPA | 0.75 / 0.49 | 0.36 / 0.66 | 0.88 | 0.79 |
| **ECAPA residual** | 0.42 / **0.11** | 0.11 / 0.32 | 0.80 | 0.75 |
| B0 ⊕ ECAPA | 0.77 / 0.60 | 0.29 / 0.67 | 0.94 | 0.81 |

- **What is shared:** gender and most of the age signal (the predictable part).
- **What differs:** ECAPA's residual is a strong, language-independent speaker code (14% EER in both languages; cross-language R@1 0.94) with little face-trait content.
- Face-trait probes are limited to the two labels we have (estimated face age, FaRL zero-shot build), so "little face-related content" is relative to those two only.

## E1b / E1c: training controls (seed 1 × 5 folds; B0 recipe; C1 = WavLM ⊕ eGeMAPS ⊕ ECAPA)

| Condition | Official mean Δ vs C1 [95% CI] | Urdu std Δ vs C1 | Urdu gender Δ vs C1 | Urdu std / gender vs B0 |
|---|---|---|---|---|
| C1 (reference; Urdu 24.03 / 36.43) | — | — | — | −2.22 / −2.50 |
| **D: B0 ⊕ fixed random N(0, 1) 192-d** | +0.01 [−1.3, +1.4] | **+1.81** [+0.8, +2.7] | **+2.07** [+0.5, +3.5] | −0.30 / −0.33 (n.s.) |
| **E: C1 with ECAPA shuffled among training speakers** (same gender and source; clean evaluation) | +1.03 [−0.3, +2.9] | **+1.96** [+0.4, +3.3] | **+2.77** [+0.8, +4.8] | −0.15 / +0.37 (n.s.) |

- **Dimensionality is ruled out:** D matches C1 on English (both ≈ B0) but has none of C1's Urdu gain.
- **Speaker alignment is essential:** with ECAPA attached to the wrong speaker during training, the Urdu gain disappears and English gets worse.
- On English (official folds) ECAPA, random features and B0 are within seed noise of each other. The ECAPA benefit is specific to the unheard language and to other people (P2 English cell −1.3).

## Is compression justified?

See `02`. A 64-d bottleneck keeps C1's Urdu performance, but neither it nor 32-d improves on plain concatenation for the unheard language.
