# Phase III — 06 Candidate stabilisation (E6, 2026-11-19)

**Recommendation:** a **3-seed C1 ensemble** (mean of per-trial distances) **+ age/gender terms refitted on the ensemble**.
- Same architecture and inputs as C1, so no new modelling assumption.
- Significantly better than single C1 on the official folds and on Urdu, and consistent (n.s.) on German.
- Mixing B0 back in is **not** recommended: it hurts the unheard language.

Candidate zip: `submissions/eval6_ecapa_concat_x3_terms.zip` (not submitted).

## Equivalent scoring configurations (3 seeds × 5 folds)

| System | Official std / gender / mean | Δ vs C1 [95% CI], P | Urdu std / gender | German std / gender |
|---|---|---|---|---|
| B0 single | 20.34 / 31.97 / 26.15 | −0.05 (i.e. C1 − B0 = +0.05 [−1.34, +1.25]) | 26.06 / 38.86 | 30.37 / 42.14 |
| B0 × 3 seeds | 20.13 / 31.36 / 25.74 | — | 25.87 / 38.70 | 30.07 / 42.23 |
| C1 single | 20.48 / 31.94 / 26.21 | — | 23.94 / 36.32 | 28.99 / 39.73 |
| C1 × 2 seeds | 20.03 / 31.36 / 25.70 | −0.51 [−0.62, −0.07], 0.99 | 23.70 / 36.03 | — |
| **C1 × 3 seeds** | **19.91 / 31.19 / 25.55** | **−0.66 [−0.86, −0.12], 0.99** | **23.62 / 35.88** | **28.85 / 39.38** |
| C1 + B0 (z-mean) | 19.96 / 31.22 / 25.59 | −0.11 vs C1 × 2 (n.s.) | 24.56 / 37.20 (**worse** than C1 × 2: +0.81 / +1.15, significant) | 29.17 / 40.72 |
| C1 + B0 (logistic, out-of-fold) | 20.04 / 31.38 / 25.71 | −0.10 vs C1 × 2 logistic | — | — |

Proxy ensembles vs single C1: Urdu −0.33 [−0.47, −0.19] std, −0.38 [−0.57, −0.17] gender; German −0.23 [−0.69, +0.29], −0.25 [−0.80, +0.34].

## Calibration and score terms

- EER is invariant to any strictly monotonic transform of a cell's scores, so affine or temperature calibration alone can't change EER. Only conditional terms or fusion can.
- **Raw-mean vs z-mean ensemble:** 25.57 vs 25.55 on the official folds. The raw mean needs no score statistics, so every trial is still scored independently, as the rules require. It is used in the zip.
- **Age/gender terms** (fitted on the fold models, exactly as for B0):

  | System | Without terms | With terms | Weights (standard: age, gender; gender track: age) |
  |---|---|---|---|
  | B0 | 26.15 | 25.94 | 0.02, 0.05; 0.03 |
  | C1 | 26.21 | 25.73 | 0.05, 0.05; 0.03 |
  | **C1 × 3 (raw mean)** | 25.57 | **25.32** (std 19.63, gender 31.01) | 0.02, 0.05; 0.02 |

- The terms are not double-counted with the trait losses (`05`). The term weights are chosen in-sample over the 5 folds from a small grid (the existing method); their small optimism applies equally to all rows.

## Model-selection accounting

- **Phase III training configurations:**
  - 8 screened at seed 1 on two protocols (controls D/E, F2 × 2, F3, F4, E5 × 2);
  - 2 promoted to 3 seeds (F2 k = 64, F3);
  - 3 run on the German confirmation (B0, C1, F3).
- **Score-level systems compared:** single, ×2, ×3, + B0 (mean / logistic).
- The intervals above do not correct for this search. The German holdout was **not** used for selection, so its C1-vs-B0 result is the least selection-biased estimate in this phase.

## Candidate artefacts (nothing overwritten, nothing submitted)

- `checkpoints/v6_ecapa_concat_ens/`:
  - members = `checkpoints/v5_ecapa_concat/full_data` (seed 1) + `seed_2/full_data`, `seed_3/full_data` (md5 `7d9face4…`, `2dc335d8…`);
  - same recipe and data, only the seed differs;
  - `attribute_terms.json` fitted with `finalize_attribute_terms.py --ensemble`.
- `submissions/eval6_ecapa_concat_x3_terms.zip`, md5 `4b6fdcd48c8801df72abf4b1d37a39e7`. Validated; same trial order as eval #3–#5; score correlation with eval #5 is 0.98.
- **Tooling:**
  - `generate_cross_modal_submission.py --checkpoint` accepts several checkpoints (mean distance per trial; single-checkpoint output is byte-identical to eval #5);
  - `finalize_attribute_terms.py --ensemble`.
