# Phase III — 05 Attribute-conditioned alignment (E5, 2026-11-19)

**Answer to RQ5:** after ECAPA concatenation, additional attribute supervision adds **little**.
- The existing trait heads contribute only about 0.1 EER, and the fixed age/gender score terms contribute about 0.5.
- The score terms are **not double-counted** with the trait heads: their gain is the same with trait heads on or off.
- No larger attribute network was built, as there is no opportunity to justify one.

Code: existing trait heads (`trait_heads.py`) and `analysis/finalize_attribute_terms.py`. Results: `results/phase3/screen_seed1/*tr0*`, `*tr01*`, `results/phase3/e5_attributes/`.

## Ablations (seed 1 × 5 folds; C1 recipe with only the trait-loss weight changed)

| Variant | Official std / gender | Official mean Δ vs C1 [95% CI] | Urdu std Δ vs C1 [95% CI] | Urdu gender Δ vs C1 [95% CI] |
|---|---|---|---|---|
| B: C1, trait weight 0.2 (existing) | 20.81 / 32.03 | — | — | — |
| A: C1 without trait losses (0) | 21.06 / 32.03 | +0.12 [−0.1, +0.5] | +0.18 [0.0, +0.3] | +0.14 [−0.1, +0.3] |
| C: C1, trait weight 0.1 | 21.06 / 31.99 | +0.10 [−0.1, +0.3] | +0.08 [0.0, +0.2] | +0.06 [−0.1, +0.2] |
| D: uncertainty-weighted traits | not run | | | |
| E: residual attribute score (existing fixed age/gender terms, fitted on fold models) | see below | | | |

- The intervals are narrow because A–C differ from C1 only by a loss weight, under the same seed and batches.
- Removing the trait losses costs about 0.1–0.2 consistently. That is real but negligible.
- **D was not run:** there are no per-label reliability estimates. The age and build labels are point estimates from frozen face models.

## Score terms vs trait heads (E; seed-1 fold models, terms refitted per variant)

| Variant | Without terms (std / gender / mean) | With terms | Gain from terms |
|---|---|---|---|
| C1 (trait 0.2) | 20.81 / 32.03 / 26.42 | 20.14 / 31.66 / 25.90 | −0.52 |
| trait 0.1 | 21.06 / 31.99 / 26.52 | 20.31 / 31.71 / 26.01 | −0.51 |
| trait 0 | 21.06 / 32.03 / 26.54 | 20.37 / 31.74 / 26.06 | −0.49 |

- The terms' gain doesn't depend on whether the embedding was trained with trait heads, so the two do not encode the same signal twice.
- With 3 seeds, C1 + terms = 25.73 (`docs/phase2/06`).

## Confounders and limits

- The terms and trait heads use estimated attributes: InsightFace face age, the audeering voice-age estimate, and FaRL zero-shot build. These are predictions, not ground truth, and are used as weak supervision only.
- Within-gender analysis is built in: the gender track's terms have w_gender = 0, and the trait-probe correlations in `01` are within gender.
- **Not done, by design:** no ethnicity, nationality or language-from-face inference, and no demographic attribute is ever used as an identity label.
