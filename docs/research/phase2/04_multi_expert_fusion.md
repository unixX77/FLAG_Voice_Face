# Phase II — 04 Multi-expert fusion (Gate 4, 2026-11-19)

**Verdict:**
- On the heard language (official folds), experts add **no information beyond a seed ensemble of B0**. Complex fusion (F3 gated, F4 mixture-of-experts) is **not justified** and was not built.
- Low-capacity fusion is the right tool, and the equal-weight mean is as good as logistic regression.
- On the unheard language, however, adding an ECAPA-based expert gives a large, significant gain (`05`).

Code: `FLAG_2027/phase2/expert_fusion.py`. Results: `results/phase2/phase4_fusion/fusion_round1_controls.json`.

## Method

- **Experts** score the same fixed fold trials. Scores are z-normalised with statistics from the *other* folds.
- **F1:** equal-weight mean.
- **F2:** logistic regression fitted on the other four folds (out-of-fold, identity-disjoint), applied to the held-out fold.
- **Controls:** every fusion is compared with a **B0 seed ensemble of the same size** (B0 seed s + B0 seed s′), so a gain can't be mere ensembling.
- The Expert C/D/E roles are covered by existing models:
  - C, traits: inside B0;
  - D, spectral: eGeMAPS, inside B0;
  - E, language-robust: the ECAPA-based models.

## Results (official five-fold, 3 seeds; mean EER %)

| System | Mean | Δ vs B0 [95% CI] | Δ vs same-size B0 ensemble [95% CI] | P(better than control) |
|---|---|---|---|---|
| F0: B0 | 26.15 | — | — | — |
| B0 × 2 seeds, mean | 25.85 | −0.30 [−0.46, +0.12] | (control) | — |
| B0 × 3 seeds, mean | 25.74 | −0.41 [−0.66, +0.20] | (control) | — |
| F1: B0 + ECAPA + FaRL expert | 25.63 | −0.52 [−2.13, +0.74] | −0.22 [−1.94, +1.00] | 0.69 |
| F2: B0 + ECAPA + FaRL expert | 25.76 | −0.39 | −0.15 [−1.44, +0.75] | 0.70 |
| F1: B0 + (B0 voice ⊕ ECAPA) expert | 25.59 | −0.56 [−1.23, +0.14] | −0.26 [−1.11, +0.33] | 0.85 |
| F1: B0 × 2 + ECAPA + FaRL expert | 25.48 | −0.68 [−1.86, +0.10] | −0.27 [−1.63, +0.47] vs B0 × 3 | 0.83 |
| F1: B0 + GAMP same-gender model | 26.07 | −0.08 | — | — |
| F1: B0 + eval #3 model (WavLM only) | 26.21 | +0.06 | — | — |

## Diversity vs B0 (pooled trials; errors at each model's own EER threshold)

| Expert | Pearson r, std / gender | B0 gender errors also made by the expert | Share of B0 gender errors the expert corrects |
|---|---|---|---|
| another B0 seed (control) | 0.975 / 0.944 | — | — |
| ECAPA + FaRL | 0.83 / **0.69** | 4,143 of 6,724 | **38%** |
| B0 voice ⊕ ECAPA | 0.95 / 0.89 | 5,082 of 6,724 | 24% |
| eval #3 (WavLM only) | 0.97 / 0.94 | 5,573 of 6,724 | 17% |
| GAMP same-gender | 0.996 / 0.99 | 6,245 of 6,724 | 7% |

The ECAPA expert is genuinely diverse: its correlation is far below a seed's. But it makes more errors of its own, so on English the diversity cancels out.

## Recommendation

- Use **F1 (equal-weight mean of z-scored cosines)** whenever fusing. It is as good as F2 and has nothing to overfit.
- A seed ensemble of the chosen model is a cheap, rule-compliant, consistent small gain (−0.3 to −0.4, n.s.).
- The language-robust ECAPA expert should be judged on the unheard-language proxy, where it matters (`05`).
