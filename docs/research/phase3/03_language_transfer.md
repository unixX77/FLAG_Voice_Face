# Phase III — 03 Unseen-language transfer (E3, 2026-11-19)

**Answer to RQ3:** C1's benefit is **not Urdu-specific**. On a second held-out language, German, which was never trained on and was used as an untouched confirmation, C1 again improves the unheard-language cells most:
- German same-gender **−2.35** [−4.38, −0.30] (P = 0.99);
- German standard −1.19 [−2.32, +0.09] (P = 0.97);
- the gender-track gap halves (+3.04 → +1.49).

Hindi does not exist in the data.

Code: `phase2/language_transfer_fusion.py` (generalised with `--prefix` / `--unheard-language`; regression-checked against the stored Phase II Urdu results), `phase3/ensemble_transfer.py`. Results: `results/phase3/e3_german/` (per-trial scores in `p3_scores.npz`).

## Protocols (strict: unseen people **and** unseen language)

| | P2 Urdu (development) | **P3 German (confirmation)** |
|---|---|---|
| Held out per fold | 1/5 of MAV-Celeb v1 people (≈ 13), English + Urdu clips | 1/5 of MAV-Celeb v3 people (10), English + German clips |
| Excluded from training | all Urdu rows | all German rows |
| Remaining leak in training | VoxCeleb2 LID: 37 Urdu, 75 Hindi of 9,176 clips | VoxCeleb2 LID: 5 German clips |
| Trials per fold and track | ≈ 8,000–10,000 | 3,252–4,194 |
| Used for selection? | yes (Phase II and E2) | **no**: candidates frozen first (`results/phase3/p3_candidates_frozen.txt`) |

The German protocol is smaller (10 people per fold), so its intervals are wider.

## Results (3 seeds × 5 folds; EER %)

| System | Std English | Std unheard | Std gap | Gender English | **Gender unheard** | Gender gap |
|---|---|---|---|---|---|---|
| **Urdu (P2)** B0 | 24.28 | 26.06 | +1.78 | 37.37 | 38.86 | +1.49 |
| Urdu C1 | 22.96 | 23.94 | +0.98 | 36.18 | 36.32 | +0.15 |
| Urdu C1 × 3 seeds | 22.66 | 23.62 | +0.96 | 35.94 | 35.88 | −0.06 |
| **German (P3)** B0 | 28.62 | 30.37 | +1.75 | 39.10 | 42.14 | +3.04 |
| German **C1** | 28.48 | **28.99** | +0.51 | 38.23 | **39.73** | +1.49 |
| German C1 × 3 seeds | 28.12 | 28.85 | +0.73 | 37.80 | 39.38 | +1.58 |
| German F3 block L2 (frozen candidate) | 28.42 | 29.44 | +1.03 | 38.96 | 40.10 | +1.14 |

**C1 − B0 [95% CI], P(better):**

| | Std English | Std unheard | Gender English | Gender unheard | Gender gap |
|---|---|---|---|---|---|
| Urdu | −1.33 [−2.35, −0.42], 1.00 | −2.06 [−3.15, −0.95], 1.00 | −1.25 [−2.69, +0.08], 0.96 | −2.49 [−3.93, −1.01], 1.00 | −1.24 [−2.61, 0.00], 0.97 |
| **German** | −0.52 [−1.70, +0.68], 0.80 | −1.19 [−2.32, +0.09], 0.97 | −1.15 [−3.05, +0.63], 0.88 | **−2.35 [−4.38, −0.30], 0.99** | −1.20 [−3.02, +0.74], 0.90 |

- **F3 block L2 vs B0 on German:** gender German −1.74 [−4.07, +0.49], weaker than C1. Plain concatenation stays best.
- **Seed ensemble (C1 × 3 vs single C1):** Urdu −0.33 / −0.38 (std / gender, significant); German −0.23 / −0.25 (n.s.).

## Negative transfer and caveats

- **No seen-language cost:** in both proxies C1 also improves (or doesn't hurt) the English cell.
- **Not a subset or recording artefact** (`04`): the Urdu gain is flat across duration bins and present for other-video genuine pairs.
- **The size of the gain depends on the language and the people.** Urdu −2.5, German −2.4 on gender, but standard Urdu −2.1 against German −1.2. Bangla may differ again.
- **Official English folds:** C1 is neutral there (+0.05; −0.21 with terms). The proxies' English cells (other people, models trained without one language) gain more. The challenge's English cells should be expected to gain little.
- German-holdout models also lost a fifth of the MAV-Celeb v3 people and all German clips. Absolute EERs (≈ 28–42) are therefore higher than on the official folds and not comparable to them.
