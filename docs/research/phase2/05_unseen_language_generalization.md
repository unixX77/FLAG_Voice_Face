# Phase II — 05 Unseen-language generalisation (Gate 5, 2026-11-19)

**Finding:**
- Adding the language-robust ECAPA speaker embedding to the voice side, as one model (WavLM ⊕ eGeMAPS ⊕ ECAPA) or as a fused expert, **improves the unheard language by about 2 EER points, significantly**, and removes the heard/unheard gap.
- This is the opposite of GAMP-VF's same-gender sampling, which helped only the heard language.
- **Open discrepancy:** on the *official* English folds the same single model is neutral (+0.05), while on the proxy's English cell (other people) it gains −1.3. See the caveats.

## Protocol (strict: unseen identities **and** unseen language)

- **P2 proxy:** each fold holds out one fifth of the MAV-Celeb v1 people (English + Urdu). The models never hear any Urdu (`--exclude-train-languages Urdu`) and never see those people. This is the configuration relevant to Bangla.
- **Trials:** identical to `evaluate_language_transfer.py` (same builder and RNG seeds): 3 genuine + 3 impostor faces per held-out voice clip, any-gender (standard) or same-gender.
- **New:** `FLAG_2027/phase2/language_transfer_fusion.py` scores each seed separately (the older tool averaged the seeds' scores), z-normalises with other-fold statistics, and compares every 2-model fusion with a 2-seed B0 ensemble.
- **Checks:** the code verifies each checkpoint's hold-out equals the fold's MAV-Celeb v1 split and that Urdu was excluded. A unit test covers this too.
- **Hindi:** no data. **German holdout:** not run (49 German speakers, and Urdu answers the question). **Unseen language with *seen* identities:** not run, because the challenge's Bangla speakers are unseen people.

## Results (3 seeds × 5 folds; EER %; `results/phase2/phase5_language/transfer_fusion_round{1,2}.json`)

| System | Std English (heard) | Std Urdu (unheard) | Std gap | Gender English | Gender Urdu (unheard) | Gender gap |
|---|---|---|---|---|---|---|
| B0 (`p2_full`) | 24.28 | 26.06 | +1.78 | 37.37 | 38.86 | +1.49 |
| B0 × 2 seeds (control) | 24.13 | 25.91 | +1.77 | 37.26 | 38.76 | +1.50 |
| GAMP same-gender sampling (`p2_b3_sg`) | 24.08 | 26.09 | +2.01 | 37.09 | 38.95 | +1.86 |
| ECAPA + FaRL | 24.92 | 24.58 | −0.34 | 38.10 | 36.14 | −1.96 |
| **WavLM ⊕ eGeMAPS ⊕ ECAPA (single model)** | **22.96** | **23.94** | +0.98 | **36.18** | **36.32** | +0.15 |
| B0 + ECAPA expert (F1 fusion) | 23.17 | 23.65 | +0.48 | 36.32 | 36.32 | 0.00 |
| B0 + concat expert (F1 fusion) | 23.25 | 24.56 | +1.30 | 36.48 | 37.20 | +0.72 |

**Paired differences [95% CI], P(better):**

| Comparison | Std English | Std Urdu | Gender English | Gender Urdu | Gender gap |
|---|---|---|---|---|---|
| concat single model vs B0 | −1.33 [−2.35, −0.42], 1.00 | **−2.06 [−3.15, −0.95], 1.00** | −1.25 [−2.69, +0.08], 0.96 | **−2.49 [−3.93, −1.01], 1.00** | −1.24 [−2.61, 0.00], 0.97 |
| B0 + ECAPA vs B0 × 2 | — | **−2.17 [−3.59, −0.64], 1.00** | −1.01 [−2.36, +0.16], 0.95 | **−2.28 [−4.22, −0.15], 0.99** | −1.27 [−2.95, +0.39], 0.93 |
| B0 + concat vs B0 × 2 | — | −1.34 [−1.94, −0.76], 1.00 | −0.91 [−1.66, −0.22], 1.00 | −1.49 [−2.23, −0.75], 1.00 | −0.58 |
| ECAPA + FaRL vs B0 | +0.64 | −1.38 [−3.69, +1.17], 0.87 | +0.55 | −2.47 [−5.86, +1.24], 0.90 | **−3.02 [−5.83, −0.52], 0.99** |
| GAMP same-gender + B0 vs B0 × 2 | — | +0.12 | −0.10 | +0.15 | +0.25 (P narrower 0.05) |

## What survives language shift, and what fails

| Change | Heard language (official folds) | Unheard language (proxy) | Survives? |
|---|---|---|---|
| GAMP same-gender sampling | −0.41 gender (n.s.) | +0.05 Urdu gender; gap widens | **no** |
| ECAPA replacing WavLM | +1.8 (worse) | −1.4 / −2.5 Urdu | language-robust but worse at home |
| **ECAPA added to WavLM ⊕ eGeMAPS** | +0.05 (−0.21 with terms) | **−2.1 / −2.5 Urdu**, −1.3 English | **yes; the only change that helps the unheard cells** |
| seed ensemble | −0.3 to −0.4 | −0.16 / −0.14 Urdu | small, consistent |
| ArcFace | +10 | not run (fails at home) | — |

## Robustness

- **Voice duration** (official folds, 3 seeds; `duration_robustness_official.json`). B0 vs concat, EER for < 3 / 3–6 / 6–10 / > 10 s clips:
  - standard: 22.5 / 19.4 / 19.3 / 21.2 vs 22.5 / 19.1 / 19.5 / 21.1;
  - gender: 33.7 / 32.3 / 29.3 / 32.6 vs 34.3 / 31.4 / 30.3 / 30.4.
  - No new short-clip weakness, although ECAPA alone is duration-sensitive (Phase 1).
- **Not run:** noise, reverberation and microphone shifts (no controlled corruption set was built), phonetic-content variation, missing traits.

## Caveats

- **Proxy English vs official English.** The proxy's English cell (MAV-Celeb v1 people, trained without Urdu) improves by −1.3, the official folds' English people by 0.
  - Possible reasons: different people and recording domains; the P2 models have less voice training data (Urdu removed), where ECAPA's prior helps more.
  - The challenge's English cells may therefore gain anywhere from 0 to about 1. The Bangla cells are what the proxy addresses.
- **One unheard language only (Urdu).** Bangla is a different language. Expected Bangla gains are not guaranteed by Urdu gains.
- ECAPA was pretrained on VoxCeleb2, which is allowed: public frozen encoders may have seen any language, per the FAQ, §8 of the progress log.
