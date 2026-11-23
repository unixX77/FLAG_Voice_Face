# Phase III — 04 Duration and recording-domain robustness (E4, 2026-11-19)

**Answer to RQ4:**
- C1's gain over B0 does **not** depend on clip duration or on whether a genuine face comes from the voice's own video.
- On the Urdu proxy the gain has the same size in every duration bin, and it holds for other-video genuine pairs.
- On the official English folds the per-bin differences scatter around 0 with wide intervals. There is no stable interaction, so **adaptive reliability weighting (F6) is not justified** (Gate 4 of `00`).

Code: `FLAG_2027/phase3/duration_domain.py`. Results: `results/phase3/e4_duration/{official_duration,p2_duration_session}.json` and `per_trial_scores.npz` (every per-seed trial score of B0 and C1).

**Method:**
- B0 and C1, 3 seeds each, scored on identical trials.
- Strata by the **voice clip's** duration (wav header for official; the extraction-time duration cache for external), by voice language, and by whether the genuine face shares the voice's source video. Impostors are unchanged in every stratum.
- Paired bootstrap: people resampled within fold, 1,000 draws.

## Official folds (English; clips per bin: < 3 s 1,087 · 3–6 s 2,643 · 6–10 s 1,483 · > 10 s 1,272)

| Track | Bin | B0 | C1 | C1 − B0 [95% CI] |
|---|---|---|---|---|
| standard | < 3 s | 22.53 | 22.53 | 0.00 [−1.37, +2.41] |
| standard | 3–6 s | 19.43 | 19.06 | −0.37 [−1.72, +1.17] |
| standard | 6–10 s | 19.25 | 19.54 | +0.28 [−1.58, +1.46] |
| standard | > 10 s | 21.16 | 21.13 | −0.02 [−1.93, +2.26] |
| gender | < 3 s | 33.73 | 34.29 | +0.57 [−2.29, +3.60] |
| gender | 3–6 s | 32.32 | 31.44 | −0.89 [−2.31, +1.18] |
| gender | 6–10 s | 29.28 | 30.27 | +0.99 [−1.32, +3.03] |
| gender | > 10 s | 32.55 | 30.44 | −2.11 [−4.22, +2.55] |

Interaction (short-bin Δ − long-bin Δ): standard +0.36 [−2.52, +3.21]; gender +1.82 [−2.57, +6.20]. **No evidence.**

## Urdu proxy (P2; MAV-Celeb clips are ≥ 4 s, so there is no < 3 s bin)

| Track · language | 3–6 s | 6–10 s | > 10 s | Genuine same video | **Genuine other video** |
|---|---|---|---|---|---|
| standard · English | −1.39 [−2.63, −0.21] | −1.35 [−2.94, −0.01] | −1.30 [−2.17, −0.38] | −0.22 [−1.90, +0.78] | −1.57 [−2.53, −0.64] |
| standard · **Urdu** | −2.30 [−3.33, −0.96] | −2.62 [−3.56, −1.39] | −1.94 [−3.05, −0.77] | −2.47 [−3.95, −0.62] | **−2.06 [−3.08, −0.95]** |
| gender · English | −0.84 [−2.66, +0.59] | −0.86 [−3.09, +0.30] | −1.45 [−2.73, +0.16] | −0.02 [−2.43, +1.56] | −1.42 [−2.85, −0.17] |
| gender · **Urdu** | −2.26 [−3.70, −0.52] | −2.89 [−4.38, −1.34] | −2.52 [−4.12, −1.02] | −3.05 [−5.21, −0.62] | **−2.40 [−3.86, −1.06]** |

Values are C1 − B0 [95% CI].

- The Urdu gain is flat across duration (−1.9 to −2.9) and present for other-video genuine pairs, so it is not a session or channel artefact.
- On English, C1 helps other-video pairs (−1.4 to −1.6) but not same-video pairs (≈ 0). That is consistent with ECAPA adding session-invariant speaker identity, which matters most when the face comes from a different recording.

## Recording conditions not analysed

There is no metadata for noise, reverberation, microphone, clipping or SNR, and no corruption set was built (it would need re-extraction from audio). Nothing about those conditions is claimed.
