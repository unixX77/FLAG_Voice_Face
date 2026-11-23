# Phase III — 00 Research plan: ECAPA–WavLM complementarity (2026-11-19)

**Goal:** explain C1's unseen-language gain, test its robustness on a second held-out language, and find the simplest change that beats **both** B0 and C1, or keep C1.

## Stage 0: preflight (verified)

- **B0** = `runs/traits/full` (WavLM all-layer mean 768 ⊕ eGeMAPS 88 → 856-d voice; FaRL 512 faces).
- **C1** = `runs/phase2/g3_cat_farl` (856 ⊕ ECAPA 192 = 1,048-d, `features/voice_v10_wavlm_egemaps_ecapa`), identical recipe.
  - Proxy runs: `runs/traits/p2_full`, `runs/phase2/p2_g3_cat_farl`.
  - Full-data model: `checkpoints/v5_ecapa_concat`; zip: `submissions/eval5_ecapa_concat_terms.zip`. Neither is touched.
- **Inputs:**
  - Both voice blocks are z-scored per dimension inside the model (training statistics), followed by input dropout 0.2 and noise 0.1.
  - ECAPA is the same encoder/revision for official, external and eval rows (official re-extraction: cos 0.99999994; external rows streamed from the zips, mapping verified on 60 clips at cos 0.9999995).
  - FaRL comes from `features/face_v2_farl` + `combined_all/train_faces.csv` (B0's).
- **Splits:**
  - Official folds: `validation/fold_k` (14 people each, fixed trial files).
  - Urdu proxy (P2): MAV-Celeb v1 people with sorted index % 5 = fold, Urdu excluded; checked inside the evaluator and by a unit test.
- **Language audit (manifests + VoxLingua LID of the VoxCeleb2 clips):**

  | Language | People | Rows | Holdout feasible? |
  |---|---|---|---|
  | English | 70 official + 113 external | 13,475 | (heard) |
  | Urdu | 64 (MAV-Celeb v1, also English) | 12,125 | **P2** (exists) |
  | German | 49 (MAV-Celeb v3; 48 also English) | 1,565 | **P3 (new):** ≈ 10 people / 300 German clips per fold |
  | Hindi | 0 | 0 | **no** |
  | VoxCeleb2 "Mixed" | 117 | 9,022 | LID top-1: en 7,239, cy 1,015 (English confusion), hi 75, ur 37, **de 5** |

  The German holdout is practically clean (5 German clips remain in VoxCeleb2). The Urdu holdout has ≤ 37 Urdu clips and 75 Hindi-labelled clips in VoxCeleb2, a negligible leak that is noted rather than fixed, to keep P2 identical to Phase II.
- **Resources:** 1 × RTX A5000, 20 CPU cores, 62 GB RAM. At least 5 GB free (5.2 GB test write OK). A fold run is 5.8 MB, so 15 runs ≈ 90 MB.

## Selection discipline

- **Development evidence** = the Urdu proxy (P2) and the official English folds. Both have guided Phase II.
- **Confirmation** = the new **German holdout (P3)**. It is run **only** for B0, C1 and at most one or two candidates chosen *before* looking at P3, and is never used for selection.
- **Screening:** seed 1 × 5 folds. **Claims:** 3 seeds × 5 folds, paired identity bootstrap. Number of configurations evaluated is reported in `07`.

## Experiment matrix

| ID | Question | Configurations (B0 recipe unless stated) | Protocols | Cost |
|---|---|---|---|---|
| E1a | What does ECAPA share with or add to WavLM ⊕ eGeMAPS? | frozen statistics: CKA, effective rank, norms, per-language shift; ECAPA residual after linear prediction from B0 features; identity-disjoint probes | features only | CPU |
| E1b | Is it just more input dimensions? | **D** B0 ⊕ fixed random N(0,1) 192-d per recording | official + P2, seed 1 | 10 runs |
| E1c | Does speaker-aligned ECAPA matter? | **E** C1 with ECAPA shuffled among *training* speakers (same gender and source); evaluation with true ECAPA | official + P2, seed 1 | 10 runs |
| E2 | Where to fuse? | **F2** ECAPA → bottleneck k ∈ {32, 64} then concat; **F3** per-block L2 (equal block weight); **F4** residual adapter h = h_base + α·adapter(ECAPA), α = 0 at start; **F5** late fusion B0 + ECAPA expert (exists, Phase II) | official + P2, seed 1; then 3 seeds for the best | 20 runs + 30 |
| E3 | Beyond Urdu? | B0, C1, best E2 candidate on **P3 German holdout** | P3, 3 seeds | 45 runs |
| E4 | Duration / session dependence | per-trial scores of B0 vs C1 on official + P2, duration and same/other-video strata, identity bootstrap | scores only | CPU |
| E5 | Do trait losses still help after ECAPA? | C1 trait weight 0 / 0.1 (vs 0.2) | official + P2, seed 1 | 20 runs |
| E6 | Stabilisation | C1 × 3 seeds ensemble, B0 + C1 mean / logistic fusion, terms with / without | scores only (+ P3) | CPU |

- **Total new training:** ≈ 135–165 runs, about 0.9 GB.
- **Not planned:**
  - F6 gated fusion: only if E4 shows a stable duration/quality interaction.
  - Uncertainty-weighted traits: no per-label reliability estimates exist.
  - Noise/reverberation corruption: no corruption set; it would need audio re-extraction.

## Decision gates

1. **After E1:**
   - If D (random) ≈ C1, the gain is dimensionality → stop investigating ECAPA content.
   - If E (shuffled) ≈ C1, speaker alignment doesn't matter → treat ECAPA as a regulariser.
   - Otherwise ECAPA content matters → E2.
2. **After E2 screening:** promote at most two variants that beat C1 on P2 Urdu *and* are not worse on the official folds (seed 1). If none, C1 stays the fusion of record.
3. **After 3-seed E2:** freeze candidates, then run P3 once.
4. **E4 → F6:** adaptive weighting only if B0-vs-C1 differences vary significantly across duration bins under the identity bootstrap.
5. **Final:** a candidate replaces C1 only if it is better on P2 and P3 unseen-language same-gender EER without significant official-fold regressions. Otherwise recommend C1 (or B0 if P3 contradicts C1).
