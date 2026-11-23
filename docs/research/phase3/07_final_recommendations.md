# Phase III — 07 Final recommendations (2026-11-19)

**Bottom line:**
- **Keep C1's architecture** (plain concatenation of frozen ECAPA with WavLM ⊕ eGeMAPS). No fusion variant beat it.
- **Ship it as a 3-seed ensemble with refitted terms** (`submissions/eval6_ecapa_concat_x3_terms.zip`), the simplest change that is reproducibly better than C1.
- C1's unheard-language gain is now **confirmed on a second held-out language (German)** that was never used for selection, and it is explained: it comes from **speaker-aligned, language-stable identity information** in ECAPA, not from extra dimensions, regularisation, face traits, clip duration or recording session.

## 1. Research questions: answers

| RQ | Answer | Evidence |
|---|---|---|
| 1. What does ECAPA add? | language-stable speaker identity that B0's input lacks (R² of ECAPA from B0 = 0.18; the ECAPA residual still verifies speakers at 14% EER in both languages) | `01` |
| 2. Is concatenation best? | yes: bottleneck, block normalisation and residual adapter tie or lose on the unheard language; late fusion with B0 hurts it | `02` |
| 3. Beyond Urdu? | yes: German gender −2.35 [−4.38, −0.30] vs B0 (confirmation holdout) | `03` |
| 4. Duration / session dependence? | none detected; the gain holds across duration bins and for other-video pairs; no basis for gated fusion | `04` |
| 5. Attribute supervision? | adds ≈ 0.1 (trait heads) / ≈ 0.5 (score terms, not double-counted); no new attribute model justified | `05` |
| 6. A more robust candidate? | the C1 × 3 seed ensemble: official −0.66 vs C1, Urdu −0.33 / −0.38 (significant), German −0.2 (n.s.) | `06` |

## 2. Configurations implemented

- **Training ablations** (B0 recipe, one change each):
  - controls D (random 192-d) and E (speaker-shuffled ECAPA);
  - F2 bottleneck 32 / 64; F3 per-block L2; F4 residual adapter;
  - E5 trait weight 0 / 0.1;
  - German-holdout runs of B0, C1, F3.
- **Score level:** seed ensembles (×2, ×3), C1 + B0 mean / logistic fusion, terms on single and ensemble models.
- **Analyses:** E1 frozen statistics (CKA, effective rank, redundancy R², ECAPA residual verification, probes), E4 duration/session strata with identity bootstrap.
- **Not built:** F6 gated fusion (E4 found no interaction), uncertainty-weighted traits (no reliability labels), noise/reverberation corruption (no corruption set), Hindi (no data).

## 3. Tests

83 tests pass:
- 11 Phase III: fusion shapes, norms and initial residual behaviour, block-L2 weighting, invalid split, shuffle-control integrity, cache column ordering, ensemble alignment, generator ensembling and independence, reserved names.
- 14 Phase II, 33 GAMP-VF, 25 original.

**Regression checks:**
- the trainer reproduces B0's and C1's stored logs bit-for-bit;
- the comparison tool, the generalised transfer evaluator and the multi-checkpoint generator reproduce stored outputs exactly.

## 4. Consolidated results (3 seeds × 5 folds unless noted; EER %)

| System | Official std / gender | Official + terms (mean) | Urdu std / gender (unheard) | German std / gender (unheard) |
|---|---|---|---|---|
| B0 | 20.34 / 31.97 | 25.94 | 26.06 / 38.86 | 30.37 / 42.14 |
| **C1** (eval #5 zip) | 20.48 / 31.94 | 25.73 | 23.94 / 36.32 | 28.99 / 39.73 |
| C1 F3 block L2 | 20.16 / 31.39 | — | 23.99 / 36.22 | 29.44 / 40.10 |
| C1 F2 bottleneck 64 | 20.20 / 31.38 | — | 24.21 / 36.42 | not run |
| **C1 × 3 seeds** (eval #6 zip) | **19.91 / 31.19** | **25.32** | **23.62 / 35.88** | **28.85 / 39.38** |
| C1 + B0 fusion | 19.96 / 31.22 | — | 24.56 / 37.20 | 29.17 / 40.72 |

## 5. Negative findings

- The random-feature and shuffled-ECAPA controls lose the entire unheard-language gain.
- Bottleneck k = 32 is worse than C1 everywhere.
- The residual adapter ties.
- F2 k = 64 and F3 look better on English at seed 1, but the gain shrinks to −0.4 n.s. over 3 seeds and comes with a significant loss on the proxies' English people.
- Mixing B0 into C1 hurts both unheard languages.
- Removing trait losses costs ≈ 0.1–0.2.
- The C1-vs-B0 gain on the official English folds is not significant (+0.05; −0.21 with terms).

## 6. Complexity and inference

- C1 and C1 × 3 need the same frozen encoders as B0, plus ECAPA (one extra frozen forward pass per clip; already extracted for all eval clips in `features/eval_ecapa`).
- Each model is ≈ 0.30 M trainable parameters in the voice branch; the ensemble triples only that.
- Every trial is still scored independently: an equal-weight mean of three cosines plus the fixed terms.

## 7. Recommendation

1. **Next submission: `submissions/eval6_ecapa_concat_x3_terms.zip`** (md5 `4b6fdcd4…`).
   - Expected, relative to C1 alone: about −0.3 to −0.6 everywhere.
   - Relative to B0: about −0.6 on official English (25.32 vs 25.94 with terms), and about −2.8 to −3.0 on unheard-language same-gender cells (Urdu 35.88 vs 38.86, German 39.38 vs 42.14) *if* Bangla behaves like Urdu and German.
   - If only one submission remains, prefer it over eval #5. It contains eval #5's model as one of its three members.
2. **Keep C1's architecture.** Don't adopt F2–F6.
3. **Do not submit anything automatically.** Both zips are validated and waiting for your decision.

## 8. Unresolved questions

- Why the official English folds barely benefit while both proxies' English cells (other people) do.
- Whether other language-stable speaker encoders (the x-vector, or several speaker encoders together) add more on top of ECAPA.
- Whether Bangla, linguistically closer to Hindi/Urdu than German is, behaves like the proxies.
- Noise and channel robustness: not measured.
