# GAMP-VF — 04 Results and recommendation (2026-11-18)

**Bottom line:**
- No GAMP-VF configuration is significantly better than B0 (`full`).
- The only consistent effect is **same-gender partner sampling** (`b3_sg`). It lowers same-gender EER on the official folds by −0.41 (95% CI [−1.01, +0.31]; better on 5/5 folds; 3 seeds). On the unheard-language proxy, however, the gain appears **only in the heard language**, and the heard→unheard gender gap widens by +0.53.
- **Prototype losses hurt slightly.** Pseudo-labelling, global matching and teacher–student refinement **cannot help with this data**: there is no unlabelled pool; in the hidden-association simulation even oracle grouping gives under 60% precision; and the hidden people's true labels themselves bring no measurable gain.
- **Recommendation: keep B0 as the competition model.** `b3_sg` may be added as an option, but only after a 3-seed check of it together with the attribute terms. Do not submit GAMP-VF as a new system.

All EERs are in %, from `compare_fold_runs.py` (paired identity-level bootstrap, 2,000 resamples). "Folds" = folds on which B beats A (standard / gender). No attribute terms unless stated.

## 1. Verification: official five-fold (English), against B0

| ID | Config | Seeds | Standard | Gender | Mean Δ [95% CI] | Gender Δ [95% CI] | P(better) mean / gender | Folds |
|---|---|---|---|---|---|---|---|---|
| B0 | `full` (teacher/reference) | 1–3 | 20.34 | 31.97 | — | — | — | — |
| B0 | `full` | 1 | 20.27 | 31.59 | — | — | — | — |
| B1 | + prototype InfoNCE (0.5) | 1 | 20.86 | 32.20 | **+0.60** [−0.13, +0.80] | +0.61 | 0.06 / 0.15 | 1/5, 1/5 |
| B2 | + language-balanced prototypes + language-balanced K | 1 | 20.41 | 31.86 | +0.21 [−0.78, +0.84] | +0.27 | 0.41 / 0.44 | 3/5, 1/5 |
| B3 | same-gender partner (ratio 0.5) | 1 | 20.39 | 31.19 | −0.14 [−0.77, +0.49] | −0.40 | 0.65 / 0.67 | 2/5, 4/5 |
| B3 | **same-gender partner (ratio 0.5)** | **1–3** | **20.27** | **31.55** | **−0.25 [−0.65, +0.22]** | **−0.41 [−1.01, +0.31]** | **0.81 / 0.82** | **3/5, 5/5** |
| B3 | most-confusable partner | 1 | 20.23 | 31.76 | +0.06 [−0.66, +0.55] | +0.17 | 0.57 / 0.51 | 3/5, 3/5 |
| B3 | same-gender most-confusable | 1 | 20.21 | 31.64 | −0.00 [−0.60, +0.58] | +0.06 | 0.52 / 0.55 | 2/5, 3/5 |
| B3 | same-gender most-confusable, ratio 1.0 | 1 | 20.54 | 31.93 | +0.31 [−0.52, +0.82] | +0.34 | 0.30 / 0.40 | 1/5, 1/5 |
| B3 | same-gender similar-age partner | 1 | 20.41 | 31.31 | −0.06 [−0.82, +0.51] | −0.27 | 0.62 / 0.63 | 2/5, 3/5 |
| B3 | same-gender similar-age partner | 1–3 | 20.43 | 31.92 | +0.02 [−0.31, +0.51] | −0.05 [−0.47, +0.68] | 0.28 / 0.37 | 2/5, 3/5 |
| B4 | B2 + same-gender partner | 1 | 20.34 | 31.37 | −0.07 [−0.82, +0.66] | −0.21 | 0.56 / 0.63 | 1/5, 3/5 |
| B5–B8 | pseudo-labels / matching / teacher–student | — | — | — | not applicable to real training: no unlabelled pool (see §3) | | | |

**Score behaviour** (`stage_C_final/diagnostics_3seeds.json`; same-gender trials, 3 seeds):

| | AUC | d′ | violation rate* | FRR at FAR 5% | genuine / impostor mean cosine |
|---|---|---|---|---|---|
| B0 | 0.741 | 0.934 | 0.358 | 0.837 | 0.677 / 0.542 |
| `b3_sg` | 0.743 | 0.945 | 0.358 | 0.835 | — |
| `b3_sga` | 0.741 | 0.937 | 0.358 | 0.835 | — |

\*Share of voice clips whose best same-gender impostor face outranks their worst genuine face.

The gender-track problem is unchanged: genuine and same-gender impostor scores overlap almost entirely (`stage_C_final/score_distributions_3seeds.png`).

**Readings:**
- Composing batches of same-gender pairs is the only hard-negative scheme with a consistent sign.
- Similarity-mined "most confusable" partners do nothing, and a 100% hard ratio is worse. Too-hard batches are known to destabilise contrastive training, and that matches what we see.
- The prototype loss compresses both score distributions without separating them.

## 2. Cross-lingual generalisation (Urdu-holdout proxy; 3 seeds; `cross_lingual/`)

| | Std English | Std Urdu (unheard) | Std gap | Gender English | Gender Urdu (unheard) | Gender gap |
|---|---|---|---|---|---|---|
| B0 (`p2_full`) | 24.11 | 25.89 | +1.78 | 37.30 | 38.71 | +1.41 |
| `b3_sg` (P2) | 23.94 | 25.83 | +1.89 | **36.82** | 38.76 | **+1.94** |
| Δ [95% CI] | −0.17 [−0.48, +0.19] | −0.06 [−0.26, +0.21] | +0.11 | **−0.48 [−0.93, +0.07]**, P = 0.95 | +0.05 [−0.34, +0.39] | **+0.53 [−0.01, +0.93]** (P narrower = 0.03) |

The same-gender gain is **specific to languages heard in training**. For the challenge, that means the English cells might gain about 0.4 while the Bangla cells would gain nothing. The expected change in the 4-cell mean is about −0.1 to −0.2, which is too small to justify a submission.

**Ablation on unseen people** (`stage_A_B/`; held-out MAV-Celeb v1, Urdu never trained on; seed 1; n = 64 people, so ±3 people ≈ ±5% top-1 is noise):

| Voice representation | Same speaker, other language (cos) | Other speaker, same language (cos) | AUC (same-speaker cross-language vs other-speaker same-language) | Voice → face top-1 among ~13 people |
|---|---|---|---|---|
| Raw WavLM (no training) | 0.775 | 0.764 | 0.545 | — |
| `p2_base` (no language losses) | 0.701 | 0.468 | 0.793 | recording 32%, prototype 42% |
| `p2_lang` (+ adversary + cross-lingual) | 0.724 | 0.488 | 0.796 | 32%, 44% |
| `p2_full` (+ eGeMAPS + traits) | 0.725 | 0.478 | 0.799 | 33%, uniform 42%, **language-balanced 47%**, English-only 47%, Urdu-only 34% |

- **Speaker aggregation helps association** (single clip → prototype: +9–14 points top-1). Language-balanced aggregation may help a little more (+5, within noise).
- Explicit language-invariance training **does not** improve speaker discrimination. It moves same-speaker cross-language pairs and other-speaker pairs closer by the same amount. It doesn't remove speaker information either.
- English-only prototypes beat Urdu-only ones (47 vs 34): the unheard language carries less usable identity information through this model.

## 3. Association recovery: hidden-association simulation (Stages E–G; seed 1; `stage_E/`, `stage_F/`, `stage_G/`)

H = 77 hidden external people; C = 39 calibration people. The teacher never saw either.

**Grouping without labels:**

| | ARI | Purity | Groups per person |
|---|---|---|---|
| H voices (teacher space) | 0.25–0.28 | 0.51–0.52 | ≈ 7.5 |
| H faces (raw FaRL) | 0.51 | 0.73 | ≈ 3.1 |
| C voices / faces | 0.37–0.41 / 0.76 | 0.61–0.69 / 0.88 | ≈ 7.5 / 2.3 |

**Matching, mean over folds** (precision = share of accepted voice-group → face-group pairs that are the same person):

| Method | H accepted pairs | H precision | H row precision | H identity recall |
|---|---|---|---|---|
| Pairwise top-1 (no filter) | 168 | 0.16 | 0.15 | 0.17 |
| F1 local, confidence ≥ 0.9-precision threshold (from C) | 0–6 per fold, 9 in total | 2 of 9 correct (0.22) | — | ≤ 0.01 |
| F2 Hungarian (no setting reached 0.9 on C → strict fallback) | 61 | 0.17 | 0.17 | 0.13 |
| F3 Sinkhorn | 0 | — | — | 0 |
| B6 local + Hungarian on ambiguous | 3–16 per fold, 39 in total | 9 of 39 correct (0.23) | — | ≤ 0.05 |
| **Oracle groups** (true identities), top-1 | 77 | 0.24 | — | — |
| **Oracle groups**, Hungarian one-to-one | 77 | 0.34 | — | — |
| **Oracle groups**, 10% most confident | 8 | 0.57 | — | — |

- **Global matching helps:** Hungarian beats independent top-1 by +10 points with oracle groups. But it starts from a base far too low for training labels.
- **No method reaches usable pseudo-label precision**, even with perfect grouping. The limit is the cross-modal signal itself (same-gender AUC about 0.74), not the clustering.
- Calibration on C (39 people) overestimated precision on H (77 people), because a larger pool is harder. Any future pseudo-labelling must calibrate on a pool of the **same size**.
- Removing 30% of the face groups hardly changed precision, which was already near the floor (`summary.json`, `H_with_30pct_faces_missing`).

**Teacher–student (seed 1; mean EER Δ against the teacher):**

| | Pseudo rows (folds 0–4) | Mean Δ [95% CI] |
|---|---|---|
| Oracle (true H labels) | 10,516 true rows | +0.58 [−0.66, +1.97] |
| Student it1 (high-precision mode) | 0 / 588 / 600 / 200 / 100 | +0.74 [−0.12, +2.00] |
| Student it1hung (noisy, ≈ 17% precision) | ≈ 4,000 per fold | +0.59 [−0.45, +1.76] |

- All three sit within single-seed run noise. Fold 0 of it1 had **zero** pseudo rows and still moved +0.75, i.e. only the batch order changed.
- So this shows neither harm nor benefit. The oracle row shows the H identities carry no measurable gain on top of the other 171 training people. **The simulation has no power to show a pseudo-label benefit, so iterations 2–3 and the EMA teacher were not run.**

## 4. Status of components

| Component | Status |
|---|---|
| Prototypes A1–A3, face prototypes, in-batch prototype loss | implemented, unit-tested, evaluated (B1/B2: no gain) |
| Clustering (agglomerative, HDBSCAN) + metrics | implemented, tested, evaluated (voices ARI ≤ 0.4) |
| Gender-aware hard-negative sampler (5 strategies, ratio, conflict groups, unknown gender, language-balanced K) | implemented, tested, evaluated (B3/B4) |
| Similarity matrix, top-K, reciprocal NN, margins | implemented, tested, used in mining |
| Confidence-calibrated pseudo-label miner with provenance | implemented, tested (synthetic end-to-end), evaluated in simulation |
| Hungarian / Sinkhorn with unmatched nodes | implemented, tested, evaluated in simulation |
| Teacher–student loop | one iteration run. More were not justified; EMA teacher not implemented |
| Stage H verifier | unchanged network; all checkpoints load in the existing comparison and submission tools |
| Ensemble agreement in Stage D | not implemented (would need several teachers per fold) |

## 5. Failure cases and caveats

- Same-gender EER is still about 32%. Most errors are same-gender, similar-age pairs (§6.7 of the progress log), and the similar-age sampler did not change that.
- Every B1–B4 conclusion except B3 `same_gender` / `same_gender_age` rests on a single seed. Earlier projects saw seed-1 effects halve with more seeds.
- The simulation pools are small (39 and 77 people), and calibration did not transfer between pool sizes.
- Hindi and German held-out-language tests were not possible or not powered.
