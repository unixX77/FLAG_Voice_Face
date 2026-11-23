# GAMP-VF — 02 Architecture specification

GAMP-VF adds **training-side** components to the existing verifier. The verifier itself is unchanged: `CrossModalIdentityNetwork` maps one face and one voice into a shared 128-d unit sphere and scores the pair by cosine, plus the fixed age/gender terms of the current submission. It needs no gallery, no gender or language label, and no other trial (Stage H).

```
training partition only
  official fold-train rows + external rows (labelled)  ──┐
  hidden pool H (simulation: unlabelled voices, faces) ─┐│
                                                        ││
  teacher (B0 `full`, or the previous student) ─► Stage D similarity ─► Stage A/B groups ─► Stage E candidates
                                                        │                                    │ calibrated on C
                                                        │                                    ▼
                                                        │                     Stage F matching (local / Hungarian / Sinkhorn)
                                                        │                                    │
                                                        ▼                                    ▼
                      Stage C sampler (gender-aware hard negatives) ◄── pseudo identities (weights, conflict groups)
                                                        │
                                                        ▼
                       student = same network, SupCon + pair + traits + language (+ prototype loss)
                                                        │
evaluation          ────────────────────────────────────▼─────────────────────────────
  one face + one voice ─► face/voice projectors ─► cosine (+ fixed terms) ─► score   (Stage H)
```

## Modules (`FLAG_2027/gamp_vf/`)

| Module | Stage | Contents |
|---|---|---|
| `prototypes.py` | A, B | `identity_prototypes`: A2 uniform, A3 language-balanced (normalised per-language means, then averaged), optional weights, quality = mean member–prototype cosine. `batch_prototypes`: differentiable in-batch version. `prototype_contrastive_loss`: face → voice-identity prototypes and voice → face-identity prototypes InfoNCE; one prototype per identity, so same-identity samples are never negatives. `speaker_similarity_statistics`: same/different speaker × same/different language. |
| `clustering.py` | A, B | agglomerative (cosine, average linkage, distance threshold, small clusters → noise −1), HDBSCAN, metrics (purity, NMI, ARI with noise as singletons, fragmentation, coverage), threshold tuning on *labelled training-side* data |
| `sampling.py` | C | `GenderAwareHardNegativeSampler`: P/2 anchors + 1 partner each. Strategies: `random`, `same_gender`, `nearest`, `same_gender_nearest`, `same_gender_age`. Mixture ratio `hard_ratio`; exact official/external share; language-balanced K; per-identity weights (pseudo-label confidence); conflict groups (never in one batch); unknown gender never satisfies a same-gender constraint. `identity_cross_modal_similarity` = ½(cos(vᵢ, fⱼ) + cos(fᵢ, vⱼ)). |
| `similarity.py` | D | chunked GPU/CPU cosine matrix, top-K both directions, reciprocal top-1, rank margin. Outputs are always **raw cosine**. |
| `pseudo_labels.py` | E | group prototypes, gender compatibility (from the teacher's trait-head gender), candidate features (score, margin, reciprocal, cross-language consistency, sizes), `ConfidenceCalibrator` (logistic, fitted on C), precision-targeted thresholds, high/ambiguous/unreliable decisions, random face × voice row pairing, conflict groups (union-find over close prototypes) |
| `matching.py` | F | F1 `local_selection`, F2 `hungarian_partial` (dummy nodes, rectangular, threshold), F3 `sinkhorn_partial` / `sinkhorn_match` (log domain, dustbins; weights are assignment weights, **not** probabilities) |
| `mine_pseudo_labels.py` | E + F | per-fold pipeline and diagnostics (below) |
| `stage_ab_diagnostics.py` | A, B | prototype/clustering/cross-language diagnostics on unseen people |
| `evaluate_gamp.py` | eval | AUC, FAR/FRR, same-gender score distributions, d′, violation rate, histograms |
| `test_gamp_vf.py`, `test_pseudo_pipeline.py` | — | 32 tests, including a synthetic end-to-end recovery test and leakage guards |

## Training integration (`FLAG_2027/train_cross_modal.py`; every option off by default, bit-identical to B0 when off)

| Option | Effect |
|---|---|
| `--prototype-contrastive-weight w` | adds w × prototype InfoNCE (Stages A/B, B1) |
| `--prototype-language-balanced` | A3 voice prototypes inside that loss (B2) |
| `--language-balanced-k` | an identity's K samples are split evenly over its languages (B2) |
| `--batch-strategy S --hard-ratio r --hard-top-k k` | Stage C sampler. Hard negatives come from *batch composition*; no extra hard-negative loss, so they are not counted twice |
| `--hard-warmup e --hard-refresh-every n` | `nearest*` strategies: identity confusability recomputed from the current model on **training rows only** every n epochs after e |
| `--exclude-external-identities file` | removes hidden (H) / calibration (C) identities from labelled training |
| `--pseudo-pairs-csv file` | adds pseudo rows (face row, voice row from the hidden pool only; checked), pseudo identity, confidence weight → sampling weight, predicted gender (−1 = unknown), conflict group |

Unknown gender is masked in the trait-head gender loss. The older same-gender losses refuse unknown genders.

## Hidden-association simulation (Stages E–G)

All 301 identities are labelled (audit §4), so pseudo-labelling can only be studied by hiding labels we have:

- **H** = every 3rd external identity (77 people, 10,516 rows). Voice clips and face images are pooled *separately*, so the identity and the row alignment (same video) are both hidden.
- **C** = 39 further external identities (4,969 rows), labelled but never trained on by any simulation model. They are used to tune the clustering thresholds (ARI), fit the confidence calibrator and choose the Hungarian/Sinkhorn parameters at a **target precision of 0.9**. Everything is frozen before it is applied to H. Since no model ever trains on C, the calibration stays valid in later iterations.
- **Teacher / lower bound** (`sim_teacher`): B0 recipe without H ∪ C.
- **Upper bound / oracle** (`sim_oracle`): B0 recipe without C, with H's true labels.
- **Student** (`sim_student_itN`): the teacher recipe plus pseudo rows mined from H by the previous model. Iteration N+1 mines with student N.
- Mining per fold:
  1. Embed H and C with the fold's teacher.
  2. Cluster voices (projected) and faces (raw FaRL or projected, whichever has the better ARI on C).
  3. Build group prototypes (language-balanced for voices).
  4. Mask gender-incompatible pairs (teacher gender head; unknown = compatible).
  5. Compute candidate features → calibrated confidence → decisions.
  6. Run the four assignment modes: `local` (high-confidence reciprocal, one-to-one), `hungarian`, `sinkhorn`, `local_plus_hungarian` (B6: global matching only among ambiguous groups).
  7. The pseudo-pair mode is chosen by the most correct pairs on C at precision ≥ 0.9.
  8. Each accepted pair yields up to 100 rows, pairing random voice clips of the voice group with random face images of the face group.
- **Diagnostics** (truth used only here): pair precision, row precision (share of pseudo rows whose face and voice truly match), identity recall, coverage, unmatched groups, errors by gender and language, and a 30%-missing-faces sensitivity test.

**One-to-one assumption:** valid at the *identity* level here (each hidden person has one voice set and one face set). Clusters may split or merge people, so every matcher allows unmatched nodes and never forces a pair.

## Stage H: final verifier

Unchanged interface. A student checkpoint has the same architecture, so `generate_cross_modal_submission.py` and `compare_fold_runs.py` work as-is. Auxiliary heads stay training-only. Inference uses no gender, language, gallery or identity information, apart from the fixed frozen age/gender terms already in the submission pipeline, which degrade gracefully when absent.
