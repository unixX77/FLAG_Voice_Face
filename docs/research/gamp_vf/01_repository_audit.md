# GAMP-VF — 01 Repository and data audit (2026-11-18)

Everything below was checked against the files on disk on 2026-11-18; nothing is taken from memory alone.

## 1. Entry points and architecture (verified)

| Component | File | Notes |
|---|---|---|
| Fold training | `FLAG_2027/train_cross_modal.py` | one identity-disjoint fold; `--extra-*` appends external identities to the *training* side only |
| Full-data training | `FLAG_2027/train_cross_modal_full.py` | fixed epochs, no validation data; now has the same auxiliary options |
| Model | `FLAG_2027/cross_modal_model.py` → `CrossModalIdentityNetwork` | two MLP projectors (face 512→256→128, voice 856→256→128), input standardisation (training statistics), input dropout/noise, L2-normalised 128-d shared space, cosine score |
| Losses | `FLAG_2027/local_validation/cross_modal_losses.py` | multi-positive InfoNCE / SupCon (same-identity candidates are positives, never negatives), pair alignment, same-gender SupCon, same-gender hard-negative hinge, OPL |
| Auxiliary heads | `FLAG_2027/trait_heads.py` | shared trait heads (age/build/gender), language adversary (GRL), cross-lingual alignment; training-only |
| Batch sampler | `FLAG_2027/local_validation/identity_sampler.py` | P×K identity-balanced; optional gender-balanced and official/external share mixing; **identities drawn uniformly at random** |
| Paired comparison | `FLAG_2027/local_validation/compare_fold_runs.py` | 3 seeds × 5 folds, identity-level cluster bootstrap, B−A with 95% CI |
| Attribute terms | `FLAG_2027/analysis/finalize_attribute_terms.py` | fixed age/gender score terms fitted on fold models (training-side) |
| Submission | `FLAG_2027/generate_cross_modal_submission.py` | scores each trial independently; validated zip |
| Unheard-language proxy | `FLAG_2027/local_validation/evaluate_language_transfer.py` + `--holdout-external-prefix mav1 --exclude-train-languages Urdu` | Urdu never trained on; deterministic trials (bug fixed 2026-11-16) |

**Current strongest reproducible model (B0 / teacher):** `full` = V2 projectors + all 231 external identities (`--official-share 0.5`) + WavLM⊕eGeMAPS voice input (856-d) + trait heads (0.2) + language adversary/cross-lingual (0.1).
- 15 fold checkpoints: `runs/traits/full/fold_{0..4}_seed_{1..3}/checkpoint_last.pth.tar`, all 90 epochs.
- 3-seed five-fold EER, no terms: standard 20.34, gender 31.97, mean 26.15. With terms: 19.97 / 31.92 / 25.94.
- These numbers come from `runs/traits/compare_all_share50_vs_full.json` and `full_attribute_terms.json`, computed two days ago on the same code and data, so they are reproducible as-is.

## 2. Encoders and features (frozen; cached as CSV)

| Modality | Features | Dim | Path |
|---|---|---|---|
| Face | FaRL ViT-B/16 image embedding | 512 | `features/face_v2_farl/train_English_faces.csv`; external `features/external/combined_all/train_faces.csv` |
| Voice | WavLM-Base+ all-layer mean (768) ⊕ eGeMAPS v02 functionals (88) | 856 | `features/voice_v8_wavlm_egemaps/…`; external `train_voices_egemaps.csv` |
| Voice (WavLM only) | | 768 | `features/voice_v4_wavlm_allmean/…`; external `train_voices.csv` |
| Traits | InsightFace age, FaRL zero-shot build (per row) | 2 | `features/traits/{train_English,combined_all}_traits.csv`; 0 missing external rows |

ECAPA speaker embeddings exist **only for the official rows** (`features/voice_v6_ecapa`), not for external data.

## 3. Data (computed from the manifests)

| | Official (`train_set/train_set/train_English.txt`) | External (`features/external/combined_all/train_manifest.txt`) |
|---|---|---|
| Rows (face–voice pairs) | 6,485 | 29,702 |
| Identities | 70 | 231 (MAV-Celeb v1 64, MAV-Celeb v3 50, VoxCeleb2-test 117) |
| Gender | 30 f / 40 m, no missing | 79 f / 152 m, no missing |
| Rows by language | English 6,485 | English 6,990 · Urdu 12,125 · German 1,565 · **Mixed** 9,022 |
| Identity language sets | English only (70) | English+Urdu 64 · English+German 48 · English+German+Mixed 1 · German 1 · Mixed 117 |
| Rows per identity | median 92 (2–226) | median 80 (18–753) |
| Unique faces / voices per identity | median 92 / 92 (min 2) | median 69 / 80 (min 13 / 18) |

- **Hindi is not in the data.** There is no Hindi row anywhere, and MAV-Celeb v2 (English+Hindi) was never downloaded. All Hindi analyses are infeasible.
- **"Mixed"** = VoxCeleb2 rows. Their language is not annotated, so they are excluded from the language losses (language id −1).
- **Language comes from the manifest path** (`voices/<Language>/…`). It was assigned when the external data was built, per source and checked with a language-ID report (`features/external/*_train_language.csv.report.json`), so it is not a guess from file names.
- **Cross-language identity overlap:** 112 external identities have recordings in two languages, always English + Urdu or English + German. Every official identity is English-only.
- **Gender** comes from metadata (`meta_file_train_set.csv`, `combined_all/gender.csv`), with one documented voice/metadata disagreement (`vox2_id02286`). No unknowns.

## 4. True face–voice associations

Every row pairs a voice clip with a face from the same identity: same video for MAV-Celeb/VoxCeleb, official pairing for the challenge data. **All 301 identities are fully labeled. There is no unlabeled pool of candidate associations.**

Consequences:
- Pseudo-label mining (Stage E), global matching (F) and teacher–student refinement (G) **cannot add new supervision**. They would only re-derive labels we already have.
- As your spec asks, they are run as a **controlled hidden-association experiment**: about a third of the *external training* identities have their face–voice link hidden. Voices and faces of those people are pooled without identity labels. The pipeline must recover the links, and recovery is scored against the hidden truth.
  - Upper bound: the same model trained with the true labels, i.e. B0.
  - Lower bound: the model trained without those identities at all.
- **One-to-one assignment** is justified *at the identity level inside this simulation*: each hidden person has exactly one voice set and one face set. It is **not** justified at the clip or cluster level, since clustering can split or merge people. Matching therefore uses partial assignment with dummy (unmatched) nodes.
- **The eval data is out of bounds.** The challenge rules (`progress_readme.md` §8, official FAQ 13 Nov) forbid any use of evaluation data for pseudo-labels, clustering or calibration, all score normalisation, and cross-trial processing. Every GAMP-VF grouping and matching step therefore runs on **training-partition data only**, and the final verifier scores each trial independently (Stage H).

## 5. Validation protocol and leakage safeguards (verified)

- `validation/fold_{0..4}`: 5 identity-disjoint folds of the 70 official identities, with 56 train / 14 held-out (8 m / 6 f) per fold. Index files are `train_indices.npy` and `validation_indices.npy`.
- Each fold has fixed trial files `standard_pairs.csv` / `gender_pairs.csv`, 1,400 trials each (3 genuine + 3 impostor faces per voice clip), plus ground truth. The trials are **pre-generated on disk, so they are deterministic** across models.
- External identities only ever go into training (`append_external`). Name overlap with official identities raises an error, and near-duplicate faces across sources were removed when the external data was built (`combined_all/report.json`).
- Input standardisation and trait z-scoring use the fold's training rows only.
- Results use `checkpoint_last` at a fixed 90 epochs, never best-on-validation.
- **The validation trials are English only.** Per-language EER on the official folds is therefore not possible. Cross-lingual questions use the Urdu-holdout proxy (P2).

## 6. Compute and disk (checked 2026-11-18)

- **GPU:** 1× RTX A5000 (24 GB), idle. **CPU:** 20 cores. **RAM:** 62 GB (59 GB available).
- **Disk:** home quota about 100 GiB, **98 GB used, about 1.5–2 GB free** (a 1.5 GB test write succeeded).
  - `/export` (910 GB free) is **not writable** for this account.
  - `/dev/shm` is RAM-backed and was wiped by today's reboot; it is never used for results.
- **Run cost:** one fold run is a 5.8 MB checkpoint plus a 35 KB log; 15 runs ≈ 87 MB. Runs use `--no-best-checkpoint`. The budget is roughly ≤ 10 three-seed configurations before space runs out. Pseudo-label tables are small CSVs.
- The external audio work folders (12.2 GB, rebuildable) are still waiting for the user's OK to delete.

## 7. Feasibility of the requested stages

| Stage / experiment | Feasible? | How |
|---|---|---|
| A1–A3 speaker prototypes (known IDs) | ✅ | in-batch identity prototypes; language-balanced version for the 112 bilingual external identities |
| A unsupervised speaker clustering + ARI/NMI | ✅ | training-side only, on WavLM and on projected embeddings; HDBSCAN via `sklearn.cluster.HDBSCAN` if available |
| A cross-language same-speaker diagnostic | ✅ | English vs Urdu/German clips of the same external identity |
| B face prototypes | ✅ | |
| C gender-aware hard-negative identity sampling | ✅ | trait age available for the age-similar variant |
| D similarity matrix, top-K, reciprocal NN | ✅ | |
| E pseudo-label mining | ⚠️ simulation only | hidden-association experiment (§4) |
| F Hungarian / Sinkhorn | ⚠️ simulation only | partial assignment with dummy nodes |
| G teacher–student | ⚠️ simulation only | 0/1/2(/3) iterations as long as diagnostics justify continuing |
| B8 EMA teacher | ❌ skipped | optional; only adds cost |
| Hindi analyses | ❌ | no Hindi data |
| English→Urdu held-out language | ✅ | existing P2 proxy |
| English→German held-out language | ⚠️ weak | only 49 German identities and 1,565 rows; possible but low power; run only if a winner needs it |
| Per-language EER on official folds | ❌ | official validation is English only |

## 8. Unverified assumptions

- Clip-level language labels inside an identity are taken as given (the language-ID report flagged and dropped 151 rows).
- External gender comes from dataset metadata. One identity disagrees with the frozen voice-gender model.
- Hidden-association results transfer only to settings where voice and face pools exist without links. Our real data has no such pool.

## 9. Files reused or modified

- **Reused unchanged:** model, losses, fold data, comparison tool, attribute-term fitting, submission generator.
- **Modified (option-gated, defaults unchanged):** `train_cross_modal.py` (new sampler options, prototype loss, pseudo-pair input), `identity_sampler.py` (hard-negative identity selection, language-balanced K, identity weights).
- **New:** package `FLAG_2027/gamp_vf/` with the prototype, sampling, similarity, clustering, matching and pseudo-label modules, tests, and run scripts.
- **Results:** `results/gamp_vf/stage_*`, kept per stage and never overwritten.
- **Runs:** `runs/gamp_vf/<config>/fold_k_seed_s`.
