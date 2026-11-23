# FLAG 2027: Project Progress Log

> A detailed record of the task, the system, every experiment and every code change.
> **Last updated: 2026-11-13.** The newest entries are in the [Changelog](#10-changelog) at the bottom.

---

## Contents
1. [The task](#1-the-task)
2. [Status at a glance](#2-status-at-a-glance)
3. [Data](#3-data)
4. [Current best system: V2](#4-current-best-system-v2)
5. [How we validate](#5-how-we-validate)
6. [Experiment log](#6-experiment-log)
7. [Code and data changes, file by file](#7-code-and-data-changes-file-by-file)
8. [Challenge rules and compliance](#8-challenge-rules-and-compliance)
9. [Next steps](#9-next-steps)
10. [Changelog](#10-changelog)
11. [Key findings (interview summary)](#11-key-findings-interview-summary)

---

## 1. The task

**Challenge:** Face–voice Association across LAnguages and Gender (FLAG) 2027, an ICASSP 2027 Grand Challenge, hosted on CodaBench (competition 18062).

**Problem:** *cross-modal verification*. Given a face image and a voice clip, decide whether they belong to the same person. Test identities never appear in training ("unseen"), and the model is trained on English only.

**What is evaluated:** four protocol cells, two tracks × two conditions.

| | English test (*heard*: the training language) | Bengali test (*unheard*: never seen in training) |
|---|---|---|
| **Standard (`no_gender`)**: impostors can be any gender | cell 1 | cell 2 |
| **Gender-controlled (`gender`)**: impostors always share the speaker's gender | cell 3 | cell 4 |

The gender track tests whether a model matches *people* rather than just "male face ↔ male voice". It is much harder.

**Metric:** Equal Error Rate (EER), the error rate at the threshold where false accepts equal false rejects. **Lower is better.** The overall score is the mean EER of the four cells. Submissions contain one **distance** per trial (smaller = more likely the same person).

**Schedule (as published):**

| Phase | Dates | Notes |
|---|---|---|
| Progress (dev) | 15 Sep – 10 Nov 2026 | Dev leaderboard |
| Evaluation FLAG (phase 1) | 10/11 – 17 Nov 2026 (Codabench closes 17 Nov, 23:00 UTC) | Evaluation inputs, no labels |
| Evaluation FLAG 2 | 19 – 26 Nov 2026 | Second window; which phase is final is to be confirmed with the organisers |
| Results | 23 Nov 2026 (website) | Provisional until the code and compliance review of the top 5 teams |

**Deliverables:** a 2-page system description (ICASSP template) and a public code repository. Teams without them are disqualified.

---

## 2. Status at a glance

### Scores of the current best system (V2)

| Cell | FOP baseline, dev | **V2, dev** (13 Oct) | FOP baseline, eval | **V2, eval phase 1** (13 Nov) |
|---|---|---|---|---|
| Standard, English | 32.54 | **23.61** | 29.10 | **26.53** |
| Standard, Bengali | 38.12 | **23.19** | 32.90 | **25.19** |
| Gender, English | 32.99 | 35.85 | 39.4 | **36.54** |
| Gender, Bengali | 44.01 | **33.79** | 39.90 | **36.87** |
| **Overall (mean EER %)** | 36.92 | **29.11** | 35.32 | **31.28** |

**Evaluation phase 1, submission #2: V2 + bias-corrected age/gender terms: 30.17%.**
- Per cell: standard 24.49 (English) / 22.69 (Bengali); gender 36.06 / 37.43.
- Change from V2's 31.28%: −1.1 overall; standard −2.0 / −2.5; gender −0.5 / +0.6. This matches the local prediction (gains on standard only).

**Leaderboard context (public API, 13 Nov):**
- Evaluation phase 1 has 34 entries. The top is **14.30** (gender 12.0 / 24.3); the median is **22.47**; 14 entries are below 22%. The median gender-track cell is 28.2%.
- **Our 30.17 ranks 32nd of 34.**
- The progress (dev) leaderboard top entries are at **0.5–2% EER**, which is implausible without cross-trial or test-data exploitation. The organisers' FAQ now forbids such methods, and rankings stay provisional until the code review.

- **Dev:** 7.8 points better than the official baseline (21% relative).
- **Evaluation phase 1:** 4.0 points better than the baseline, and better in all four cells.
- **Language transfer works:** Bengali (never seen) is as good as English in both phases.
- **Bottleneck:** same-gender matching, at about 34–37% EER against about 23–27% on the standard track.
- The evaluation cells are small (196–401 true pairs each), so a single cell can move by about ±3 points by chance.

### Submitted
- Eval #1: V2, 31.28%. Eval #2: V2 + age/gender terms, **30.17%**.

### Running right now
- Nothing running. Layer attention finished: no gain ([§6.11](#611-wavlm-layer-attention--runslayer_attention-13-nov)).
- Finished: the hyperbolic experiment gives −1.0 to −1.3 points on the standard track (significant) and no change on gender ([§6.10](#610-hyperbolic-poincaré-embeddings--standard-track---gender-track))).

---

## 3. Data

All official data comes from **MAV-Celeb v4** (English–Bengali), distributed through the challenge Google Drive folder.

| Split | Location in project | Size | Used for |
|---|---|---|---|
| Training | `train_set/train_set/train_English.txt` + `voices/`, `faces/` | **70 identities, 6,485 face–voice clips**, English only | All model training and local validation |
| Gender metadata | `meta_file_train_set.csv` | 70 rows (`idNNN,m/f`): 40 men, 30 women | Fold balancing, gender-specific analyses |
| Dev (progress phase) | `dev_set/{gender,no_gender}/{English,Bangla}_test.txt` | 982–1,468 trials per cell | Dev leaderboard only |
| Evaluation phase 1 | `evaluation_phase/{gender,no_gender}/{English,Bangla}_valid.txt` | 392 / 802 / 416 / 716 trials | Evaluation submission only |

**Where the evaluation data came from:** it is not on CodaBench. It is `evaluation_phase.zip` (472 MB, added 31 Oct) in the same Google Drive folder as the training and dev sets. It was downloaded on 13 Nov with `gdown` and extracted to `evaluation_phase/`.

**Data problem found in the official evaluation files:** every line (all 416) of `evaluation_phase/gender/English_valid.txt` has a space instead of `/` in the face path, e.g. `English_valid/faces 00000.jpg`.
- The official file is left untouched.
- A corrected copy is in `features/eval_manifests/gender/English_valid.txt` (fields joined with `/`), with an explanation in `features/eval_manifests/README.md`.
- After the fix, every referenced file exists and every face number equals its voice number. Trial IDs and order are identical to the official file.
- Worth reporting on the CodaBench forum.

**Not used so far:** external datasets, and Bengali audio for training (forbidden by the rules; see [§8](#8-challenge-rules-and-compliance)).

---

## 4. Current best system: V2

Internally this is "config E" (`runs/ablation_regularized/E_allmean_reg_noSG`). Everything needed to reproduce it is in `checkpoints/v2/` (model, fold runs, submissions, README with commands).

### 4.1 Pipeline

```
face image ──► FaRL ViT-B/16 (frozen) ──► 512-d ──► face projector  ──► 128-d unit vector ─┐
                                                                                            ├─► cosine ──► distance = 1 − cos
voice clip ──► WavLM Base+ (frozen) ──► 768-d ──► voice projector ──► 128-d unit vector ───┘
```

Only the two projectors are trained (about 0.4M parameters). The pretrained encoders stay frozen.

### 4.2 Pretrained encoders (frozen feature extractors)

| | Face | Voice |
|---|---|---|
| Model | **FaRL** ViT-B/16, `FaRL-Base-Patch16-LAIONFace20M-ep16` (CLIP-style model pretrained on 20M face image–text pairs) | **WavLM Base+**, `microsoft/wavlm-base-plus-sv` backbone, pinned revision `feb593a6c23c1cc3d9510425c29b0a14d2b07b1e` |
| Checkpoint file | `models/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth` (sha256 `67b4d6bc…`) | Hugging Face cache (offline, `HF_HUB_OFFLINE=1`) |
| Preprocessing | Official OpenAI CLIP ViT-B/16 preprocessing (resize 224, centre crop, normalise). No face detection or alignment. fp16. | 16 kHz mono PCM; 10 s chunks with 5 s overlap; frame-weighted average over chunks |
| Output used | Final image embedding, L2-normalised, **512-d** | Frame mean of all 13 hidden states (CNN output + 12 transformer layers), then **averaged over the 13 layers**, **768-d** |
| Feature files | train `features/face_v2_farl/`; dev `features/dev_farl_wavlm/`; eval `features/eval_v2/` | train `features/voice_v4_wavlm_allmean/`; dev `features/dev_v2/`; eval `features/eval_v2/` |

Why average all layers: the x-vector head of WavLM-sv keeps mostly speaker identity. The intermediate layers retain broader voice traits (pitch, timbre, age, vocal effort) that are more likely to be shared with the face. The all-layer average beat single layers and the x-vector in earlier ablations.

### 4.3 Model: `CrossModalIdentityNetwork` (`FLAG_2027/cross_modal_model.py`)

Each modality has its own **projector**:
1. **z-score standardisation** of the input, using training-set mean and std stored as buffers in the checkpoint;
2. **Gaussian input noise** σ = 0.1 and **input dropout** 0.2 (training only);
3. **Linear → 256, LayerNorm, GELU, Dropout 0.3**;
4. **Linear → 128**, then **L2 normalisation** onto the unit sphere.

Trainable parameters: **403,968** (full-data model). That includes an unused ArcFace classifier of 70 × 128, which is kept for checkpoint compatibility and has loss weight 0.

### 4.4 Training objective

`total = 1.0 × SupCon + 0.25 × pair loss` (identity loss weight 0; no orthogonal, prototype, hard-negative or same-gender losses).

- **Symmetric supervised contrastive loss (SupCon), temperature 0.07.** Within a batch, the cosine matrix between all faces and all voices is computed. Each face is pulled towards every voice of the same identity and pushed from all others, and symmetrically voice → face. This is the same idea as CLIP's alignment loss, but with several positives per identity.
- **Pair loss:** the mean of `1 − cos(face_i, voice_i)` over the row-aligned pairs (the face and voice of the same clip).

### 4.5 Hyperparameters

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-4 (constant) |
| Weight decay | 1e-4 |
| Gradient clipping | 5.0 |
| Batch | 16 identities × 8 samples = 128 (identity-balanced sampler) |
| Batches per epoch | 50 (full data) |
| Epochs | **90, fixed** (no early stopping for the final model) |
| Embedding size | 128 |
| Hidden size | 256 (one hidden layer per projector) |
| Hidden dropout | 0.3 |
| Input dropout / noise | 0.2 / σ = 0.1 |
| Contrastive temperature | 0.07 |
| Pair-loss weight | 0.25 |
| Seed | 1 |

**Why 90 epochs:** five-fold runs of 150 epochs without early stopping showed the mean validation EER flat between epochs 40 and 140, with its smoothed minimum at 86.

### 4.6 Final model and submissions

- Full-data model: `checkpoints/v2/full_data/checkpoint_full.pth.tar` (md5 `64cfdc7d9f0aebb5cc986fb6340a66d6`), trained on all 70 identities. It stores `model_kwargs`, so it can be rebuilt with `CrossModalIdentityNetwork(**ckpt["model_kwargs"])`.
- Dev submission: `submissions/v2.zip` (md5 `6517447d…`), scoring **29.11%**.
- Evaluation phase 1 submission: `submissions/eval1_v2.zip` (md5 `266b7a938f30e25300cd084866c95d29`), scoring **31.28%**.
- Scoring: each trial gets `1 − cos(face, voice)`, computed from that trial's own face and voice only.

---

## 5. How we validate

The leaderboard can't be used for tuning (rules, plus a limited number of submissions), so all decisions are made locally.

- **Folds:** 5 identity-disjoint folds of the training set (`validation/fold_0`–`fold_4`). Each trains on 56 identities (about 5,190 clips) and evaluates on the 14 held-out identities (about 1,300 clips). Every identity is held out exactly once.
- **Local protocols per fold:** `standard_pairs.csv` (any-gender impostors) and `gender_pairs.csv` (same-gender impostors), mirroring the two challenge tracks.
- **Seeds:** new experiments use **3 seeds × 5 folds** at a **fixed 90 epochs**, exactly the full-data recipe, so a result reflects what the submitted model would do, not a lucky early-stopping epoch.
- **Statistics:** a **paired identity-level cluster bootstrap** (`FLAG_2027/local_validation/compare_fold_runs.py`). Held-out identities are resampled within each fold, and both systems are scored on the same resampled trials. The tool reports the B − A difference, a 95% interval and P(B better). A result only counts if the interval clearly excludes zero, or at least most folds agree.
- **Track record:** the local fixed-epoch estimate of V2 (28.8%) predicted its dev leaderboard score (29.11%) within 0.3 points.
- **V2 reference numbers in this setup** (seeds 1–3, fixed 90 epochs, `runs/voice_aug/base`): **standard 22.88%, gender 34.17%, mean 28.52%**.
- **EER formula note:** the comparison tool interpolates the EER crossing point slightly differently from the training loop. On identical scores they differ by 0.1–0.15 points. Comparisons always put both systems through the same tool, so this doesn't affect any conclusion.

---

## 6. Experiment log

Status: ✅ improvement adopted · ❌ no gain or worse · 🔬 diagnostic · ⏳ running

### 6.1 V1 → V2 ✅
- **V1:** FaRL faces + WavLM x-vector voices, with ArcFace-regularised projectors. Five-fold, best epoch per fold: standard 25.9%, gender 37.5%, mean 31.7%.
- **V2:** the all-layer WavLM voice average, the SupCon loss without ArcFace, and the regularisation above (z-scoring, input noise and dropout, a single 256-d hidden layer). Best epoch per fold: 22.4% / 33.5% / 28.0%; fixed epoch 90: 23.1% / 34.4% / 28.8%.
- Paired bootstrap, V2 vs V1: standard **−2.9 points** [95% CI −5.5, −0.2], gender **−3.9 points** [−7.3, −0.5]. Both significant.

### 6.2 Face backbones ❌
FaRL with flip averaging, FaRL all-layer average, DINOv2 patch tokens, DINOv2 all-layer CLS, and FaRL + DINOv2 combined (`runs/face_features`) all matched FaRL within noise. DINOv2 alone was worse (gender 36%). InsightFace ArcFace was the V1-era face input.
**Why:** the face side already captures far more than the voice side (see [§6.6](#66-diagnostic-what-information-does-the-voice-carry-)). The voice limits the match, so sharper face features can't help.

### 6.3 Losses and training tricks ❌
Same-gender SupCon (no change on gender, +1 point on standard), hard-negative mining, prototype loss, orthogonal projection loss (OPL / FOP-style), GADH: no gains. Physiology features (F0, formants) added to the voice: no gain.

### 6.4 Learned pairwise verifier ❌ (`runs/pairwise_farl_wavlm`)
An MLP on [face, voice, |difference|, product], trained with hard same-gender negatives (the "multimodal encoder" idea of the FAA paper, arXiv 2404.09509):

| Scoring | Standard | Gender | Mean |
|---|---|---|---|
| Cosine | 28.9 | 39.3 | 34.1 |
| Verifier alone | 33.3 | 39.5 | 36.4 |
| Best blend | 28.6 | 38.7 | 33.7 |

**Why it failed:** on training identities the embeddings are almost perfectly separated (training EER 0.2%), so the head learns from an easy problem that looks nothing like unseen people.

### 6.5 Score normalisation and ensembles ❌
AS-norm didn't help on English (local test only, never submitted; now forbidden by the rules anyway). Score-averaging ensembles gave at most −0.7 points on gender, not significant.

### 6.6 Diagnostic: what information does the voice carry? 🔬
**Question:** what besides gender links a face to a voice?

**Age** (`FLAG_2027/analysis/estimate_face_age.py`, `probe_voice_age.py`; results in `features/face_age/`)
- Face ages come from InsightFace `buffalo_l` (gender agreed with the metadata for 70/70 identities). The label per identity is the median over its frames (single frames vary by about 6 years).
- A ridge regression from WavLM voice features predicts face age **for unseen identities, within each gender**: men r = 0.65, women r = 0.59, **p < 0.005** (200 permutations). It orders 84% (men) and 89% (women) of same-gender pairs 10+ years apart correctly.
- For comparison, face features predict the same labels at r = 0.92 / 0.80.

**Body build** (`score_face_attributes_farl.py`, `probe_voice_build.py`; results in `features/face_build/`)
- Zero-shot labels come from the FaRL text encoder with "fat/heavy-set" vs "thin/slim" prompt sets. The method was validated first: zero-shot gender AUC 1.0 against the metadata, and zero-shot age correlates r = 0.81 / 0.76 with InsightFace ages. I also checked the extremes by eye: correct for women, noisier for men.
- Build is a stable per-person trait (63% of its variance is between identities) and nearly unrelated to age.
- Voice → build: men r = 0.31 (p = 0.02), and 0.34 with age removed (p = 0.01); women r = 0.29 (p = 0.04), and 0.27 with age removed (p = 0.05). About half the strength of age.

### 6.7 Diagnostic: does V2 already use age and build? 🔬 (`probe_embedding_age.py`, `age_fusion_test.py`, `attribute_fusion_test.py`)
- V2's voice embedding predicts age as well as the raw features (r = 0.66 / 0.66), so V2 already learned it.
- Same-gender AUC rises with the impostor's age gap: 0.62 (< 5 years), 0.70, 0.74, 0.82 (20+ years).
- **Similar-age impostors (< 5 years, about a third of same-gender pairs) are the main source of error.**
- Adding explicit age/build-difference scores (weights chosen on the other folds), same-gender EER:

| Scoring | EER | Folds improved |
|---|---|---|
| V2 | 35.1% | – |
| + age | 34.3% | 3/5 |
| + build | 35.1% (similar-age AUC 0.624 → 0.639) | 3/5 |
| + age + build | 33.9% | 3/5 |

  Suggestive, but not reliable.

### 6.7b Where V2's gender-track errors come from 🔬 (13 Nov)
Measured on each fold's real gender-controlled protocol (`validation/fold_k/gender_pairs.csv`), with V2 at its own EER threshold, over 5 folds × 3 seeds (`runs/voice_aug/base`). Ages are InsightFace face-age medians per identity.

| Impostor age gap | Share of impostor trials | False-accept rate | Share of all false accepts |
|---|---|---|---|
| 0–5 years | 35% | 46% | **47%** |
| 5–10 years | 25% | 34% | 25% |
| 10–20 years | 21% | 27% | 17% |
| 20+ years | 19% | 20% | 11% |

- **Correction to an earlier statement:** similar-age pairs are about **half** of the false accepts, not "the bulk of the errors".
- **Errors split evenly in two:** at the EER point, errors are half **false accepts** (3,592) and half **false rejects** (3,583, i.e. 34% of true pairs scored too low).
- **V2 under-uses age:** even impostors 20+ years apart are falsely accepted 20% of the time. An age-consistency term can address the ≥5-year half of the false accepts.
- **What the rest needs:** similar-age false accepts and false rejects need finer person-specific cues (speaker-verification embeddings), or less noisy embeddings (more training identities).

### 6.12 Speaker embedding (ECAPA) ❌ + age-consistency term ✅ (13 Nov)
**Why:** about half of the gender-track false accepts are similar-age pairs, and half of all errors are false rejects (§6.7b). Those need finer person-specific voice cues. The other false accepts have clear age gaps, which V2 under-uses.

**Frozen models added** (public pretrained, used without fine-tuning):

| Model | Source / revision | Licence | Output per clip |
|---|---|---|---|
| ECAPA-TDNN speaker verification, trained on VoxCeleb2 (5,994 speakers); VoxCeleb1-O EER 1.29% | `yangwang825/ecapa-tdnn-vox2` @ `c605f571` | Apache-2.0 | 192-d speaker embedding |
| wav2vec2-large-robust age/gender (audeering) | `audeering/wav2vec2-large-robust-24-ft-age-gender` @ `bad2737f` | **CC BY-NC-SA 4.0 (non-commercial)** | 1024-d mean hidden state; age (years); p(female/male/child) |

Both are loaded with the same 16 kHz PCM loader as the WavLM features, and each clip is encoded whole and independently. Script: `FLAG_2027/feature_extraction/extract_voice_features_speaker_age.py`. Outputs: `features/voice_v6_ecapa/`, `features/voice_v7_agegender/`.

**Smoke checks** (20 training clips):
- ECAPA: same-speaker cosine 0.52 against 0.09 for different speakers.
- Age-gender model: gender correct on the clips checked (p ≈ 0.98).

**Experiments:**
1. **ECAPA** (`runs/speaker_embedding/run_all.sh`): `wavlm_ecapa` (V2 WavLM 768 + ECAPA 192 = 960-d input) and `ecapa_only` (192-d), 3 seeds × 5 folds × 90 epochs, against V2.
2. **Age-consistency term B** (`FLAG_2027/analysis/age_consistency_test.py`):
   - score = cos − w_age·|voice age − face age|/10 − w_gender·|p_male(voice) − male(face)|;
   - **only the trial's own clip and image are used** (no identity averages), as the rules require;
   - weights are chosen per track on the other four folds;
   - paired bootstrap against V2.
   Gated on a quick check: voice-age vs face-age correlation within gender, against the earlier probe's r ≈ 0.6.
3. **Approach A** (audeering features as an extra input): only if B gains.

**Results: quick check ✅** (`--check-only`). Voice gender matches the metadata for 70/70 identities.

| | Earlier 70-identity probe | audeering voice age, per identity | Single clip vs single face image |
|---|---|---|---|
| Men | r = 0.65 | **0.79** | 0.63 |
| Women | r = 0.59 | **0.72** | 0.52 |

  Voice age is biased upwards: mean voice age 49.1 vs face age 42.5 (men), and 46.7 vs 30.9 (women).

**Results: approach B** (3 seeds × 5 folds, V2 fold models `runs/voice_aug/base`; JSON in `features/voice_v7_agegender/age_consistency_B*.json`):

| Variant | Standard | Gender | Mean |
|---|---|---|---|
| V2 | 22.88 | 34.17 | 28.52 |
| V2 + raw terms | 22.55 (−0.33, n.s.) | 34.17 (±0.00) | 28.36 (−0.16, n.s.) |
| **V2 + bias-corrected terms** | **21.50 (−1.37, CI [−2.05, −0.30], P = 0.997)** | 33.94 (−0.23, n.s., P = 0.85) | **27.72 (−0.80, CI [−1.18, −0.17], P = 0.997)** |
| Hyperbolic τ = 0.1 | 21.86 | 34.15 | 28.00 |
| Hyperbolic + bias-corrected terms (weights × 1.36) | 21.36 | 34.06 | 27.71 (−0.30 vs hyperbolic, n.s.) |

- **Why the raw terms failed:** the upward voice-age bias penalises *true* pairs. A woman's voice reads about 16 years older than her face, so her true pair shows a large "gap", while an older-looking impostor face shows a smaller one.
- **The bias correction** is a linear map from voice age to face age, per *predicted* voice gender (p_male ≥ 0.5), fitted on each fold's training rows only. On fold 0: men face ≈ 0.61·voice + 12.0; women face ≈ 0.41·voice + 12.9. It is a fixed model component fitted on training data, like input z-scoring; no test-set statistics are used.
- **Weights chosen** (leave-one-fold-out):
  - standard track: age 0.05–0.08 per 10 years, gender 0.1;
  - gender track: age 0.01, gender 0.
- **Interpretation:**
  - the terms give a **significant −0.8 mean improvement, almost all on the standard track** (4/5 folds better);
  - on the gender track, single-clip age estimates are too noisy (r ≈ 0.5–0.6 clip-to-image) to separate same-gender people;
  - hyperbolic and the terms overlap (no additive gain), so **V2 + corrected terms is the best and simplest candidate** (no retraining).
- **Caveat:** two variants of B were tried (raw, corrected). The correction was motivated by the diagnosed bias, but the second try carries a little selection optimism.
- **ECAPA experiment, preliminary** (seed 1 only, all 5 folds; seeds 2–3 still training):

| Variant | Standard | Gender | Mean |
|---|---|---|---|
| V2 (seed 1) | 23.17 | 34.41 | 28.79 |
| `wavlm_ecapa` (WavLM 768 + ECAPA 192) | 22.43 (−0.7, P = 0.91) | 33.90 (−0.5, P = 0.75; folds 3/5 better) | 28.16 (−0.6, P = 0.86) |
| `ecapa_only` | 26.40 (+3.2) | 37.59 (+3.2) | 31.99 (**+3.2, clearly worse**) |

  - ECAPA alone is much worse. Speaker-verification embeddings discard the broad traits (age, pitch, timbre classes) that link a voice to a face; this matches V1, where the WavLM x-vector lost to the all-layer average.
  - Concatenated with WavLM, it shows a small, not yet significant gain that also reaches the gender track. Waiting for seeds 2–3.
- **ECAPA experiment, final** (3 seeds × 5 folds; JSON in `runs/speaker_embedding/compare_base_vs_*.json`):

| Variant | Standard | Gender | Mean |
|---|---|---|---|
| V2 | 22.88 | 34.17 | 28.52 |
| `wavlm_ecapa` | 22.83 (−0.05) | 34.80 (+0.64) | 28.82 (+0.30): **no gain** |
| `ecapa_only` | 26.27 (+3.4) | 37.26 (+3.1) | 31.76 (+3.2, CI [+0.8, +6.0]): **worse** |

  - The seed-1 gain disappeared with seeds 2–3. Per fold, gender improved on folds 0 and 3 but worsened on fold 1 (36.7 → 41.6) and fold 4, so the extra input made training **less stable**.
  - **Conclusion:** dropped. With 70 identities, extra voice inputs (layer spread, attention, ECAPA) keep adding variance rather than signal.
- **Decision:** the submission candidate is **V2 + bias-corrected age/gender terms** (no retraining).

### 6.8 Voice augmentation ❌ (`runs/voice_aug`)
- **Setup:** 5 augmented WavLM views per training clip (`features/voice_v5_wavlm_allmean_aug/`), mixed in during training:
  - a random 50–100% segment of the clip, always;
  - synthetic coloured noise at 5–20 dB SNR (p = 0.6);
  - a synthetic room echo, RT60 0.2–0.7 s (p = 0.4);
  - telephone or low-pass band-limiting (p = 0.3);
  - 8-bit μ-law coding (p = 0.2).
- No pitch or speed changes, to keep identity cues. No external audio.
- **Results** (3 seeds × 5 folds, 90 epochs):

| Variant | Mean EER change | Gender EER change |
|---|---|---|
| Clean voice 50% of the time | **+0.8** (worse) | +1.1 |
| Clean voice 25% of the time | **+1.1** (worse; 95% CI excludes 0) | +1.3 |

- **Likely reasons:**
  - V2 is already strongly regularised;
  - the augmentations move features far: an augmented view's cosine to its own clip is 0.82, against 0.63 to other clips;
  - possibly, recording conditions genuinely help matching (a face frame and a voice from the same video share its quality).
- **Decision:** dropped, and face augmentation not pursued.

### 6.9 Papers reviewed
- **FAA** ("Fuse after Align", arXiv 2404.09509): learned matching head, multi-similarity loss, progressive clustering. Its main idea was effectively tested in §6.4. Clustering is for unlabelled data; test-time pooling is forbidden by the rules.
- **PAEFF** (arXiv 2505.17002): its biggest gain is the alignment loss, which V2 already has. Gated fusion has effectively been tested. **New for us: hyperbolic (Poincaré) embeddings**, which gave −1.0 EER points in their ablation. Their same-gender AUC (71.5%) is about where ours is (≈ 71%), so the gender problem is open for the state of the art too.

### 6.10 Hyperbolic (Poincaré) embeddings ✅ standard track / ❌ gender track (`runs/hyperbolic`, 13 Nov)
- **Idea:** in hyperbolic space, room grows towards the boundary, which suits hierarchies (gender → age/build → individual). This might separate same-gender people better.
- **Implementation:**
  - the projector output (no L2 normalisation) is norm-clipped to radius 1 (feature clipping, Guo et al., CVPR 2022), then mapped onto the Poincaré ball (curvature c = 1) with the exponential map at the origin;
  - similarity = **negative geodesic distance**;
  - the same distance is used in SupCon (as logits / temperature), in the pair loss (mean distance of matching pairs), in fold validation, in the comparison tool and in the submission file (distance written directly).
- **Variants:** temperature 0.1 (`hyp_t010`) and 0.2 (`hyp_t020`). Everything else is identical to V2. 3 seeds × 5 folds × 90 epochs, compared with `runs/voice_aug/base`.
- **Smoke test** (fold 0, 10 epochs): trains stably, val std 22.9% / gender 36.6%, against V2's 22.9% / 37.6% at the same point. Too early to mean anything.
- **Results** (3 seeds × 5 folds, fixed 90 epochs, paired bootstrap against V2 `runs/voice_aug/base`; JSON in `runs/hyperbolic/compare_base_vs_*.json`):

| | Standard EER | Gender EER | Mean EER |
|---|---|---|---|
| V2 (sphere) | 22.88 | 34.17 | 28.52 |
| Hyperbolic, τ = 0.1 | **21.86** (−1.0, CI [−2.0, −0.25], P = 0.99) | 34.15 (±0.0) | 28.00 (−0.5, P = 0.91) |
| Hyperbolic, τ = 0.2 | **21.57** (−1.3, CI [−2.4, −0.24], P = 0.99) | 34.41 (+0.25) | 27.99 (−0.5, P = 0.84) |

  - Standard track, per fold, τ = 0.1: 19.2 / 23.2 / 20.1 / 25.8 / 21.1 against V2's 18.7 / 25.9 / 21.1 / 27.3 / 21.4, so **4 of 5 folds improved**.
  - Gender track, per fold, τ = 0.1: 31.1 / 36.8 / 31.4 / 38.2 / 33.3 against 31.4 / 36.7 / 30.6 / 37.5 / 34.7: mixed.
- **Conclusion:**
  - The hyperbolic space **reliably improves the standard track by about 1–1.3 points**, matching PAEFF's reported ~1-point gain.
  - It does **not** help the gender track, so the hierarchy argument (more room to separate people within a gender) didn't materialise.
  - Overall the mean improves about 0.5 points, which is suggestive but not significant on its own.
  - Both temperatures behave alike. τ = 0.1 is the safer choice (no gender cost).
- **Status:** a candidate for combining with the best layer-pooling variant before any submission.

### 6.11 WavLM layer attention ❌ (`runs/layer_attention`, 13 Nov)
- **Idea:** V2 throws information away by averaging the 13 WavLM layers equally and dropping each layer's spread over time. Let the model learn which layers matter.
- **Input:** the stored per-layer statistics `features/voice_v2_wavlm_layers/train_English_layers.npy`, shape (6,485, 13, 1,536), the frame mean and std of each layer. No re-extraction needed; dev and eval layer files exist too. Alignment checked: averaging the layer means reproduces V2's CSV (max difference 4e-8).
- **Implementation:**
  - `VoiceLayerPool` z-scores each layer with training statistics, then pools the 13 layers;
  - the result goes into the usual voice projector, whose own z-scoring is turned off.
- **Variants:**

| Variant | What it tests | Extra parameters |
|---|---|---|
| `mix_mean` | One learned softmax weight per layer, shared by all clips. Starts as an equal average (≈ V2). | 13 |
| `attn_meanstd` | Per-clip additive attention over layers (Linear 1536 → 64, tanh, Linear → 1, softmax over layers), on mean + std | ≈ 300k |
| `mix_meanstd` | Learned layer weights on mean + std; separates "adding std" from "per-clip attention" | 13 + larger first layer |

- **Smoke test** (fold 0, 10 epochs): both versions train; gender 37.0% (`mix_mean`) and 37.3% (`attn_meanstd`), against 37.6% for V2.
- **Preliminary results** (seeds 1–2 × 5 folds, 90 epochs; seed 3 still running), paired bootstrap against V2 on the same seeds:

| Variant | Standard | Gender | Mean |
|---|---|---|---|
| V2 (seeds 1–2) | 23.05 | 34.40 | 28.73 |
| `mix_mean` | 22.76 (−0.3, P = 0.80) | 34.54 (+0.1) | 28.65 (−0.1): **no change** |
| `mix_meanstd` | 23.55 (+0.5) | 35.74 (+1.3, P(better) = 0.02) | 29.65 (+0.9): **worse** |
| `attn_meanstd` | 24.34 (+1.3) | 36.47 (+2.1, P(better) = 0.00) | 30.41 (+1.7): **clearly worse** |

  - **Learned layer weights** (`mix_mean`, mean over 10 runs; uniform = 0.077): layers 0–5 at 0.088–0.098, layers 6–12 at 0.060–0.064. The model consistently prefers the **lower, acoustic layers** (timbre, pitch, resonance) over the upper layers (more phonetic and content information), but only mildly (maximum deviation from uniform 0.022). So V2's plain average is already near-optimal.
  - **Adding each layer's spread over time hurts**, and per-clip attention hurts most. With 70 identities, the extra inputs and parameters (≈ 300k for attention) overfit.
  - **Preliminary conclusion:** layer pooling gives no gain; V2's all-layer mean stays.
- **Final results** (all 3 seeds × 5 folds, 90 epochs; JSON in `runs/layer_attention/compare_base_vs_*.json`):

| Variant | Standard | Gender | Mean |
|---|---|---|---|
| V2 | 22.88 | 34.17 | 28.52 |
| `mix_mean` | 22.70 (−0.2, CI [−0.7, +0.3]) | 34.20 (+0.0) | 28.45 (−0.1): **no change** |
| `mix_meanstd` | 23.33 (+0.5) | 35.40 (+1.2, CI [+0.1, +2.1]) | 29.37 (+0.85): **worse** |
| `attn_meanstd` | 24.43 (+1.6, CI [+0.4, +3.2]) | 36.30 (+2.1, CI [+0.4, +3.6]) | 30.36 (+1.8): **significantly worse** |

- **Final conclusion:** the seed-3 runs confirm the preliminary result. Learned layer weights equal V2; adding temporal spread or per-clip attention overfits. Dropped.
- Each run is limited to 3 CPU threads (`OMP_NUM_THREADS=3`), because six parallel jobs were oversubscribing the 20-core machine.
- **Results:** *pending.*

---

## 7. Code and data changes, file by file

Paths are relative to `Face-to-Voice/`. The challenge code lives in `FLAG_2027/` (a clone of `github.com/SwapnilKhandoker101/FLAG_2027`, last upstream commit 18 Sep 2026). Every change keeps V2's behaviour identical by default. This was verified by bit-exact reproduction of V2 training losses and dev submission files.

### Model and training
| File | Change |
|---|---|
| `FLAG_2027/cross_modal_model.py` | **Projector:** `normalize_output` and `output_map` options. **New:** `PoincareMap` (norm clip + exponential map), `poincare_distance_matrix`, `poincare_pair_distance`, `VoiceLayerPool` (per-layer z-score + learned mix or attention), `VoiceLayerBranch`. **Network:** `geometry` ("sphere" or "hyperbolic"), `curvature`, `clip_radius`, `voice_layers` and `voice_layer_pooling` arguments; `similarity_matrix()` and `pair_similarity()` methods, the single source of truth for scoring. V2 checkpoints load with identical keys. |
| `FLAG_2027/train_cross_modal.py` (fold training) | New options: `--voice-views-npy` and `--clean-view-probability` (augmentation); `--geometry`, `--curvature`, `--clip-radius` (hyperbolic); `--voice-layers-npy`, `--voice-layer-features {mean,meanstd}`, `--voice-layer-pooling {mix,attention}` (layer attention). The loss uses `model.similarity_matrix`. In hyperbolic mode the pair loss is the mean geodesic distance, and cosine-only losses are rejected. |
| `FLAG_2027/train_cross_modal_full.py` (full-data training) | Earlier: supports the V2 options and stores `model_kwargs` in the checkpoint. (The new hyperbolic and layer options will be added only if a variant wins.) |
| `FLAG_2027/local_validation/cross_modal_losses.py` | `symmetric_cross_modal_contrastive_loss` accepts a precomputed `similarities` matrix (needed for non-cosine geometry). |
| `FLAG_2027/local_validation/model_evaluation.py` | Fold validation scores hyperbolic models with their own geodesic distance. |

### Evaluation and submission
| File | Change |
|---|---|
| `FLAG_2027/local_validation/compare_fold_runs.py` | **New.** A paired, multi-seed, identity-level bootstrap comparison of two fold-run sets. It reads each run's geometry and voice input (CSV or layer array) from its checkpoint config. Checked against known V2 numbers. |
| `FLAG_2027/generate_cross_modal_submission.py` | Earlier: loads models from `model_kwargs`. **New:** `--split` (`test` for dev, `valid` for evaluation) and a help text on `--dev-root` (trial-list folder). Hyperbolic models write their geodesic distance. A regression check regenerated the dev submission byte for byte. |

### Feature extraction
| File | Change |
|---|---|
| `FLAG_2027/feature_extraction/extract_voice_features_wavlm_augmented.py` | **New.** Augmented WavLM views (see §6.8). `--no-augment` mode reproduces V2 features (checked: max difference 6e-4, from fp16). |
| `features/extract_dev_voice_v2.sh` | Earlier: dev-set WavLM features for V2. |
| `features/extract_eval_v2.sh` | **New.** Evaluation-phase FaRL and WavLM features with settings identical to training and dev, reading the corrected trial lists. |

### Analysis (diagnostics in §6.6–6.7)
`FLAG_2027/analysis/`: `estimate_face_age.py`, `probe_voice_age.py`, `probe_embedding_age.py`, `age_fusion_test.py`, `score_face_attributes_farl.py`, `probe_voice_build.py`, `attribute_fusion_test.py`. All new.

### Tests (`FLAG_2027/local_validation/test_cross_modal.py`)
- Added: augmented voice-view sampling, and hyperbolic geometry (points stay inside the ball; matrix and per-pair scores agree; same checkpoint keys as the sphere model).
- Suite status: **38/38 passing.**

### Data
- `evaluation_phase/`: the official evaluation set (unchanged).
- `features/eval_manifests/`: trial lists used by our pipeline, with the `gender/English_valid.txt` path fix.
- `features/eval_v2/`, `features/eval_voice_v2_wavlm_layers/`: evaluation features.
- `checkpoints/v2/`: V2 reference folder (models, logs, submissions, CodaBench results, README).

---

## 8. Challenge rules and compliance

From the official FAQ (received 13 Nov):

**Allowed**
- Public pretrained encoders (face and audio), even if their pretraining data contains Bengali.
- External datasets and extra pretrained models, if documented, **except** that Bengali speech may not be used to train, fine-tune or adapt an audio encoder.
- Frozen pretrained age/gender models, including a *fixed* gender-consistency term in the score.
- Deterministic encoder preprocessing (e.g. fixed chunking and averaging).
- Using the leaderboard to *choose between our own finished systems*.

**Forbidden**
- Any use of progress or evaluation data for training, adaptation, pseudo-labels, clustering or calibration.
- Score normalisation (Z/T/S-norm, AS-norm, cohort methods), even with training-only cohorts.
- Cross-trial processing: every trial must be scored independently.
- Test-time adaptation.
- Tuning parameters on the leaderboard.

**Our compliance**
- All submissions so far score each trial independently with `1 − cosine`.
- No normalisation, calibration or evaluation-data statistics are used.
- Input standardisation uses training-set statistics as part of the model, which is allowed.
- **Policy:** decide everything on local folds; submit only systems that win locally; one submission per finished system.

---

## 9. Next steps

1. **Candidate for the next evaluation submission: hyperbolic τ = 0.1** (−1.0 standard, significant; ±0 gender). It needs full-data training support in `train_cross_modal_full.py` and a 70-identity retrain. Layer attention is dropped, so no combination run is needed. For a winner: retrain on all 70 identities, build its evaluation zip, and submit (deadline 17 Nov, 23:00 UTC).
2. **Age/build auxiliary heads** on top of the best voice model: predict face-estimated age and FaRL build from both embeddings. Age is the main target, build secondary at a lower weight.
3. **Frozen pretrained age/gender consistency term** (explicitly allowed). Use a public voice age/gender model, with the fusion weight chosen on training folds and fixed before submission.
4. **External training data** (likely for the second window): MAV-Celeb v1/v3 (same pipeline, English–Urdu / English–German), then VoxCeleb1/2. Bengali speech must be filtered out with a frozen public language-ID model, and everything documented.
5. **Time-frame attention** over WavLM frames (needs frame-level re-extraction), if layer attention helps.
6. **Deliverables:** the 2-page ICASSP system description and a cleaned, reproducible public repository.
7. **Questions for the organisers:** which evaluation phase is final, the `gender/English_valid.txt` path error, and the submission limit (15 vs 100).

---

### 9.1 Gender-track strategy (discussed 13 Nov)

**What the evidence says**
- The gender-track errors are mostly same-gender, similar-age pairs (§6.7).
- The voice side is the weaker half: it predicts age at r ≈ 0.6, against ≈ 0.9 for the face.
- Every training-objective change aimed at gender has failed so far: same-gender SupCon, hard negatives, augmentation, the learned verifier, hyperbolic geometry.
- That points to two limits: **too few identities** (about 35 per gender) and **voice features that carry too little person-specific information**.

**Options, ranked by expected gain on the gender track**
1. **More training identities (external paired face–voice data).** Expected: largest. Literature: FAA, trained on 924 VoxCeleb identities, reaches a same-gender AUC of 0.80, against our ≈ 0.71 with 70. Bengali speech must be filtered out. Effort: days, realistic for the second evaluation window.
2. **Stronger or complementary frozen voice encoders:**
   - WavLM Large (24 layers);
   - a dedicated speaker-verification embedding (ECAPA/ResNet trained on VoxCeleb) combined with the WavLM trait features;
   - a frozen public voice age/gender model (e.g. audeering wav2vec2 age-gender) used as an extra input.
   Expected: moderate. Effort: about 1 day each (extraction plus a five-fold test).
3. **A frozen pretrained age-consistency term in the score** (voice age model vs InsightFace face age), with a fixed weight from the folds; allowed by the rules. Our ridge-based version gave −0.8 (3/5 folds). A pretrained voice-age model may make it reliable. Expected: about 1 point.
4. **Ensemble of diverse finished models** (V2, hyperbolic, layer variants, several seeds; fixed equal weights). Expected: 0.5–1 point. Effort: low.
5. **Attribute auxiliary heads** (age and build). Expected: up to about 1 point.
6. **3D face-shape features** (geometry linked to the vocal tract). Speculative.

**Candidate frozen voice encoders for option 2** (13 Nov; none tested yet):

| Family | Models | Information carried | Priority |
|---|---|---|---|
| Speaker verification (trained to tell about 6k VoxCeleb speakers apart) | ECAPA-TDNN (extractor already in repo: `yangwang825/ecapa-tdnn-vox2`), WeSpeaker / 3D-Speaker ResNet-293, CAM++, ERes2Net, NVIDIA TitaNet, ECAPA2 | Fine person-specific timbre; trained to ignore content and channel | **1:** concatenate with V2's WavLM features |
| Paralinguistic / attribute | audeering `wav2vec2-large-robust-24-ft-age-gender` (age regression + gender), Google TRILLsson (non-semantic speech), speaker-trait models (voice quality, accent) | Age, sex, voice quality: the traits shared with the face | **2:** hidden states as input + age prediction as a fixed consistency term |
| Larger self-supervised speech | WavLM Large (24 layers), HuBERT Large/XLarge, wav2vec2 XLS-R (multilingual), w2v-BERT 2.0 | Broad traits, like V2's WavLM Base+ but higher capacity | **3:** direct upgrade of V2's input |
| Audio–language (CLAP-style) | LAION-CLAP, MS-CLAP, paralinguistic CLAP variants | Descriptive attributes ("deep, older male voice") in a text-aligned space | 4: speculative |

Rules: all are public pretrained models, so frozen use is allowed even if their pretraining data includes Bengali. No Bengali fine-tuning. Document each model in the system description.

**Decision (13 Nov): how to use the audeering age-gender model.** Primary: **B, a fixed age-consistency score term**.
- Score = cos(face, voice) − w · |voice age − face age|.
- Voice age from the frozen audeering model; face age from frozen InsightFace.
- A single weight w is chosen on the training folds and frozen before submission.
- Secondary: **A, its 1,024-d features as an extra voice input**, tested only after B.
- Both are gated on a quick check: its voice-age estimates must correlate with face age within gender clearly better than our 70-identity probe (r ≈ 0.6).

**Deprioritised** (already tested, no gain): loss reweighting towards same-gender negatives, hard-negative mining, augmentation, other face backbones, learned verifier heads.

### 9.2 External training data plan (13 Nov; downloads pending)

**Why:** five changes to the frozen-features + small-network setup all came out flat, and we sit 8 points behind the leaderboard median, mostly on the gender track. With 70 identities (about 35 per gender) the model cannot learn the fine cues that separate same-gender people. More identities is the main lever.

**Datasets** (checked 13 Nov):

| # | Dataset | Languages | Train identities | Size | Access | Status |
|---|---|---|---|---|---|---|
| 1 | MAV-Celeb v3 | English + German | 50 (185 + 169 videos) | 1.7 GB | public Google Drive (`github.com/mavceleb/mavceleb_baseline`) | Drive quota error; needs browser download / Drive copy |
| 2 | MAV-Celeb v1 | English + Urdu | 64 (344 + 518 videos) | 11.3 GB | same | same |
| 3 | MAV-Celeb v2 | English + Hindi (per the original 154-identity MAV-Celeb) | to be counted | 13.5 GB | same | same |
| 4 | VoxCeleb2 test (pilot) | mostly English, multilingual | 118 | 8.9 GB (mp4) | HF `ProgramComputer/voxceleb` (gated) or the official VGG form | access request needed |
| 5 | VoxCeleb2 dev | mostly English, multilingual | 5,994 | 267 GB (mp4, 9 parts) | same | after the pilot |

- MAV-Celeb test splits are unlabelled, so only the training splits are used.
- VoxCeleb1 is skipped: the mirror has audio only (no faces).
- MAV-Celeb split lists are saved in `external_data/mavceleb/splits/` (v1, v3). The download script is `external_data/mavceleb/download.sh`.

**Rules safeguards:**
- **No Bengali speech.** Every external clip goes through a frozen public language-ID model (SpeechBrain `lang-id-voxlingua107-ecapa`). Clips detected as Bengali are dropped, with a conservative threshold. All drops are logged.
- **No evaluation or dev data** is used in any step, including deduplication.
- **Deduplication against our 70 training identities:** compare FaRL face centroids of external identities with official training identities, and drop close matches after a manual check. Otherwise a held-out fold identity could leak into training and make local validation optimistic.
- Possible overlap of external identities with the hidden evaluation identities cannot be checked without touching eval data, so we will ask the organisers.
- Every dataset and its licence goes into the system description.

**How the data is merged** (detailed plan, 16 Nov):

*Step 1: read the raw data without unpacking.*
- The zips stay in `external_data/<source>/raw/` as the permanent raw copy.
- Extractors read the training-list files straight from the zip, so nothing is duplicated on disk.
- Audio is decoded to 16 kHz mono, like the official clips.
- Faces are matched to the official 224 × 224 crops.
  - MAV-Celeb frames come from the same pipeline as the official data; their size and framing are checked first, and they are re-cropped with the InsightFace detector only if they differ.
  - VoxCeleb2 videos are already face-centred tracks; one or more frames per clip are sampled, and the audio comes from the same mp4.

*Step 2: filter, before any training.*
- **Language:** a frozen VoxLingua107 language-ID model scores every clip. Clips with a meaningful Bengali probability are dropped, and all drops are logged per source.
- **Quality:** drop clips under 2 s and frames where InsightFace finds no face.
- **Gender per identity:** VoxCeleb2 metadata where available. Otherwise a majority vote of InsightFace face gender and audeering voice gender, with disagreements checked by hand. It is needed for the gender metadata file and the sampler.
- **Duplicates:** compare FaRL face centroids and WavLM voice centroids per identity, then check the closest candidates by eye.
  - Against the 70 official training identities: drop the external copy.
  - Between external sources (e.g. v1 vs v2 from the same 154-celebrity pool, MAV-Celeb vs VoxCeleb2): merge into one identity.
  - Dev and evaluation data are never used, not even for deduplication.

*Step 3: one format and the same features.*
- Each source gets a manifest in the official format with namespaced identities (`mav1_id0001`, `mav2_…`, `mav3_…`, `vox2_id…`) and class indices continuing after 70.
- A gender metadata file in the same `id,m/f` format.
- Identical frozen extraction:
  - FaRL ViT-B/16 face features;
  - WavLM Base+ voice features (13-layer mean, 10 s chunks, 5 s overlap);
  - audeering age/gender and InsightFace face age, for the attribute terms.
- Outputs go to `features/external/<source>_{faces,voices,attributes}.csv`.

*Step 4: append to training only; validation is unchanged.*
- Combined feature matrices: the official 6,485 rows first, external rows appended after. The existing `validation/fold_k/` files (`train_indices.npy`, pair and ground-truth lists) stay valid unchanged.
- For each fold: training = official training rows of that fold + all external rows. Validation = the same 14 held-out official identities and the same trials. Every result is therefore a paired comparison against V2 (3 seeds × 5 folds, `compare_fold_runs.py`).
- New training options (off by default, so V2 is unchanged): `--extra-face-csv`, `--extra-voice-csv`, `--extra-manifest`, `--extra-gender-metadata`, plus a sampler share for official identities.
- z-score statistics are fitted on all training rows of the fold (official + external).

*Step 5: training changes.*
- **Domain-balanced batches:** P16 × K8 with a fixed share of official identities (try 50%, 25% and natural), so the in-domain data is not swamped.
- **Cross-language positives:** MAV-Celeb identities have English and Urdu/German/Hindi clips. A batch's K samples per identity mix languages, which teaches language-independent matching (relevant for the unheard Bangla cells).
- **Training length:** fixed by optimisation steps rather than epochs, since epochs grow with the data. Start at V2's step count, then test longer.
- **Model size:** with roughly 4× the identities, test a wider projector (512 hidden).

*Step 6: ablation, then the final model.*
- Add sources one at a time (+v3, +v1, +v2, +VoxCeleb2 test) to measure each one's contribution. This is useful for the system description and to drop a source that hurts.
- Keep only what beats V2 locally.
- Final model: all 70 official identities + the accepted external data with the fixed recipe; re-fit the attribute terms with the new fold models; build the evaluation zip.

*Step 7: documentation.* A data card in this file with, per source:
- identities and clips kept;
- clips dropped (language, quality, duplicates);
- licence;
- how gender was assigned.

**Staging:**
1. MAV-Celeb v1–v3: about 26 GB, same domain, 100–250 new identities (about 2–4× our data).
2. VoxCeleb2 test pilot (118 identities).
3. VoxCeleb2 dev subset scaling (1,000 → 6,000 identities).

Each stage is accepted only if it beats V2 on the official five-fold test.

**Disk-space decision (13 Nov):**
- The user has about 14 GB left.
  - The project and home directory are on NFS. The filesystem shows 6.5 TB free, but the home directory already uses 95 GB, so the 14 GB is most likely a per-user quota (it cannot be read: no `quota` tool).
  - The local root disk has 13 GB free.
- **Scope reduced to MAV-Celeb v3 + v1:**
  - v3: 50 identities, 1,422 English + 1,595 German clips;
  - v1: 64 identities, 5,786 English + 12,252 Urdu clips;
  - together 13.0 GB of zips. MAV-Celeb v2 and VoxCeleb2 are dropped for now.
- **Low-space processing:**
  1. Download v3 first.
  2. Read only the training-list files straight from the zip (no extraction) and write the features.
  3. Delete the zip.
  4. Then do the same for v1.
  - Peak usage is about 11.3 GB.
- **Learning curve to judge whether 114 more identities is enough** (`runs/learning_curve/`, new `--max-train-identities`): V2 trained on 20 / 28 / 42 / 56 training identities per fold, with gender-balanced subsets and the same validation folds.
  - Summary script: `FLAG_2027/analysis/learning_curve.py` (epoch-90 EER, log-linear fit).
  - **Final, all 15 runs per size (5 folds × 3 seeds, epoch 90):**

    | Training identities | Standard EER | Gender EER | Mean EER |
    |---|---|---|---|
    | 20 | 25.09 | 37.55 | 31.32 |
    | 28 | 24.55 | 35.86 | 30.20 |
    | 42 | 23.87 | 35.00 | 29.43 |
    | 56 | 22.86 | 34.25 | 28.55 |

  - **About −1.6 EER points per doubling of identities on both tracks** (28→56: standard −1.66, gender −1.60, mean −1.63). The gender slope over 20→56 is −2.1, 95% CI [−2.9, −1.4].
  - Log-linear extrapolation (assumes in-domain identities):

    | Training identities | Standard EER | Gender EER |
    |---|---|---|
    | 184 | 20.6 | 30.5 |
    | 500 | 18.5 | 27.4 |
    | 1,000 | 17.0 | 25.2 |
    | 6,000 | 13.3 | 19.7 |

  - **Conclusion:**
    - MAV-Celeb v1 + v3 (about 170 training identities per fold) is expected to give about −2.5 to −3 points mean.
    - Reaching the leaderboard median (about −7) would need about 1,000+ identities with the current frozen-feature model.
    - Data alone is not enough. It should be combined with encoder fine-tuning or stronger features.
  - (The preliminary numbers from 2 runs per size suggested a steeper standard-track curve, −3.2 per doubling. That did not hold.)

### 9.3 Redesign and gender-track plan (16 Nov, after eval #3 = 28.84)

**Diagnosis.**
- About 15 head-side changes (losses, verifiers, geometry, layer pooling, extra voice inputs) were all within noise on the gender track.
- Only more identities moved it: −1.9 locally, −1.1 on CodaBench.
- The limit is the **frozen voice representation × the number of identities**, not the projector.

**Framework with full freedom:**
1. **Large-scale paired pretraining.** VoxCeleb1 + VoxCeleb2 dev, about 7k identities, faces and voices from the same videos.
   - Stream-processed (download → decode in memory → features → discard), so disk is not the limit.
   - Subsample about 20 clips per identity, since identities matter more than clips.
2. **Fine-tune the voice encoder** (WavLM upper layers or LoRA) with the cross-modal loss on non-Bengali speech only; keep FaRL frozen (§6.2).
   - At 70 identities fine-tuning would overfit; at thousands it is the standard approach.
3. **In-domain stage** on the 70 official identities (`--official-share`-style mixing, as now).
4. **Gender-aware training:** gender-pure batches plus a within-gender hard-negative curriculum, and/or per-gender projector experts gated by the frozen voice-gender estimate (accurate for 70/70 identities).
5. **Language robustness for the unheard Bangla cells:**
   - multilingual non-Bengali external speech (Urdu, Hindi, German);
   - a language-adversarial head on the voice side;
   - preferring the lower, acoustic WavLM layers.
6. **Scoring:** cosine + the fixed age/gender terms + a fixed-weight ensemble of seeds and architectures (every trial scored independently, within the rules).

**Cheap retests now that 287 identities per fold exist** (things that failed from overfitting at 56):
- audeering 1,024-d features as an extra voice input (approach A, never run);
- ECAPA concatenation;
- hyperbolic + external data;
- removing the gender subspace from the embeddings on the gender track;
- a seed ensemble of the final model;
- MAV-Celeb v2 (English + Hindi).

## 10. Changelog

### 2026-11-19 (later): Phase III, ECAPA–WavLM complementarity (`docs/phase3/00`–`07`)
- **Why C1 works (E1):**
  - B0's input predicts only 18% of ECAPA's variance on unseen people.
  - The unpredictable part is a language-stable speaker code: 14% EER in both languages.
  - **Controls:** random 192-d features and speaker-shuffled ECAPA both lose the entire Urdu gain, so it is not dimensionality or regularisation.
- **Fusion (E2):** bottleneck 32/64, per-block L2 and a residual adapter don't beat plain concatenation on the unheard language. The seed-1 English gains shrank to −0.4 n.s. over 3 seeds. Late fusion with B0 hurts unheard languages.
- **German confirmation holdout (E3, new protocol, frozen candidates):** C1 vs B0, German gender −2.35 [−4.38, −0.30], standard −1.19 (P = 0.97). **The gain generalises beyond Urdu.**
- **Duration/session (E4):** no interaction, so no gated fusion. **Traits (E5):** trait heads ≈ 0.1; terms ≈ 0.5, not double-counted.
- **Stabilisation (E6):** C1 × 3 seed ensemble: official −0.66 vs C1 (significant), Urdu −0.33 / −0.38 (significant), German −0.2.
- **Candidate eval #6** (not submitted): `submissions/eval6_ecapa_concat_x3_terms.zip` (md5 `4b6fdcd4…`); official + terms 25.32 (C1 25.73, B0 25.94).
- **Code (all off by default, B0/C1 bit-identical):**
  - `VoiceBlockFusion` (`cross_modal_model.py`); `--voice-fusion*` and `--shuffle-voice-block-start` (`train_cross_modal.py`);
  - multi-checkpoint `generate_cross_modal_submission.py`; `finalize_attribute_terms.py --ensemble`;
  - German support in `phase2/language_transfer_fusion.py`; `FLAG_2027/phase3/`.
  - 83 tests pass.
- **Bugs caught:**
  - an expert named `b0` silently replaced the reference in two fusion scripts; now rejected;
  - single-seed control crash, fixed;
  - guessed parameter counts in a draft doc, replaced with computed ones.

### 2026-11-19: Phase II, representation bottleneck analysis (in progress; `docs/phase2/`)
- **Gate 1, audio** (`docs/phase2/01_audio_representation.md`):
  - B0's voice input (WavLM all-layer mean ⊕ eGeMAPS) is a weak, session-dominated speaker representation. Same-gender EER across videos is 30.5% (LDA backend), 34% across languages, and 10% within one video.
  - **Frozen ECAPA-TDNN (VoxCeleb2) reaches 9.8% / 11.0%** (same / cross language, across videos), and cross-language R@1 is 0.96. WavLM fine-tuning is not warranted yet.
  - The GAMP-VF voice-clustering failure was a representation failure: ECAPA reaches ARI 0.95 with the same method.
- **Gate 2, faces** (`02_face_representation.md`): ArcFace (InsightFace) is at 1.8% same-gender EER, against FaRL (B0's input) at 12.3%, or 18.2% for similar-age impostors.
- **New caches** (frozen encoders, streamed from the kept zips with byte- and embedding-level checks):
  - `combined_all/train_voices_ecapa.csv`, `train_voices_xvector.csv`, `train_voice_durations.npy`;
  - `features/voice_v9_xvector_perclip/`;
  - ArcFace for external faces (running).
- **Finding:** the old official x-vector cache (`voice_v1_wavlm`) was batch-padded across files. Clips < 3 s differ (median cos 0.974); it was superseded by `voice_v9`.
- **Tool fix:** `compare_fold_runs.py` now uses each run's own face CSV (previously one shared CSV for all runs). Regression check: identical output on an earlier comparison.
- **Gate 3, cross-modal** (`03_cross_modal_alignment.md`):
  - Identity encoders hurt the face–voice link: ArcFace +10 EER, ECAPA alone +1.8.
  - B0's projector and loss are locally optimal (linear +2.45, deeper +1.1, multi-positive InfoNCE +0.96).
  - **ECAPA appended to B0's voice is neutral on English** (+0.05; −0.21 with terms).
- **Gate 4, fusion** (`04`): experts add nothing beyond a same-size B0 seed ensemble on English; mean fusion is as good as logistic; no gated/MoE fusion.
- **Gate 5, unseen language** (`05`). Urdu-holdout proxy, WavLM ⊕ eGeMAPS ⊕ ECAPA vs B0:
  - Urdu standard −2.06 [−3.15, −0.95], Urdu gender −2.49 [−3.93, −1.01], English −1.3;
  - the gap closes. **The first change that helps the unheard language.**
- **Candidate eval #5** (not submitted): `submissions/eval5_ecapa_concat_terms.zip` (md5 `77b9ca00…`), model `checkpoints/v5_ecapa_concat/`. Recommendation in `docs/phase2/06_final_recommendations.md`.
- **Mistakes caught and fixed:**
  - ArcFace streaming first mapped 14 MAV-Celeb rows to the Urdu copy of a frame (same video in two language folders). The byte-level hash check stopped the write; fixed and re-extracted, 0 mismatches.
  - Editing `run_all.sh` while it ran broke its final line; runs were unaffected.

### 2026-11-18: disk cleanup (user-approved)
- **Deleted the external audio work folders:** `external_data/mavceleb/work` (9.9 GB, v1 + v3 extracted clips) and `external_data/voxceleb2/work` (2.2 GB).
- Before deleting, the source zips in `raw/` were verified against `raw/md5sums.txt` (all OK) and kept.
- **Rebuild:** `FLAG_2027/external_data/prepare_mavceleb.py` / `prepare_voxceleb2.py`, about 20 min. This is only needed for new audio processing (e.g. voice-encoder fine-tuning); all extracted features are unaffected.
- About 12 GB is now free in the home quota.

### 2026-11-18: GAMP-VF results (full report: `docs/gamp_vf/04_results.md`)
- **Verdict: keep B0 (`full`).** No GAMP-VF configuration beats it significantly.
- **Same-gender partner sampling (`b3_sg`)** is the only consistent effect. 3 seeds: gender −0.41 [−1.01, +0.31], better on 5/5 folds; mean −0.25, n.s.
  - On the Urdu-holdout proxy the gain is heard-language only: English gender −0.48 (P = 0.95), Urdu +0.05; the gender gap widens by +0.53.
  - So it is not worth a submission; for Bangla the expected effect is ≈ 0.
- **Seed-1 screening (mean Δ vs B0):**
  - prototype InfoNCE +0.60; language-balanced prototypes +0.21;
  - most-confusable hard negatives +0.06; same-gender most-confusable 0.00; same at ratio 1.0 +0.31;
  - same-gender similar-age −0.06 (3 seeds: +0.02); B4 −0.07.
- **Hidden-association simulation (77 hidden + 39 calibration external people):**
  - Voice clustering of unseen people is poor (ARI ≈ 0.27, about 7 groups per person).
  - Cross-modal matching is the real limit: with *perfect* grouping only 24% top-1 and 34% Hungarian, 57% precision at 10% coverage. No mode reached usable pseudo-label precision.
  - The H people's *true* labels add nothing measurable (oracle +0.58 vs teacher, n.s.). Students are within noise. Iterations 2–3 were not run.
- **Cross-lingual ablation:** language losses move same-speaker cross-language pairs and other-speaker pairs closer equally, so speaker discrimination is unchanged (AUC 0.793 → 0.796).
- **Runs:** 95 GAMP-VF runs, 522 MB. All stage outputs are in `results/gamp_vf/stage_*`.

### 2026-11-18: GAMP-VF (gender-aware multilingual prototype learning), setup
- **Docs:** `docs/gamp_vf/01_repository_audit.md` (verified data and code audit) and `02_architecture.md`. Results per stage go to `results/gamp_vf/stage_*`, never overwritten.
- **Audit findings:**
  - no Hindi in the data;
  - all 301 identities are labelled, so pseudo-labels and matching can only be studied by *hiding* labels (simulation);
  - the official validation is English-only;
  - the challenge rules forbid any grouping or matching on eval data, so all of it runs on training data only.
- **Code:**
  - new package `FLAG_2027/gamp_vf/` (prototypes, clustering, gender-aware hard-negative sampler, similarity, pseudo-labels, matching, mining, diagnostics);
  - new options in `train_cross_modal.py`, all off by default. A 2-epoch check is bit-identical to the old trainer and to the 16 Nov `full` run;
  - `trait_heads.py` masks unknown gender (no-op otherwise);
  - backups in `results/gamp_vf/code_backups/`.
- **Tests:** 32 new and 25 existing, all pass.
- **Stage A/B diagnostics** (`results/gamp_vf/stage_A_B/`, P2 fold models, held-out MAV-Celeb v1 people never seen and Urdu never trained on):
  - Same speaker across English/Urdu is far more similar than different speakers in the projected space (0.73 vs 0.48; AUC 0.80 against same-language other speakers). Raw WavLM barely separates speakers (AUC 0.67).
  - **Voice clustering of unseen people is poor** (best ARI 0.38). Faces cluster better (raw FaRL ARI 0.75).
  - Voice → face identity retrieval among about 13 people: single clip 33% top-1, uniform voice prototype 42%, language-balanced 47%, English-only 47%, Urdu-only 34% (n = 64 people, so ±3 identities is noise).
- **Runs:** `runs/gamp_vf/` (`run_all.sh`).

### 2026-11-16
- Wrote the detailed merge plan for the external data (§9.2, steps 1–7).
- **Downloads started.** The Drive quota had reset, so `external_data/mavceleb/download.sh` (curl with resume and size check) now runs at about 43 MB/s.
  - **MAV-Celeb v3:** complete (1,693,351,345 bytes).
    - 3,017 clips: 1,422 English + 1,595 German, 11.9 h.
    - 50 identities.
    - 16 kHz mono wav and 224 × 224 face jpgs, the same format as the official data.
    - Median clip 8.7 s (official 4.7 s).
    - 41,650 face frames in the zip, against 3,017 listed pairs.
  - v1 and v2 are downloading.
- New scripts:
  - `FLAG_2027/external_data/prepare_mavceleb.py`: extracts only the listed files from the zip into `external_data/mavceleb/work/<v>` and writes an official-format manifest. Identities are namespaced `mav3_id0001`; class indices start at 1000 (v3).
  - `FLAG_2027/external_data/language_id.py`: SpeechBrain `lang-id-voxlingua107-ecapa` @ `0253049a` (Apache-2.0), first 60 s per clip. A clip is dropped if Bengali is top-1 or p(Bengali) ≥ 0.05.
  - `features/external/extract_source.sh`: V2 features with the identical FaRL and WavLM settings.
- **v3 language check: 0 of 3,017 clips dropped.**
  - Top-1 counts: German 1,571, English 958, Welsh 269 (a known VoxLingua confusion for English), Luxembourgish/Dutch 29 each.
  - Max p(Bengali) is below 0.05 everywhere.
- **MAV-Celeb v1 downloaded** (11,301,778,303 bytes, md5 `8389b219…`; v3 md5 `78a715be…`; both in `raw/md5sums.txt`).
  - 18,038 listed clips, 64 identities.
  - Work copy of the listed files: 8.6 GB.
- **v2 hit the Drive quota again** (an HTML error page instead of the file). Removed; to retry later or download in the browser.
- **v1 language check: 142 of 18,038 clips dropped (0.8%).**
  - Listed as English (80) or Urdu (62), over 43 identities, at most 5% of any identity.
  - Bengali was top-1 for only 19; median p(Bengali) of the dropped clips is 0.10.
  - These look like scattered false alarms (Urdu is labelled Hindi 6,427 times), not Bengali speakers. All are dropped anyway.
- New `FLAG_2027/external_data/build_external_training_set.py` merges sources into `features/external/combined_<set>/`. It reads no dev or eval data.
  - **Language:** drops the flagged clips.
  - **Face outliers:** drops frames whose FaRL similarity to their identity's face centroid is below the 1st percentile of the official training rows (0.714). This removes wrong-person frames, e.g. a young woman's frame inside a male identity.
  - **Gender per identity:** InsightFace face-gender majority + a voice-gender logistic regression fitted on the official training WavLM features. Disagreements are resolved by manual inspection (`features/external/gender_overrides.json`): `mav3_id0031` = f (older woman, low voice), `mav3_id0027` = m.
  - **Duplicates:** FaRL face centroids, with the threshold calibrated on official same-identity halves vs different identities (0.939). Duplicates of official identities are dropped; external–external duplicates are merged.
- **v3 final: 2,950 clips, 50 identities (34 m / 16 f), 67 face outliers dropped, no duplicates.** The closest official identity has similarity 0.833; every official same-identity pair is at least 0.885.
- Training support for external data in `train_cross_modal.py` (all off by default, so V2 is unchanged):
  - `--extra-face-csv`, `--extra-voice-csv`, `--extra-manifest`, `--extra-gender-metadata` append external identities to the training side only;
  - `--official-share` sets the fraction of each batch's identities drawn from the official ones (new `primary_share` in `IdentityBalancedBatchSampler`).
  - Runs use `--batches-per-epoch 40`, the same number of optimisation steps as V2.
- Started **V2 + v3** (`runs/external/run_all.sh v3`): configurations `share50` and `natural`, 5 folds × 3 seeds. v1 features are still extracting.
- **VoxCeleb2 test set downloaded** from the gated HF mirror `ProgramComputer/voxceleb`, now that the user has access (`external_data/voxceleb2/download.sh`).
  - `vox2_test_mp4.zip`: 8,906,971,319 bytes, md5 `b3c555e7…`.
  - `vox2_meta.csv`: identity, gender, set. The test set has 39 f / 81 m in the metadata.
  - The zip holds 118 identities, 4,911 videos and 36,237 clips: 224 × 224 face-track mp4 at 25 fps, 16 kHz mono audio, about 4 s per clip.
  - VoxCeleb2 frames are framed tighter on the face than the official crops. This is a possible domain difference, to check if VoxCeleb2 does not help.
- New `FLAG_2027/external_data/prepare_voxceleb2.py` writes an official-format manifest and the gender file from the metadata.
  - Up to 80 clips per identity, round-robin over its videos.
  - Each clip's audio is written as 16 kHz wav.
  - The paired face is the middle frame of a clip from a *different* video of the same identity, as in MAV-Celeb's pairing, so the model cannot match through a shared recording.
- VoxCeleb2 test converted: 9,176 clips, 118 identities (79 m / 39 f, gender from `vox2_meta.csv`); work copy 2.2 GB. Language ID and features started.
- **Submission plan agreed with the user:**
  1. Local five-fold test of V2 + external data.
  2. Only if it beats V2 + age/gender terms locally: refit the terms with the new fold models.
  3. Train the final model on all 70 official + all accepted external identities with the winning settings.
  4. Build the eval zip (the existing eval FaRL/WavLM features are reused) and submit.
  - Expected from the learning curve: about −3 points at best (70 → ~290 identities ≈ 2 doublings), less if an external identity is worth less than an official one.
- `train_cross_modal_full.py` gained `--extra-face-csv`, `--extra-voice-csv` and `--official-share`. External class indices (10000+) must not overlap official ones. Off by default.
  - Smoke-tested: 9,435 rows, 120 identities.
  - The final model uses `--batches-per-epoch 50` (official rows // 128, as V2's full model), so the step count matches V2.
- **Incident:** the internet connection dropped mid-session.
  - The background jobs survived the drop itself, but **all of them were killed at about 10:27**, when the old session's processes were torn down. No crash or out-of-memory.
  - Kept:
    - 16/30 V2 + v3 runs;
    - v1 language ID and face features;
    - VoxCeleb2 language ID and face features.
  - Lost: v1 WavLM features (30%) and v1 face age (18%).
  - **Relaunched at 11:35 with `setsid nohup`**, detached from the Claude session so a disconnect can no longer kill them.
  - `run_all.sh` skips finished runs; `extract_source.sh` now skips steps whose output exists.
- **VoxCeleb2 language check: 9 of 9,176 clips dropped.** Top-1: English 7,239, Welsh 1,015 (the English confusion), French 114, Hindi 75, Korean 67.
- **Freed 33 GB of home-directory quota** (user-approved). The home directory went from 95 GB to 63 GB, leaving about 47 GB free (against the ~14 GB before). Deleted:
  - the LLaVA-Video-7B (16.1 GB) and SigLIP (3.5 GB) Hugging Face caches, from another project and re-downloadable;
  - an old VS Code server backup from 11 Nov (7.3 GB);
  - an unused Python venv `~/flag2027/.venv` (6.0 GB).
- The official challenge zips and the conda package cache were kept.
- There is now space to keep both MAV-Celeb zips (13 GB) and a slim raw copy.
- **Result: V2 + MAV-Celeb v3 (50 extra identities, 56 → 106 per fold), paired against V2, 3 seeds × 5 folds** (`runs/external/compare_base_vs_v3_{share50,natural}.json`):

  | Config | Standard EER | Gender EER | Mean EER | Mean Δ (95% CI) | P(better) |
  |---|---|---|---|---|---|
  | V2 (baseline) | 22.88 | 34.17 | 28.52 | — | — |
  | v3, `share50` (half of each batch official) | 21.70 (−1.17) | 34.00 (−0.16) | 27.85 | −0.67 [−2.23, +0.98] | 0.79 |
  | v3, `natural` (identities sampled uniformly) | 21.70 (−1.17) | 34.16 (−0.00) | 27.93 | −0.59 [−2.22, +0.89] | 0.80 |

  - Both configurations point the right way but are **not significant**. The two batch mixes are practically the same.
  - The learning curve predicted about −1.5 on both tracks for 56 → 106 identities. The **standard track** gained about as much as predicted (−1.2, P = 0.92–0.94).
  - The **gender track did not move**. External identities seem to help the "easy" across-gender separation more than the within-gender matching. A possible reason is that v3 adds mostly male identities (34 m / 16 f).
  - Not enough on its own to submit; the larger sets (v3 + VoxCeleb2, then + v1) decide.
- **The process watchers never fired.** Their `pgrep -f` pattern also matched the watcher's own command line. They were replaced by watchers that wait on the job's PID.
- **Built `combined_v3vox2`:** MAV-Celeb v3 + VoxCeleb2 test.
  - VoxCeleb2 gender comes from `vox2_meta.csv`. The build script now uses `features/external/<source>_gender.csv` when it exists, and reports metadata/voice disagreements instead of voting.
  - VoxCeleb2: 9 clips dropped by language ID, 145 face outliers dropped.
  - Duplicate checks, all inspected by eye (face strips):
    - `vox2_id00154` vs official `id067` (0.922, below the 0.939 threshold): two different men, kept apart. No duplicates of official identities.
    - `mav3_id0043` = `vox2_id07494` (0.973): the same person (a tennis player), merged.
    - `vox2_id01000` vs `vox2_id05459` (0.946): different women, so the merge was rejected in the new `features/external/merge_overrides.json`.
    - `vox2_id02286`: the metadata says f, the voice classifier says male (0.88). It is a woman with a low voice, so f is kept.
  - **Final: 11,972 clips, 167 external identities (112 m / 55 f), median 78 clips per identity.** Per fold: 56 official + 167 external = 223 training identities.
- Started **V2 + v3 + VoxCeleb2** five-fold runs (`runs/external/run_all.sh v3vox2 share50 natural`, 30 runs, about 45 min, detached).
- **Result: V2 + v3 + VoxCeleb2 (167 extra identities, 56 → 223 per fold), paired against V2, 3 seeds × 5 folds** (`runs/external/compare_base_vs_v3vox2_{share50,natural}.json`; all 30 runs reached 90 epochs):

  | Config | Standard EER | Gender EER | Mean EER | Mean Δ (95% CI) | P(better) |
  |---|---|---|---|---|---|
  | V2 (baseline) | 22.88 | 34.17 | 28.52 | — | — |
  | v3 + VoxCeleb2, `share50` | **21.06 (−1.82, CI [−3.25, −0.37])** | 33.17 (−1.00, CI [−2.85, +0.67]) | **27.11** | **−1.41 [−2.98, +0.08]** | 0.97 |
  | v3 + VoxCeleb2, `natural` | 21.54 (−1.34) | 33.32 (−0.85) | 27.43 | −1.09 [−3.25, +1.01] | 0.86 |

  - **`share50` is the best external-data configuration so far.** The standard track gain is significant, the first significant gain over V2 from external data. The mean gain is borderline (P = 0.97; the 95% interval just touches 0).
  - The **gender track now moves too** (−1.0, P = 0.88), unlike with v3 alone, though it is not significant.
  - `share50` > `natural` once external identities outnumber official ones (167 vs 56): keeping half of every batch official helps.
  - Per fold (`share50`), standard / gender EER:
    - fold 1 improves a lot (25.9 → 19.0 / 36.7 → 30.2);
    - folds 0 and 2 improve slightly;
    - fold 3 standard improves 27.3 → 23.8;
    - **fold 4 gets worse** (21.4 → 23.8 / 34.7 → 37.0).
    - The gain is not uniform over held-out identities.
  - **Less than the learning curve predicted.** 2 doublings of identities predicted about −3.2; we got −1.4. One external identity is worth roughly half an official one, as expected from the domain shift (VoxCeleb2's tighter face crops, other languages, different recordings).
  - This comparison is against V2 *without* the age/gender score terms. The submission plan requires beating V2 + terms; the terms have to be refit on the new fold models before that comparison.
- **Age/gender terms refit on the v3 + VoxCeleb2 `share50` fold models** (`finalize_attribute_terms.py`, training data only; `runs/external/v3vox2_share50_attribute_terms.json`):
  - weights: standard track w_age 0.05 / 10 years, w_gender 0.05; gender track w_age 0.02, w_gender 0;
  - five-fold mean EER with terms, against V2 + terms (`checkpoints/v3_candidates/v2_attribute_terms.json`):

    | System | Standard | Gender | Mean |
    |---|---|---|---|
    | V2 + terms (submitted as eval #2, 30.17 on CodaBench) | 21.36 | 33.94 | 27.65 |
    | **V2 + v3 + VoxCeleb2 (`share50`) + terms** | **20.29 (−1.07)** | **32.74 (−1.20)** | **26.51 (−1.14)** |

  - **It passes the submission criterion** (beats V2 + terms locally on both tracks). Caveats:
    - this difference has no paired bootstrap interval (`compare_fold_runs.py` does not apply terms);
    - both rows choose their term weights on the same folds they are scored on, so both are slightly optimistic in the same way.
  - The terms still add on top of the external data (standard −0.8, gender −0.4).
- Trained the **final v3 + VoxCeleb2 model as a fallback submission** while v1 finishes: `checkpoints/v3vox2/full_data`. Settings:
  - all 70 official + 167 external identities;
  - V2 settings, `--official-share 0.5`, `--batches-per-epoch 50`, 90 epochs.
  - 18,457 rows, 237 identities; md5 `7755cdd0…`.
- **Built the fallback evaluation zip** `submissions/eval3_v3vox2_terms.zip` (md5 `43827289…`, copy and README in `checkpoints/v3vox2/`).
  - Same eval features and attribute files as eval #2, with the new model and the refit terms.
  - Passed the generator's format validation; the trial order is identical to eval #2.
  - Score correlation with eval #2: 0.93 on the standard track, 0.78–0.82 on the gender track. The gender-track scores change more.
  - **Not submitted.** It waits for the v1 result and the user's go-ahead.
- **MAV-Celeb v1 features finished (13:48):** WavLM voices, FaRL faces and InsightFace face age, 18,038 rows.
- **Built `combined_all`** (v3 + VoxCeleb2 + v1):
  - v1: 142 clips dropped by language ID, 166 face outliers dropped.
  - The gender vote set aside 2 v1 identities: the face said mixed, the voice said clearly female (p_male 0.03).
    - Inspected by eye: `mav1_id0054` (a woman cricketer) and `mav1_id0069` (a woman with glasses and a headscarf) are both women.
    - Both were added as f to `gender_overrides.json`.
  - No duplicates of official identities. The closest v1 pair, `mav1_id0066` vs official `id039` (0.90), is two different men.
  - No new external merges.
  - **Final: 29,702 clips, 231 external identities (152 m / 79 f).** Per fold: 56 official + 231 external = 287 training identities.
- Started the **V2 + all external data** five-fold runs (`runs/external/run_all.sh all share50 natural`, 30 runs, about 45 min, detached).
  - The first PID watcher fired at once: `pgrep -f` matched the launching shell's own command line again. Re-attached to the real `run_all.sh` PID.
- **Result: V2 + all external data (231 extra identities, 56 → 287 per fold), paired, 3 seeds × 5 folds** (`runs/external/compare_*_all_*.json`; all 30 runs reached 90 epochs):

  | Config | Standard EER | Gender EER | Mean EER | Mean Δ vs V2 (95% CI) | P(better) |
  |---|---|---|---|---|---|
  | V2 | 22.88 | 34.17 | 28.52 | — | — |
  | v3 + VoxCeleb2, `share50` | 21.06 | 33.17 | 27.11 | −1.41 [−2.98, +0.08] | 0.97 |
  | **all, `share50`** | **21.18 (−1.70, CI [−3.32, −0.29])** | **32.28 (−1.89, CI [−3.83, +0.02])** | **26.73** | **−1.79 [−3.45, −0.26]** | **0.99** |
  | all, `natural` | 21.01 (−1.87) | 33.06 (−1.10) | 27.04 | −1.49 [−3.52, +0.64] | 0.91 |

  - **First external-data gain over V2 that is significant on the mean** (the interval excludes 0). The gender track is borderline (P = 0.97).
  - Adding v1 to v3 + VoxCeleb2 (paired, `share50`): standard +0.12 (no change), **gender −0.90 [−1.98, +0.37], P = 0.90**, mean −0.39.
    - v1 helps the gender track. It is mostly Urdu/English South-Asian speakers, closer to the official data than VoxCeleb2.
  - `share50` beats `natural` on the gender track (−0.8) when external data is large. Keeping half of every batch official matters.
  - Per fold (`share50`), standard / gender EER:
    - fold 0: 18.1 / 28.7; fold 1: 19.9 / 31.1; fold 2: 20.4 / 30.9; fold 3: 22.9 / 34.2; fold 4: 24.7 / 36.5.
    - Fold 4 is still worse than V2 (21.4 / 34.7).
  - Total gain from V2 (56 → 287 identities, 2.4 doublings): −1.8, against about −3.8 predicted by the official-data learning curve. One external identity is worth roughly 0.4–0.5 of an official one.
- **Terms refit on the `all_share50` fold models** (`runs/external/all_share50_attribute_terms.json`):
  - weights: standard w_age 0.05, w_gender 0.05; gender w_age 0.03, w_gender 0;
  - five-fold with terms: **standard 20.42, gender 32.20, mean 26.31**, against V2 + terms at 27.65 (−1.34) and v3 + VoxCeleb2 + terms at 26.51.
  - On the gender track the terms now add little (−0.08); the external data does the work there.
- **Final model trained** (`checkpoints/v3all/full_data`): all 70 official + 231 external identities, 36,187 rows, 301 identities; `share50`, 50 batches/epoch, 90 epochs; md5 `01cee1aa…`.
- **Evaluation zip `submissions/eval3_all_terms.zip`** (md5 `ffc4a41a…`, copy and README in `checkpoints/v3all/`).
  - Validated; the trial order is identical to eval #2.
  - Score correlation with eval #2: 0.93 on the standard track, 0.76–0.80 on the gender track.
  - **This is the recommended next submission.** Not submitted; the user uploads it to CodaBench.
  - Expected CodaBench score, if the local gain transfers as it did for eval #2 (local −1.1 → CodaBench −1.1): about 30.17 − 1.3 ≈ **28.8–29**.
- **First upload rejected:** the zip had been re-compressed in macOS Finder, which added `__MACOSX/._*` copies and an `eval3_all_terms/` top folder, so CodaBench saw 2 files per cell. The user re-uploaded the original zip.
- **Eval #3 (V2 + all external data + terms) scored 28.84%**, against 30.17% for eval #2:

  | Cell | Eval #2 | Eval #3 | Δ |
  |---|---|---|---|
  | no_gender English (heard) | 24.49 | 22.96 | −1.53 |
  | no_gender Bangla (unheard) | 22.69 | 21.20 | −1.49 |
  | gender English (heard) | 36.06 | 34.62 | −1.44 |
  | gender Bangla (unheard) | 37.43 | 36.59 | −0.84 |
  | **Average** | **30.17** | **28.84** | **−1.33** |

  - **The local prediction transferred exactly again:** local −1.34, CodaBench −1.33, and the predicted range was 28.8–29.
  - The gender Bangla cell gains least and is now the worst cell. Gender heard→unheard gap +2.0; standard gap −1.8.
  - Leaderboard reference (13 Nov): median 22.47, top 14.30.
- **Started the voice → traits → face architecture** (§9.3; user request). Everything is built on top of eval #3, i.e. `all_share50` settings plus the additions, so every result is a paired comparison against what was submitted.
  - **Data check:** 113 external identities speak two languages: 64 MAV-Celeb v1 people (English + Urdu) and 49 MAV-Celeb v3 people (English + German). This is the supervision for "one face, several languages".
  - **eGeMAPS v02** (openSMILE 2.5.1, 88 functionals: F0 level/range/slopes, loudness, jitter, shimmer, HNR, H1–H2, alpha ratio, Hammarberg index, spectral slopes (vocal effort), formants F1–F3, temporal features).
    - New `FLAG_2027/feature_extraction/extract_egemaps.py`, run by `features/egemaps/extract_all.sh` over official train, the 3 external sources and the 4 eval trial lists. No Bengali is used for training: eval clips are only *encoded* for scoring.
    - New `build_wavlm_egemaps.py` makes 856-d voice inputs (WavLM 768 + eGeMAPS 88). The eGeMAPS block is median-imputed and clipped to the official-training [0.5, 99.5] percentiles.
  - **Trait labels** (new `FLAG_2027/analysis/build_trait_labels.py`), person-level, from frozen face models only:
    - age: median InsightFace age (VoxCeleb2 face ages now being estimated);
    - build: mean FaRL zero-shot "heavy-set vs slim" score (computed for all 29,702 external rows from the stored FaRL embeddings).
  - **New training-only modules** (`FLAG_2027/trait_heads.py`), stored under `aux_state_dict` so the scoring model and all evaluation/submission tools are unchanged:
    - `TraitHeads`: *shared* linear heads (age, build, gender) applied to both the face and the voice embedding, so both modalities place traits along the same directions;
    - `LanguageAdversary`: gradient reversal + MLP predicting English/Urdu/German from the voice embedding, class-balanced;
    - `cross_lingual_alignment`: 1 − cos between voices of one person in two languages.
  - **`train_cross_modal.py` options** (all off by default): `--trait-csv`, `--extra-trait-csv`, `--trait-weight`, `--trait-targets`, `--language-adversary-weight`, `--language-adversary-scale`, `--cross-lingual-weight`, plus `--exclude-train-languages` and `--holdout-external-prefix` for the language protocol.
  - **New unheard-language proxy** (`FLAG_2027/local_validation/evaluate_language_transfer.py`):
    - per fold, one fifth of the MAV-Celeb v1 people (13) are held out, and **no Urdu at all** is used in training;
    - their English clips are "heard" and their Urdu clips "unheard" (Urdu stands in for Bengali);
    - fixed trials per fold: 3 same-person faces + 3 impostor faces per voice clip, any gender or same gender;
    - EER per track and language, with a paired identity bootstrap.
  - Smoke test (fold 0, 3 epochs, `lang`): trains; the cross-lingual distance falls 0.79 → 0.73; checkpoints keep the aux heads.
  - **Runs** (`runs/traits/run_all.sh`; 3 seeds × 5 folds each):
    - P1 (official five-fold, decides submissions): `lang`, `traits`, `traits05`, `egemaps`, `full`;
    - P2 (language proxy): `p2_base`, `p2_lang`, `p2_full`.
    - `lang`, `p2_base` and `p2_lang` started at 15:40. The rest start once eGeMAPS and the VoxCeleb2 face ages finish.
  - eGeMAPS finished for all 39k clips with 0 failures, and VoxCeleb2 face ages for 9,176 images with 0 failures (16:01).
    - The trait labels and the 856-d inputs are built (`features/voice_v8_wavlm_egemaps/`, `combined_all/train_voices_egemaps.csv`, `features/eval_v8/`).
    - No near-constant eGeMAPS column. Eval values stay within the clipped training range (99th percentile |z| = 2.9).
    - `traits`, `traits05`, `egemaps`, `full` and `p2_full` started at 16:02.
- **Result: language invariance (`lang` = adversary 0.1 + cross-lingual 0.1)**, 3 seeds × 5 folds:
  - **Official five-fold, paired against eval #3's fold runs (`all_share50`):** standard −0.27, gender +0.10, **mean −0.08 [−0.32, +0.18]**: no change (`runs/traits/compare_all_share50_vs_lang.json`).
  - **Unheard-language proxy** (13 held-out v1 people per fold, no Urdu in training; about 106k trials; `runs/traits/language_transfer_p2_base_vs_lang.json`):

    | | Standard English (heard) | Standard Urdu (unheard) | Gap | Gender English | Gender Urdu | Gap |
    |---|---|---|---|---|---|---|
    | `p2_base` | 25.13 | 26.34 | **+1.21** | 37.39 | 38.74 | **+1.36** |
    | `p2_lang` | 25.23 | 26.20 | +0.97 | 37.62 | 38.67 | +1.05 |

  - **The proxy reproduces the CodaBench pattern.** An unheard language costs about +1.2 to +1.4 EER points (CodaBench gender track: +2.0), so the proxy is a reasonable stand-in for Bengali.
  - **Language invariance narrows the gap by only 0.2–0.3 points** (P = 0.80–0.92, not significant). It does so mostly by making the heard language slightly worse, not the unheard one better.
  - **Conclusion:** the voice embedding is already largely language-independent, so the language gap is not the bottleneck. This matches the leaderboard analysis: within-gender discrimination is the problem in both languages.
- **Incident (16:15): home quota full.** Runs crashed with "PytorchStreamWriter failed writing file" (`Disk quota exceeded`; `du` = 100 GiB).
  - 33 runs had finished; 9 partial runs crashed.
  - **Freed without deleting:** moved the 171 unused `checkpoint_best.pth.tar` files of today's runs (839 MB) to the RAM disk `/dev/shm/flag_best_checkpoints_2026-11-16/`. Every result uses `checkpoint_last`, since picking "best" on the validation folds would bias the estimate. These files are lost on reboot.
    - **Update 2026-11-18:** the machine rebooted at 13:41, so these 171 files are gone. No result depends on them. Home usage is now 98 GB of the ~100 GiB quota, with about 1.5–2 GB of headroom.
  - New `--no-best-checkpoint` flag (used by `runs/traits/run_all.sh`) halves the disk use per run.
  - Relaunched with 6 in parallel; finished runs are skipped, partial runs restarted.
  - The largest regenerable items are the external audio work folders (12.2 GB: v1 8.6, VoxCeleb2 2.2, v3 1.4), rebuildable from the kept zips. **Deleting them needs the user's OK.**
- **Preliminary, seed 1 only (5 folds), paired against `all_share50` seed 1** (`runs/traits/prelim_seed1_*.json`):

  | Config | Standard | Gender | Mean | P(better) |
  |---|---|---|---|---|
  | `all_share50` (eval #3), seed 1 | 21.37 | 32.77 | 27.07 | — |
  | `traits` (trait weight 0.2) | 21.24 (−0.13) | 32.77 (0.00) | 27.01 (−0.06) | 0.81 |
  | `traits05` (trait weight 0.5) | 20.99 (−0.39) | 32.79 (+0.01) | 26.89 (−0.19) | 0.85 |
  | `egemaps` (WavLM + eGeMAPS input) | 20.77 (−0.60) | 31.99 (−0.79) | 26.38 (−0.69) | 0.80 |
  | **`full`** (eGeMAPS + traits 0.2 + language) | **20.27 (−1.10)** | **31.59 (−1.19)** | **25.93 (−1.14)** | 0.94 |

  - **Caveat:** ECAPA showed a similar seed-1 gain that vanished with seeds 2–3 (§6.12). The decision waits for all 3 seeds.
- **Final results: voice → traits → face, 3 seeds × 5 folds, paired against eval #3's fold runs (`all_share50`)** (`runs/traits/compare_all_share50_vs_*.json`; all 75 runs reached 90 epochs):

  | Config | Standard | Gender | Mean Δ (95% CI) | P(better) |
  |---|---|---|---|---|
  | eval #3 (`all_share50`) | 21.18 | 32.28 | 26.73 | — |
  | `lang` (adversary + cross-lingual) | 20.91 (−0.27) | 32.38 (+0.10) | −0.08 [−0.32, +0.18] | 0.71 |
  | `traits` (0.2) | 20.95 (−0.23) | 32.27 (0.00) | −0.12 [−0.26, +0.04] | 0.92 |
  | `traits05` (0.5) | 20.89 (−0.29) | 32.27 (0.00) | −0.15 [−0.45, +0.08] | 0.92 |
  | `egemaps` | 20.60 (−0.59) | 32.20 (−0.07) | −0.33 [−1.03, +0.56] | 0.65 |
  | **`full`** (eGeMAPS + traits 0.2 + language) | **20.34 (−0.84)** | **31.97 (−0.31)** | **−0.57 [−1.38, +0.23]** | **0.90** |

  - **As feared, the seed-1 gain roughly halved with seeds 2–3** (`full` mean −1.14 → −0.57; gender −1.19 → −0.31).
  - `full` is the best configuration and points the right way on both tracks, but it is **not significant** against eval #3.
  - **The parts add up:** `full` beats `egemaps` alone by −0.25 [−0.62, −0.05] (P = 0.99), so the trait heads and language losses help *on top of* eGeMAPS, though not without it.
  - Per fold (`full`, standard / gender), against eval #3:
    - fold 0: 17.3 / 27.7 vs 18.1 / 28.7;
    - fold 1: 18.6 / 29.9 vs 19.9 / 31.1;
    - fold 2: 20.6 / 30.8 vs 20.4 / 30.9;
    - fold 3: 22.3 / 33.6 vs 22.9 / 34.2;
    - fold 4: 22.9 / 37.8 vs 24.7 / 36.5.
    - Standard improves on 4/5 folds; gender on 3/5, while fold 4 gender is worse.
  - **With age/gender terms** (refit on the `full` fold models, `runs/traits/full_attribute_terms.json`): standard **19.97**, gender **31.92**, mean **25.94**, against eval #3 + terms at 20.42 / 32.20 / 26.31: **−0.37**. Expected CodaBench about 28.84 − 0.4 ≈ **28.4–28.5**.
- **Unheard-language proxy, final** (`runs/traits/language_transfer_p2_base_vs_{lang,full}.json`):
  - **Bug fixed first:** impostor identities were drawn from an unsorted Python `set`, so the trials changed between invocations. The paired differences were valid, but the absolute numbers drifted by about 0.4. Now deterministic.

  | | Standard English / Urdu | Gap | Gender English / Urdu | Gap |
  |---|---|---|---|---|
  | `p2_base` | 24.42 / 26.56 | +2.14 | 37.19 / 39.11 | +1.92 |
  | `p2_lang` | 24.41 / 26.34 | +1.93 | 37.47 / 38.93 | +1.46 |
  | `p2_full` | 24.11 / 25.89 | +1.78 | 37.30 / 38.71 | +1.41 |

  - The proxy's unheard-language cost (+2.1 / +1.9) now **matches CodaBench's Bangla gender gap (+2.0)** closely.
  - `full` improves the **unheard** language most: Urdu standard −0.67 [−1.12, −0.10] (P = 0.99), Urdu gender −0.40 (P = 0.86). The heard language is unchanged, so the gap shrinks by 0.4–0.5 (n.s.).
- **Final model and zip (candidate eval #4, not submitted):**
  - model `checkpoints/v4_traits/full_data` (md5 `26cc6e8d…`; 36,187 rows, 301 identities, 856-d voice input);
  - zip `submissions/eval4_traits_terms.zip` (md5 `74d5e141…`), validated, with the eval eGeMAPS in `features/eval_v8`;
  - score correlation with eval #3: 0.97 standard, 0.92–0.94 gender, i.e. a small change, consistent with the small expected gain;
  - README in `checkpoints/v4_traits/`.
  - `train_cross_modal_full.py` gained the same auxiliary options (`--manifest`, `--gender-metadata`, `--extra-manifest`, `--extra-gender-metadata`, `--trait-*`, `--language-adversary-*`, `--cross-lingual-weight`).
- v1 features are still extracting (WavLM 42%, face age 55% at 12:42).

### 2026-11-13 (later)
- Disk space about 14 GB, so the external data is reduced to MAV-Celeb v3 + v1, processed zip-by-zip without extraction.
  - New `train_cross_modal.py --max-train-identities N`: a gender-balanced identity subset drawn with seed × 1000 + fold. Off by default, so V2 is unchanged.
  - Started the learning-curve experiment (`runs/learning_curve/run_all.sh`, 45 runs).
  - **Learning curve finished:** about −1.6 EER per doubling of identities on both tracks. v1 + v3 is expected to give about −3. The median needs about 1,000+ identities with frozen features (§9.2).
  - Disk check: `/cache` is full (850 MB), `/export` is not writable (`ipbstaff` group), and `/tmp` is on the 89%-full system disk.
    - Plan: keep a slim FLAC copy of the training-list files so features can be re-extracted and encoders fine-tuned later.
    - The user decides whether to free about 18 GB of unused HF model caches (LLaVA-Video-7B, SigLIP) and the 2.8 GB of already-unpacked official zips.
- **External data plan (§9.2).**
  - Checked availability.
    - MAV-Celeb v1/v2/v3 training archives (11.3 / 13.5 / 1.7 GB; 64 and 50 training identities in v1/v3) are in the public Drive folder. Both gdown and the direct endpoint return "quota exceeded", so a browser download or Drive copy is needed.
    - VoxCeleb2 (mp4 with audio, 118 test + 5,994 dev identities) is on HF `ProgramComputer/voxceleb`. It is gated, and our token is not yet authorised.
  - Saved the MAV-Celeb split lists to `external_data/mavceleb/splits/` and the download script to `external_data/mavceleb/download.sh`.
- **Eval #2 scored 30.17%** (V2 + terms), −1.1 against V2, matching the local estimate. Leaderboard check: rank 32 of 34; median 22.47; top 14.30. The dev leaderboard top is ~0.5% (implausible).
- Built evaluation submission #2 `submissions/eval2_v2_terms.zip` (md5 `94746adf…`): V2 + fixed bias-corrected age/gender terms. Validated against the official trial IDs; the added terms average 0.11 (no_gender) and 0.01 (gender). Details in `checkpoints/v3_candidates/README.md`.
- ECAPA final: `wavlm_ecapa` +0.3 mean (no gain, unstable); `ecapa_only` +3.2 (worse). Dropped.
- Fixed the age/gender terms for V2 from training data only (`checkpoints/v3_candidates/v2_attribute_terms.json`): voice-age correction men face ≈ 0.56·voice + 15.3, women ≈ 0.38·voice + 14.3; standard track w_age 0.08/10 yrs, w_gender 0.1; gender track w_age 0.01, w_gender 0. New `analysis/finalize_attribute_terms.py`; the submission script gained `--attribute-terms`, `--voice-attribute-root` and `--face-attribute-root` (output byte-identical without them). Eval-set voice age/gender (`features/eval_agegender/`) and face age (`features/eval_face_age/`) extracted (`features/extract_eval_attributes.sh`); every face detected.
- ECAPA preliminary (seed 1): `wavlm_ecapa` −0.6 mean (gender −0.5), n.s.; `ecapa_only` +3.2 (worse). `compare_fold_runs.py` now loads each run's own voice CSV from its checkpoint config.
- Age-consistency term B: raw version no gain; **bias-corrected version −0.8 mean EER (significant), −1.37 on standard, −0.23 on gender**. No additive gain with hyperbolic. `analysis/age_consistency_test.py` gained `--correct-voice-age` and `--weight-scale`; `analysis/estimate_face_age.py` now accepts 3-column trial lists. Started eval-set attribute extraction (`features/extract_eval_attributes.sh`).
- Started the speaker-embedding (ECAPA) and age-consistency experiments (§6.12). New extractor `extract_voice_features_speaker_age.py`, new test `analysis/age_consistency_test.py`, new run script `runs/speaker_embedding/run_all.sh`. Run scripts now check completion by 90-epoch logs.
- **Environment incident (fixed):** `pip install speechbrain` in the `video_features` env upgraded torch from 2.7.1+cu118 to 2.14.1 (CUDA 13). Restored torch/torchvision/torchaudio 2.7.1+cu118 and removed the CUDA-13 packages; that removal deleted a shared cuDNN file, so the CUDA-11 NVIDIA libraries were force-reinstalled. **Verified:** FaRL re-extraction is bit-identical to the stored features; WavLM matches to 5.6e-4 (the same fp16 rounding as before). Later installs pin torch with a constraints file. New packages in the env: speechbrain 1.0.3, torchaudio 2.7.1+cu118, requests, hyperpyyaml, ruamel.yaml, sentencepiece.
- Measured where gender-track errors come from (§6.7b): 47% of false accepts are similar-age (<5 yrs), 53% have ≥5-year gaps; errors split evenly between false accepts and false rejects.
- Decided how to use the age model: B (fixed age-consistency term) first, A (extra input) second, both gated on a quick correlation check (§9.1).
- Listed candidate frozen voice encoders (§9.1).
- Layer attention final (3 seeds): `mix_mean` no change (−0.1 mean); `mix_meanstd` +0.85; `attn_meanstd` +1.8 (significantly worse). Dropped.
- Layer-attention preliminary results (seeds 1–2): `mix_mean` ≈ V2; `mix_meanstd` worse by 0.9; `attn_meanstd` worse by 1.7. The learned weights mildly favour the lower WavLM layers.
- **Correction:** `checkpoint_last.pth.tar` is written after every epoch, so counting those files overstates finished runs. Completion is now checked by counting 90-epoch training logs. (All comparisons reported so far were made only after checking every run had 90 epochs.)
- Added the gender-track strategy (§9.1).
- Hyperbolic experiment finished (30 runs): standard −1.0 points (τ = 0.1) and −1.3 (τ = 0.2), significant; gender unchanged; mean −0.5 (not significant). Layer-attention runs started automatically.

### 2026-11-13
- Created this file.
- Downloaded and checked the evaluation data, found and worked around the `gender/English_valid.txt` path error, extracted evaluation features, and built and submitted `eval1_v2.zip`: **31.28%** (baseline 35.32%).
- Read the official rules FAQ and recorded the compliance policy (§8).
- Implemented hyperbolic geometry and WavLM layer pooling (§6.10–6.11, §7). Verified that V2 behaviour is unchanged, and started 30 hyperbolic runs with 45 layer runs queued.

### 2026-10-13 to 2026-10-16
- V2 full-data model, dev submission **29.11%**, `checkpoints/v2/` reference folder.
- Rules check on external data (inconclusive at the time) and review of the FAA and PAEFF papers.
- Age and build diagnostics (§6.6–6.7), the voice augmentation experiment (§6.8), and the new paired-comparison tool.

### Before 2026-10-13
- Local identity-disjoint five-fold validation.
- V1 (FaRL + WavLM x-vector), then V2 ablations.
- Face-backbone study, loss studies and the pairwise verifier (§6.1–6.5).

---

## 11. Key findings (interview summary)

- **Built a cross-modal face–voice verifier** that beats the official challenge baseline: 29.1% vs 36.9% EER on dev, 31.3% vs 35.3% on evaluation, and better in every cell. It uses only 70 training identities and a 0.4M-parameter model on frozen foundation-model features.
- **Language transfer solved:** with no Bengali training data, Bengali matches English performance. Multi-layer self-supervised speech features (WavLM) carry identity cues that transfer across languages.
- **Located the real bottleneck with diagnostics, not guesswork:** the voice predicts face age (r ≈ 0.6) and body build (r ≈ 0.3) within each gender, and the model already uses them. The remaining errors are same-gender, similar-age pairs.
- **A rigorous evaluation discipline:** identity-disjoint folds, multiple seeds, fixed-epoch training matching the final recipe, and a paired identity-level bootstrap. Local estimates predicted the leaderboard within 0.3 points, and several plausible ideas were ruled out with evidence (augmentation, learned verifier, six face backbones).
- **Careful engineering:** each new option is off by default with V2 behaviour verified bit for bit, the pipeline is reproducible (pinned model revisions, checksums, scripts), and an error in the official evaluation file was found and handled transparently.
