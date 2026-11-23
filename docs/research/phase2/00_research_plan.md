# Phase II — 00 Research plan (2026-11-19)

**Goal:** find where identity information is lost in face → voice verification, and fix that component. Each gate decides whether the next experiment is justified.

## Gate 0: audit and storage (done 2026-11-19)

- **B0 = `full`**, reproducible bit-for-bit: a 2-epoch rerun on 18 Nov matched the 16 Nov log exactly (`docs/gamp_vf/01_repository_audit.md` §1). B0 checkpoints, terms and the eval zip are not touched.
- **Storage:** about 12 GB free after the user-approved deletion of the external audio work folders (8.4 GB test write OK). Fold runs are 5.8 MB each; Phase II diagnostics write only small arrays. The source zips stay in `external_data/*/raw` (md5 verified 18 Nov).
- **Languages actually available:** English (official + external), Urdu (MAV-Celeb v1), German (MAV-Celeb v3), unlabelled "Mixed" (VoxCeleb2). **No Hindi**, so every Hindi experiment is infeasible and reported as such.

**Voice representations already cached:**

| Name | Model / revision | Dim | Preprocessing | Coverage |
|---|---|---|---|---|
| WavLM all-layer mean | `microsoft/wavlm-base-plus-sv` @ `feb593a6` hidden states, layers 0–12 mean-pooled | 768 | 16 kHz, 10 s chunks / 5 s overlap, fp16 | official + all external (identical script and revision) |
| WavLM per-layer | same, per layer (mean ⊕ std) | 13 × 1536 | same | official + all external |
| WavLM-SV x-vector | same model's x-vector head | 512 | same | **official only** |
| ECAPA-TDNN | `yangwang825/ecapa-tdnn-vox2` @ `c605f571` (VoxCeleb2-trained) | 192 | 16 kHz whole utterance | **official only** |
| eGeMAPS v02 | openSMILE 2.5.1 functionals (median-imputed, clipped in the 856-d file) | 88 | — | all |
| wav2vec2 age/gender hidden state | `audeering/wav2vec2-large-robust-24-ft-age-gender` @ `bad2737f` | 1024 | — | **official only** |

- The official training audio has **4,029 unique recordings in 6,485 rows**. Genuine trials exclude identical recordings by SHA-256 hash.
- External clips are unique by path. They carry a video ID, which is used as the session.

## Plan by phase and gate

| Gate | Question | Experiments | Decision rule |
|---|---|---|---|
| 1 | RQ1: is the voice representation speaker-discriminative, also across languages? | Unimodal verification (same/cross-language, same/different session, same/different gender, similar-F0 impostors, duration bins), retrieval, probes, layer sweep, clustering re-check, for every cached representation + LDA backend + the B0/P2 projected voice | Fine-tune or extract a better encoder **only if** frozen features are far from a real speaker encoder *and* the gap matters cross-modally (Gate 3) |
| 2 | RQ2: is the face representation identity-discriminative? | Same protocol for FaRL, InsightFace (ArcFace), DINOv2 caches; duplicates by hash | Replace only if face identity is the weak side |
| 3 | RQ3: is unimodal identity transferable across modalities? | Encoder substitution 2 × 2 (best audio × best face), projection-capacity and loss ablations with everything else fixed | Pick the factor with a 3-seed paired gain |
| 4 | RQ4: do experts add information? | Expert score correlation and error overlap first; low-capacity fusion (weighted mean, logistic) on out-of-fold scores | Build fusion only if experts are complementary |
| 5 | RQ5: does it survive an unseen language? | Urdu-holdout proxy (P2: unseen identities **and** unseen language) for every candidate; German holdout only if powered | Reject gains that exist only for heard languages |

**Rules:** frozen encoders first; identity-disjoint fitting for every backend, probe and fusion; fixed deterministic trials shared by all compared systems; 3 seeds × 5 folds and a paired identity-level bootstrap for any claim; every result saved to `results/phase2/phase<k>_*/`.
