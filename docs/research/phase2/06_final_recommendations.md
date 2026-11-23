# Phase II — 06 Final recommendations (2026-11-19)

## Where identity information is lost

1. **Not in the encoders' identity content.**
   - Frozen ECAPA recognises speakers across sessions and languages (9.8% / 11.0% same-gender EER; cross-language R@1 0.96).
   - ArcFace recognises faces (1.8%).
   - Our inputs are far weaker unimodally: WavLM ⊕ eGeMAPS 30.5% across sessions, FaRL 12.3%.
2. **Identity encoders lose the face–voice link.** ArcFace costs +10 EER cross-modally; ECAPA alone costs +1.8. What transfers between voice and face are shared physical attributes, kept by attribute-rich encoders (FaRL, WavLM / eGeMAPS) and discarded by recognition training.
3. **Not in the projector or loss.** Linear, deeper, less regularised, multi-positive InfoNCE and margin-ranking variants were all equal or worse.
4. **The heard/unheard-language gap comes from the voice side's language sensitivity** (WavLM neighbours share the query's language 80% of the time). A language-invariant speaker embedding *added* to WavLM ⊕ eGeMAPS closes that gap.
5. **The residual is within-gender ambiguity.** Even the best model ranks the true face first among about 7 same-gender candidates only 43% of the time, and its errors are ranking errors, not calibration. These are finite experiments, not an information-theoretic bound.

## Consolidated comparison

Official five-fold: English, 3 seeds, no terms unless stated. Unseen-language proxy: P2 Urdu holdout, 3 seeds.

| System | Standard | Gender | Mean | Proxy std English / **Urdu** | Proxy gender English / **Urdu** | Status |
|---|---|---|---|---|---|---|
| **B0** `full` | 20.34 | 31.97 | 26.15 | 24.28 / 26.06 | 37.37 / 38.86 | baseline (eval #3 model family; eval #4 zip) |
| B0 + terms | 19.97 | 31.92 | 25.94 | — | — | basis of `eval4_traits_terms.zip` |
| Best audio-only representation in the cross-modal model: ECAPA + FaRL | 22.21 | 33.64 | 27.92 | 24.92 / 24.58 | 38.10 / 36.14 | worse at home, better unheard |
| Best face representation: FaRL (B0) | — | — | — | — | — | ArcFace +10, rejected |
| Best projection / loss: B0's | — | — | — | — | — | no change helped |
| **WavLM ⊕ eGeMAPS ⊕ ECAPA (single model)** | 20.48 | 31.94 | 26.21 | **22.96 / 23.94** | **36.18 / 36.32** | **recommended candidate** |
| … + refitted terms | 19.94 | 31.52 | **25.73** | — | — | `eval5_ecapa_concat_terms.zip` (not submitted) |
| Best simple fusion: B0 + ECAPA expert (F1) | 20.13 | 31.13 | 25.63 | 23.17 / 23.65 | 36.32 / 36.32 | a 2-model alternative |
| Multi-expert gated / MoE fusion | — | — | — | — | — | not justified (Gate 4) |

**What produced the gain:**
- A single input change, appending the frozen ECAPA embedding (192-d) to B0's voice features.
- Everything else (projector, loss, traits, language losses, training budget, folds, trials) is identical to B0.
- On the proxy it beats B0 on all four cells (P ≥ 0.96), most on the unheard language (−2.1 standard, −2.5 gender).

## Recommendation

1. **Candidate eval #5 = `submissions/eval5_ecapa_concat_terms.zip`** (md5 `77b9ca00…`).
   - Model `checkpoints/v5_ecapa_concat/full_data` (md5 `95524bb8…`), with terms from its fold models.
   - Validated archive, same trial order as eval #3/#4.
   - **Expected effect:** about 0 to −0.2 on the English cells (official folds) and about −2 on the Bangla cells if Urdu transfers. It is the first change whose gain is concentrated where we are weakest.
   - **Risks:** a single proxy language; and on the official English folds the gain is not significant (+0.05 without terms, −0.21 with). Treat it as a test of the unheard-language hypothesis, not as a confirmed improvement.
   - Submitting is your decision. Rules: allowed (frozen public encoder, every trial scored independently).
2. **Do not pursue** ArcFace, voice-identity replacement, projector/loss tuning, or gated/MoE fusion.
3. **Next research, if eval #5 confirms the Bangla gain:**
   - a 3-seed ensemble of the concat model (cheap, −0.3 to −0.4 at home);
   - other language-invariant voice descriptors next to WavLM (the x-vector, lower WavLM layers L2–L5);
   - larger paired pretraining (VoxCeleb1/2 dev) for more identities (§9.3), since identity count was the only other lever that ever moved the gender track.
4. **If eval #5 does not help Bangla:** the Urdu proxy does not represent Bangla. Rebuild the proxy with a typologically closer held-out language, e.g. Hindi (MAV-Celeb v2, if it can be downloaded).

## Deliverables and reproducibility

- **Docs:** `docs/phase2/00`–`06`.
- **Results** (raw per-trial scores where feasible): `results/phase2/phase{1_audio,2_face,3_alignment,4_fusion,5_language}/`.
- **Code:** `FLAG_2027/phase2/`, plus the face-input fix in `local_validation/compare_fold_runs.py` (backup `results/gamp_vf/code_backups/compare_fold_runs.py.pre_phase2`).
- **Runs:** `runs/phase2/<config>/fold_k_seed_s` (`run_all.sh`).
- **New feature caches** (all validated against existing caches):
  - `combined_all/train_voices_{ecapa,xvector,egemaps_ecapa}.csv`, `train_faces_arcface.csv`, `train_voice_durations.npy`;
  - `voice_v9_xvector_perclip`, `voice_v10_wavlm_egemaps_ecapa`;
  - `eval_ecapa`, `eval_v10`.
- **Tests:** 14 new (`phase2.test_phase2`), and all 72 project tests pass.
- **Not done:** WavLM / ECAPA fine-tuning (not warranted by Gate 1), residual / compatibility heads (not justified by 3.3), noise / reverberation robustness, Hindi and German holdouts.
