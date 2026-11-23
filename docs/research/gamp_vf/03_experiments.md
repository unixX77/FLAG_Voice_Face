# GAMP-VF — 03 Experiment history (seeds, folds, checkpoints, data versions)

**Common to every run:**
- **Data:** official `train_set/train_set/train_English.txt` (6,485 rows, 70 people) and external `features/external/combined_all` (29,702 rows, 231 people), unchanged since 2026-11-16.
- **Features:** face `features/face_v2_farl` (FaRL 512) and external `train_faces.csv`; voice `features/voice_v8_wavlm_egemaps` (WavLM 768 ⊕ eGeMAPS 88) and external `train_voices_egemaps.csv`; traits `features/traits/*`.
- **Folds:** `validation/fold_{0..4}` with fixed trial files. 90 epochs, `checkpoint_last` only.
- **Runner:** `runs/gamp_vf/run_all.sh <config>` (B0 arguments + additions). Run directories are `runs/gamp_vf/<config>/fold_<k>_seed_<s>/`.
- **Code:** `FLAG_2027/gamp_vf/` and `FLAG_2027/train_cross_modal.py` as of 2026-11-18. Pre-change copies are in `results/gamp_vf/code_backups/`.

| # | Date | Experiment | Configs | Seeds × folds | Output |
|---|---|---|---|---|---|
| E0 | 11-18 | Regression check: new trainer, default options vs old trainer and the 16 Nov `full` run (2 epochs) | B0 | 1 × fold 0 | bit-identical (scratch only) |
| E1 | 11-18 | Smoke runs for every new option (3 epochs) | 7 options | 1 × fold 0 | scratch only |
| E2 | 11-18 | Stage A/B diagnostics on unseen people (held-out MAV-Celeb v1, Urdu unseen) | `runs/traits/p2_full` | seed 1 × 5 | `results/gamp_vf/stage_A_B/*.json` |
| E3 | 11-18 | Cross-lingual ablation: no language losses vs language losses vs full | `p2_base`, `p2_lang`, `p2_full` | seed 1 × 5 | `results/gamp_vf/stage_A_B/cross_lingual_*` |
| E4 | 11-18 | Screening B1–B3 | `b1_proto`, `b2_protobal`, `b3_sg`, `b3_nn`, `b3_sgn`, `b3_sgn100`, `b3_sga` | seed 1 × 5 | `results/gamp_vf/stage_C_screening/` |
| E5 | 11-18 | Promotion of the two gender-track trends to 3 seeds | `b3_sg`, `b3_sga` | seeds 1–3 × 5 | `results/gamp_vf/stage_C_final/` |
| E6 | 11-18 | B4 = B2 + same-gender sampling | `b4_sg` | seed 1 × 5 | `results/gamp_vf/stage_C_screening/` |
| E7 | 11-18 | Simulation teacher (no H, no C) and oracle (H labelled, no C) | `sim_teacher`, `sim_oracle` | seed 1 × 5 | `results/gamp_vf/stage_G/` |
| E8 | 11-18 | Pseudo-label mining it1 (Stages E + F; chosen mode on C) and noisy variant it1hung (Hungarian, all matches) | teacher = `sim_teacher` seed 1 | 5 folds | `results/gamp_vf/stage_E/{it1,it1hung}/fold_k/` |
| E9 | 11-18 | Oracle-group matching upper bound | teacher = `sim_teacher` seed 1 | 5 folds | `results/gamp_vf/stage_F/oracle_group_matching.json` |
| E10 | 11-18 | Students from it1 / it1hung | `sim_student_it1`, `sim_student_it1hung` | seed 1 × 5 | `results/gamp_vf/stage_G/` |
| E11 | 11-18 | Urdu-holdout proxy for same-gender sampling (vs existing `runs/traits/p2_full`) | `p2_b3_sg` | seeds 1–3 × 5 | `results/gamp_vf/cross_lingual/` |

**Disk:** 95 GAMP-VF runs = 522 MB in `runs/gamp_vf/` (checkpoint_last only); stage results = 2.3 MB. About 0.5 GB of the home quota was still free afterwards.

**Not run, with reasons:**
- **Iterations 2–3 and the EMA teacher (B7/B8):** the it1 diagnostics did not justify another iteration. Precision on H was below 0.5, and even oracle grouping gives only 57% precision at 10% coverage. The H labels themselves carry no measurable gain (E7).
- **Hindi analyses:** there is no Hindi data.
- **English → German held-out language:** only 49 German speakers and 1,565 rows; not run, since no candidate needed it.
- **Per-language EER on the official folds:** the official validation is English only. The Urdu-holdout proxy (P2) is the cross-lingual test.

**Hard-negative definitions used in E4:**
- `same_gender` = random same-gender partner.
- `nearest` / `same_gender_nearest` = one of the 5 most confusable identities by current-model identity prototypes, refreshed every 5 epochs from epoch 6, on training rows only.
- `same_gender_age` = one of the 5 same-gender identities with the closest median face-age estimate.
- Ratio 0.5 unless the name says `100`.
- Prototype loss weight 0.5 at temperature 0.07.
