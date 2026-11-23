# Phase III — 02 Controlled fusion ablation (E2, 2026-11-19)

**Answer to RQ2:** plain input concatenation (F1 = C1) remains the best practical choice.
- No fusion variant beats C1 on the unheard language.
- The two variants that looked better on English at seed 1 (64-d bottleneck, per-block L2) shrink to a non-significant −0.4 over 3 seeds, and they are significantly *worse* than C1 on the proxy's English cell.
- Late fusion with B0 (F5) **hurts** the unheard language against a same-size C1 ensemble.
- Gated fusion (F6) is not justified (E4 found no duration interaction).

Code:
- `VoiceBlockFusion` in `FLAG_2027/cross_modal_model.py` (modes `bottleneck`, `block_l2`, `residual`; off by default, and B0/C1 are bit-identical with it off);
- `--voice-fusion*` options in `train_cross_modal.py`; runner `runs/phase3/run_all.sh`.
- Results: `results/phase3/{screen_seed1,e2_3seeds,e6_stabilization}/`.

**Controls held fixed:** identity splits, training rows, the P × K sampler and its seeds, 90 epochs, FaRL faces, trait and language losses, and the fixed evaluation trials. Every variant differs from C1 only in how the 192 ECAPA dimensions enter the voice projector.

## Seed-1 screen (5 folds; Δ vs C1)

| ID | Variant | Voice-branch parameters | Official mean Δ [95% CI] | Urdu std Δ | Urdu gender Δ | Proxy English gender Δ |
|---|---|---|---|---|---|---|
| F1 | C1: concat 1,048 → 256 → 128 | 301,952 | — | — | — | — |
| F2 | bottleneck ECAPA → 32, concat | 267,168 | +0.59 [−0.7, +1.8] | +0.62 | +0.69 | −0.14 |
| F2 | bottleneck ECAPA → 64, concat | 281,536 | −1.04 [−2.1, +0.2] | +0.12 | −0.03 | −0.05 |
| F3 | per-block L2, equal block weight | 301,952 | −0.68 [−2.0, +0.8] | −0.17 | −0.41 | −0.20 |
| F4 | residual adapter (64), α₀ = 0 | 273,473 | +0.26 [−0.8, +1.4] | −0.19 | −0.21 | +0.27 |

Parameter counts (computed with the model code; B0's 856-d voice branch has 252,800) include the 256-d hidden LayerNorm and output layers. Inference cost differences are negligible: every voice branch has 0.25–0.30 M parameters, against the frozen encoders' tens of millions each (WavLM Base+, FaRL ViT-B/16, ECAPA-TDNN). Latency was not measured.

## 3 seeds × 5 folds for the two seed-1 leaders

| Variant | Official std / gender | Official mean Δ vs C1 [95% CI], P | Folds better (std / gen) | Urdu std Δ vs C1 [95% CI] | Urdu gender Δ vs C1 [95% CI] | Proxy English std Δ vs C1 |
|---|---|---|---|---|---|---|
| C1 | 20.48 / 31.94 | — | — | (23.94) | (36.32) | (22.96) |
| F2 k = 64 | 20.20 / 31.38 | −0.42 [−1.00, +0.29], 0.88 | 3/5, 4/5 | +0.24 [−0.12, +0.61] | +0.13 [−0.45, +0.70] | **+0.50** [+0.03, +1.00] |
| F3 block L2 | 20.16 / 31.39 | −0.43 [−1.15, +0.40], 0.81 | 3/5, 3/5 | +0.02 [−0.40, +0.47] | −0.10 [−0.79, +0.58] | **+0.41** [+0.03, +0.81] |

- The seed-1 English gains (−1.04, −0.68) were mostly seed noise; about −0.4 n.s. remains.
- Both variants trade a small, uncertain official-English gain for a significant loss on the proxy's English people and no unheard-language gain. **Neither replaces C1.**
- F3 was nevertheless carried into the German confirmation (`03`) as the strongest structural alternative, frozen before any German result was seen.

## Score-level fusion (F5; `results/phase3/e6_stabilization/`)

| System (3 seeds) | Official mean | Δ vs C1 | Urdu std | Urdu gender |
|---|---|---|---|---|
| C1 | 26.21 | — | 23.94 | 36.32 |
| C1 × 2 seeds | 25.70 | −0.51 [−0.62, −0.07] | 23.70 | 36.03 |
| C1 + B0 (F1 mean of z-scores) | 25.59 | −0.62 (vs C1 × 2: −0.11, n.s.) | 24.56 | 37.20 |
| C1 + B0 vs C1 × 2 on Urdu | — | — | **+0.81** [+0.24, +1.35] | **+1.15** [+0.32, +1.95] |

Mixing B0 back in **undoes part of the unheard-language gain**. On English it is no better than a second C1 seed.
