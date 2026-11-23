# Phase II — 01 Audio representation (Gate 1, 2026-11-19)

**Answer to RQ1:** the voice features B0 uses (WavLM-Base+-SV all-layer hidden-state mean ⊕ eGeMAPS) are a **weak speaker representation**. Across recording sessions they reach only 30–38% same-gender EER, and their similarity is dominated by the session (channel/video), not the speaker. A dedicated frozen speaker encoder that we already had, **ECAPA-TDNN (VoxCeleb2)**, is about 3× better and almost language-independent: 9.8% same-language / 11.0% cross-language EER across sessions. **Fine-tuning WavLM is not warranted yet.** The frozen ECAPA is already strong, so the open question moves to Gate 3: does better speaker identity help *cross-modal* verification?

Code: `FLAG_2027/phase2/{unimodal_eval,audio_diagnostics,audio_speaker_encoders,extract_external_speaker_embeddings}.py`. Results: `results/phase2/phase1_audio/` (`trials.npz`, `scores*.npz` raw per-trial scores, `metrics*.json`).

## Protocol

- **Pools, impostors drawn from the same group:**
  - **Official**: 70 people (English), in the 5 validation folds (14 each), so B0's projected voices are on unseen people.
  - **Bilingual**: MAV-Celeb v1 (English + Urdu, 64) and v3 (English + German, 49), in 10 groups. The v1 groups are exactly the P2 Urdu-holdout split.
- **Trials:** deterministic and built from metadata only (`build_trials`, seed = group).
  - Genuine pairs never share a recording: SHA-256 for official (6,485 rows hold only 4,029 unique recordings), path for external.
  - Genuine pairs are tagged same/different language and same/different session (source video).
  - Impostors per genuine anchor: same gender/same language, different gender, same gender/other language, and nearest median-F0 same-gender person.
  - The same trials are used for every representation.
- **Backends** are fitted on identity-disjoint people only:
  - official pool: all external people;
  - bilingual pool: official + VoxCeleb2 people.
  - Backends: raw cosine, centred cosine, z-score, LDA (shrinkage, ≤ 150 dims).
- **Encoder provenance:** see `00_research_plan.md`.
  - WavLM features: identical script/revision/chunking for official and external.
  - ECAPA for external rows: streamed from the source zips. The encoder check reproduces the cached official ECAPA (min cos 0.99999994), and the mapping check reproduces cached WavLM layers for 60 external clips (min cos 0.9999995).
  - **Finding:** the cached official x-vectors (`voice_v1_wavlm`) were batch-padded across files. Clips ≥ 10 s match a per-clip re-extraction exactly, but clips < 3 s reach only 0.97 median cos (min 0.76). Per-clip x-vectors for both pools are in `features/voice_v9_xvector_perclip/` and `combined_all/train_voices_xvector.csv`.

## 1. Verification (EER %, same-gender impostors unless stated)

**Bilingual pool** (sessions known; the challenge-relevant view):

| Representation (best backend) | Same video | Same language, other video | **Other language, other video** | Different gender | Similar F0 | R@1 (other videos) | Cross-language R@1 |
|---|---|---|---|---|---|---|---|
| WavLM all-layer mean, raw cosine | 23.9 | 42.0 | 43.2 | 36.8 | 42.5 | 0.56 | 0.38 |
| WavLM all-layer mean, LDA | 11.1 | 30.5 | 34.0 | 13.7 | 31.8 | 0.77 | 0.62 |
| WavLM ⊕ eGeMAPS (B0 input), LDA | 10.4 | 30.5 | 34.2 | 12.7 | 31.7 | 0.77 | 0.63 |
| eGeMAPS alone, LDA | 13.8 | 39.0 | 40.4 | 22.7 | 40.9 | 0.51 | 0.36 |
| WavLM-SV x-vector (per clip), LDA | 12.4 | 22.3 | 25.1 | 6.5 | 22.8 | 0.90 | 0.80 |
| **ECAPA-TDNN, centred** | **4.3** | **9.8** | **11.0** | **5.9** | **9.9** | **0.97** | **0.96** |
| ECAPA ⊕ WavLM, LDA | 4.6 | 14.4 | 16.6 | 5.5 | 15.2 | 0.96 | 0.93 |
| P2 projected voice (unseen people, Urdu unseen; v1 groups only) | 15.1 | 31.6 | 34.4 | 6.9 | 32.3 | 0.69 | 0.48 |

**Official pool** (English; no session information, so it may include same-video pairs):

| Representation | All impostors | Same gender | Different gender | Similar F0 | R@1 |
|---|---|---|---|---|---|
| WavLM all-layer mean, raw (as B0 consumes it before standardisation) | 27.7 | 30.1 | 25.2 | 30.8 | 0.94 |
| WavLM all-layer mean, LDA | 11.6 | 15.9 | 5.5 | 17.1 | 0.99 |
| WavLM ⊕ eGeMAPS, z-score / LDA | 15.3 / 11.3 | 17.7 / 15.2 | 12.5 / 5.3 | 19.6 / 16.8 | 0.98 / 0.99 |
| x-vector per clip, LDA | 10.5 | 14.5 | 4.5 | 15.1 | 0.97 |
| **ECAPA, LDA** | **4.2** | **5.1** | **2.8** | 6.2 | **1.00** |
| B0 projected voice | 13.2 | 18.7 | 3.6 | 19.3 | 0.97 |
| wav2vec2 age/gender hidden state | 20.5 | 29.6 | 3.7 | 31.3 | 0.87 |

**Readings:**
- **Session dominates WavLM.** Same-video vs other-video costs 20 points of EER for WavLM but 5.5 for ECAPA. The official-pool numbers look better than the bilingual ones mainly because official pairs can come from the same video.
- **Language costs little once the session is controlled.** Cross-language minus same-language: WavLM +3.5, x-vector +2.8, ECAPA +1.2.
- **Same-gender is the hard case for every representation**, and similar-F0 impostors are no harder than random same-gender ones. Pitch is not what remains confusable.
- B0's projected voice keeps gender almost perfectly (6.9% different-gender EER), but its within-gender speaker identity is only at the level of WavLM + LDA.

## 2. Retrieval and neighbour bias (bilingual, other-video gallery)

- **Language-biased neighbours:**
  - WavLM: 80–82% of nearest neighbours share the query's language (base rate 53%); x-vector 76–77%; ECAPA 81% (its neighbours are also mostly correct, R@1 0.97).
  - **Cross-language retrieval** (non-English query → English gallery): R@1 0.37 (WavLM raw), 0.62 (LDA), 0.80 (x-vector), **0.96 (ECAPA)**.
- **Linear identity probe** (trained on a speaker's English clips, tested on their other-language clips vs unseen English videos): WavLM LDA 0.56 vs 0.69; x-vector 0.74 vs 0.83; **ECAPA 0.93 vs 0.95**.

## 3. Representation diagnostics

| | Variance explained by identity / language / gender | Language probe, unseen people (balanced acc.) | Best ARI (oracle threshold) |
|---|---|---|---|
| WavLM centred | 0.23 / 0.05 / 0.04 | 0.83 | 0.24 |
| WavLM LDA | 0.31 / 0.04 / 0.10 | 0.82 | 0.47 |
| eGeMAPS LDA | 0.23 / 0.01 / 0.07 | 0.61 | 0.35 |
| x-vector centred | 0.69 / 0.03 / — | 0.68 | 0.60 |
| **ECAPA centred** | **0.51 / 0.03 / —** | 0.79 | **0.95** |
| P2 projected voice | 0.64 / 0.01 / 0.18 | 0.75 | — |

- **The GAMP-VF clustering failure was a representation failure, not a clustering failure.** With an oracle threshold, WavLM reaches only 0.24–0.48 ARI, while ECAPA reaches 0.95 with the same agglomerative method.
- Language is linearly decodable from every representation (0.68–0.83). For ECAPA this coexists with near-perfect cross-language identity, so **high language predictability does not imply missing speaker information**.
- **WavLM layer sweep** (centred, same-gender EER; official / bilingual same-language / bilingual cross-language):
  - L2–L4: 17.0–17.7 / 36.6–37.6 / 37.9–39.3;
  - L12: 28.2 / 38.6 / 41.9.
  - Lower layers carry more speaker identity and are less language-sensitive. The all-layer mean dilutes them.

## 4. Error analysis

- **Duration** (bilingual, same language, other video, same gender):
  - ECAPA 22.6 / 11.0 / 9.5 / 6.6% for < 3 / 3–6 / 6–10 / > 10 s;
  - x-vector 37.6 → 19.2%;
  - official B0 projected voice 21.5 → 13.8%.
  - Short clips are the main residual error for ECAPA. Official clips are short (median 5.2 s, p10 2.5 s; external median 8.0 s).
- Similar-F0 impostors are not harder than random same-gender impostors for any encoder.
- **Not available:** recording-quality metadata for external clips; Hindi (no data).

## 5. Gate-1 decision

- **WavLM fine-tuning: not now.** A frozen, already-available encoder (ECAPA) reaches near-ceiling unimodal speaker verification across languages.
- **Recommended audio input for Phase 3: ECAPA (centred)**, tested against the current WavLM ⊕ eGeMAPS in the 2 × 2 encoder substitution.
- Caveat: §6.12 of the progress log found ECAPA *concatenation* useless at 56 training people. That test used ECAPA only for the official rows; now it exists for all 301 people, so the retest is new information.
