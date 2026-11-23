# Phase II — 02 Face representation (Gate 2, 2026-11-19)

**Answer to RQ2:**
- **InsightFace ArcFace** (`buffalo_l`, w600k R50) is a near-perfect face-identity encoder on unseen people: 1.8–2.2% same-gender EER, 2.0% against similar-age impostors.
- **FaRL** (B0's face input) is much weaker on identity: 12.3% same-gender, and 18.2% against similar-age impostors. Its similarity follows age and appearance strongly.
- **Decision:** test ArcFace in the Gate-3 encoder substitution. Keep FaRL as the anchor: a face-*recognition* embedding may discard exactly the appearance traits (age, build, morphology) that voices can predict, so the best unimodal encoder need not be the best cross-modal one.

Code: `FLAG_2027/phase2/face_diagnostics.py` (same trial machinery as Phase 1). Results: `results/phase2/phase2_face/`.

## Protocol

- Same deterministic trial builder as Phase 1. Genuine pairs never share an image (SHA-256; 6,485 official rows = 6,041 unique images).
- "Similar" impostors are the same-gender person with the closest median InsightFace age estimate.
- **Official pool:** 5 folds × 14 unseen people. All official encoders were cached.
- **External pool:** 231 people, FaRL only. The session is the source video of the frame, so same-video frames (near-duplicates) are reported separately.

## Results (EER %)

| Representation | All | Same gender | Different gender | Similar age | R@1 |
|---|---|---|---|---|---|
| **ArcFace (InsightFace buffalo_l), centred** | 2.8 | **1.8** | 2.0 | **2.0** | 1.00 |
| ArcFace, raw | 2.4 | 2.2 | 2.4 | 2.2 | 1.00 |
| FaRL, raw | 12.0 | 15.4 | 7.8 | 19.6 | 1.00 |
| FaRL, centred | 8.9 | 12.3 | 4.1 | 18.2 | 1.00 |
| FaRL + flip, centred | 8.8 | 12.0 | 4.1 | 18.1 | 1.00 |
| DINOv2 L12 patch, centred | 14.3 | 20.3 | 3.6 | 27.9 | 1.00 |
| DINOv2 L1–12 CLS, centred | 21.1 | 26.8 | 13.6 | 31.5 | 1.00 |
| FaRL + flip ⊕ DINOv2, centred | 19.7 | 25.5 | 12.6 | 30.6 | 1.00 |
| **B0 projected face** (fold models, unseen people) | 10.3 | 14.5 | 2.1 | 20.7 | 1.00 |

- **External faces (FaRL, centred):** same video 1.3%, **other video 12.1%**, similar age 11.9%. R@1 is 0.95 across videos and 0.995 when same-video frames are allowed.
- **Clustering ARI** by threshold (0.2 / 0.3 / 0.4 / 0.5): 0.16 / 0.29 / 0.43 / 0.60. The GAMP-VF threshold (0.35, tuned on 39 people) under-merged.
- Retrieval R@1 is about 1.0 for every encoder in 14-person galleries, so it does not discriminate here; EER does.

## Quality diagnostics (official, same-gender trials)

| Bin | ArcFace EER | FaRL EER | Genuine trials |
|---|---|---|---|
| detection score 0.50–0.65 | 2.0 | 13.0 | 1,877 |
| detection score 0.65–0.75 | 1.2 | 10.6 | 533 |
| detection score > 0.75 | 0.3 | 11.8 | 307 |
| face box 100–130 px | 5.5 | 11.7 | 408 |
| face box 130–160 px | 1.4 | 12.6 | 2,293 |

- ArcFace degrades with low detection confidence and small faces. FaRL is flat, consistent with it encoding appearance rather than fine identity detail.
- Images are 224 × 224 crops. Head pose, occlusion and alignment were never stored; pose would need landmark re-extraction, which is not done.

## Implications for Gate 3

- B0's projected face (14.5% same-gender) keeps less identity than FaRL itself (12.3%). The cross-modal objective discards some visual identity, as expected, since only voice-predictable information can be aligned.
- In B0's input space, *both* modalities are mid-quality identity encoders (voice about 30% across sessions, face about 12%). Gate 3 must separate whether cross-modal errors come from weak unimodal identity or from the face–voice link itself. That requires the 2 × 2 substitution:
  - {WavLM ⊕ eGeMAPS, ECAPA} × {FaRL, ArcFace};
  - ArcFace for the external faces has to be extracted first (cached for official rows only).
