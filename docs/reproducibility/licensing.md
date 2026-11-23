# Licensing, attribution and redistribution

**This repository has no licence yet.** The author must choose one before the repository is made public. Options:

1. **MIT or Apache-2.0 for the code**, with this notice: the complete pipeline depends on non-commercial components (audeering age/gender CC BY-NC-SA 4.0; InsightFace pretrained models; openSMILE) and on datasets with their own access terms.
2. **A non-commercial licence** (e.g. CC BY-NC 4.0) for the whole repository, matching the most restrictive component.

| Component | Licence / terms (as checked on 2026-11-23) | In this repository |
|---|---|---|
| Our code (`FLAG_2027/` minus the organisers' files, `scripts/`, `tests/`, `site/`) | to be chosen (above) | included |
| Organisers' starter code (github.com/SwapnilKhandoker101/FLAG_2027: `main.py`, `retrieval_model.py`, `computeScore.py`, `online_evaluation.py`, `test.py`, `challenge/`, five `feature_extraction/` files) | **no licence published** | **excluded**; obtain from the organisers |
| ECAPA-TDNN `yangwang825/ecapa-tdnn-vox2` | Apache-2.0 (model card) | referenced only |
| SpeechBrain, VoxLingua107 LID | Apache-2.0 | referenced only |
| WavLM Base+ | MIT (microsoft/unilm repository; the HF card lists no licence) | referenced only |
| FaRL | MIT (FacePerceiver/FaRL repository); check the weight terms in its README | referenced only |
| audeering wav2vec2 age/gender | **CC BY-NC-SA 4.0** (non-commercial) | referenced only |
| InsightFace `buffalo_l` | InsightFace states its pretrained models are for non-commercial research use; **verify** | referenced only |
| openSMILE | audEERING licence (research / non-commercial; GitHub reports no SPDX id); **verify** | referenced only |
| MAV-Celeb, VoxCeleb2, FLAG 2027 data | dataset access terms; audio and video remain the property of their owners | **not included** |
| Trained checkpoints | derived from the data and models above | **not included**; reproduce with `scripts/train_final_c1x3.sh` |
| Result JSONs, per-trial numeric scores, figures | numbers only | included |

Biometric data (faces, voices) is never redistributed. `scripts/privacy_scan.py` (also run in CI) fails if media files, checkpoints, machine paths, e-mail addresses or tokens appear in the repository.
