#!/usr/bin/env bash
# Order of the frozen-feature extraction used by the project (see docs/reproducibility/reproduce.md).
# Each step calls a script in scripts/features or FLAG_2027/; environment variables select Python envs:
#   PY_TRAIN (torch 2.6, training + eGeMAPS), PY_AV (transformers/speechbrain/FaRL), PY_FACE (InsightFace).
set -euo pipefail
cd "$(dirname "$0")/.."
cat <<'STEPS'
1. Official train set (FaRL faces, WavLM layers -> all-layer mean CSV): see docs/reproducibility/reproduce.md §2
2. External sources: FLAG_2027/external_data/prepare_mavceleb.py, prepare_voxceleb2.py, then
   scripts/features/process_external_source.sh <name> <work-dir>   (language ID + FaRL + WavLM)
   FLAG_2027/external_data/build_external_training_set.py           (filtering, dedup, merge)
3. eGeMAPS: scripts/features/extract_egemaps_all.sh ; FLAG_2027/feature_extraction/build_wavlm_egemaps.py
4. ECAPA + per-clip x-vectors for external rows (streamed from the zips): FLAG_2027/phase2/extract_external_speaker_embeddings.py
   ECAPA for official/eval rows: FLAG_2027/feature_extraction/extract_voice_features_speaker_age.py --model ecapa
5. Age/gender estimates: extract_voice_features_speaker_age.py --model agegender ; FLAG_2027/analysis/estimate_face_age.py
   (eval: scripts/features/extract_eval_attributes.sh)
6. Trait labels: FLAG_2027/analysis/build_trait_labels.py
7. Eval features: scripts/features/extract_eval_v2.sh, then build_wavlm_egemaps.py (eval_v8) and append ECAPA (eval_v10)
STEPS
