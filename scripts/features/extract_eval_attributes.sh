#!/usr/bin/env bash
# Evaluation-phase voice age/gender (audeering, frozen) and face age/gender (InsightFace buffalo_l, frozen),
# same scripts and settings as for the training set. Each clip and image is processed independently.
set -euo pipefail
cd "$(dirname "$0")/../.."
for track in gender no_gender; do
  for lang in English Bangla; do
    manifest=features/eval_manifests/$track/${lang}_valid.txt
    (cd FLAG_2027/feature_extraction && ${PY_AV:-python} extract_voice_features_speaker_age.py \
      --model agegender --manifest ../../$manifest --audio-root ../../evaluation_phase/$track \
      --output-prefix ../../features/eval_agegender/$track/${lang}_valid --cache-dir ../../models/hf_cache > /dev/null 2>&1)
    ${PY_FACE:-python} FLAG_2027/analysis/estimate_face_age.py \
      --manifest $manifest --image-root evaluation_phase/$track \
      --output-csv features/eval_face_age/$track/${lang}_valid_face_age.csv > /dev/null 2>&1
    echo "$track $lang done"
  done
done
