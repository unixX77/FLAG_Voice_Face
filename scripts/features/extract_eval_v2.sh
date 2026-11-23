#!/usr/bin/env bash
# Evaluation-phase features for V2, identical settings to the dev and training features:
# FaRL ViT-B/16 (LAIONFace20M ep16) faces; WavLM Base+ (pinned revision) all-layer mean voices.
# Trial lists come from features/eval_manifests (official copies, one path-format fix; see its README).
set -euo pipefail
cd "$(dirname "$0")/../.."
REV=feb593a6c23c1cc3d9510425c29b0a14d2b07b1e
for track in gender no_gender; do
  for lang in English Bangla; do
    manifest=features/eval_manifests/$track/${lang}_valid.txt
    root=evaluation_phase/$track
    ${PY_AV:-python} FLAG_2027/feature_extraction/extract_face_features_farl.py \
      --manifest $manifest --image-root $root \
      --checkpoint models/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth \
      --output-csv features/eval_v2/$track/${lang}_valid_faces.csv > /dev/null 2>&1
    HF_HUB_OFFLINE=1 ${PY_AV:-python} FLAG_2027/feature_extraction/extract_voice_features_wavlm_layers.py \
      --manifest $manifest --audio-root $root \
      --output-npy features/eval_voice_v2_wavlm_layers/$track/${lang}_valid_layers.npy --revision $REV > /dev/null 2>&1
    ${PY_TRAIN:-python} FLAG_2027/feature_extraction/build_voice_feature_csv.py \
      --manifest $manifest \
      --layers-npy features/eval_voice_v2_wavlm_layers/$track/${lang}_valid_layers.npy --layers 0-12 --pooling mean \
      --output-csv features/eval_v2/$track/${lang}_valid_voices.csv > /dev/null
    echo "$track $lang done"
  done
done
