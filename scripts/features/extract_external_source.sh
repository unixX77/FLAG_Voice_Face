#!/usr/bin/env bash
# V2 features for one external source, with exactly the settings of the official
# training features: FaRL ViT-B/16 (LAIONFace20M ep16, CLIP preprocessing) faces;
# WavLM Base+ (pinned revision) all-13-layer mean voices (10 s chunks, 5 s overlap).
# Usage: scripts/features/extract_external_source.sh <name> <work-dir>   e.g. mav3 external_data/mavceleb/work/v3
set -euo pipefail
cd "$(dirname "$0")/../.."
name=$1; root=$2
REV=feb593a6c23c1cc3d9510425c29b0a14d2b07b1e
manifest=features/external/manifests/${name}_train.txt
out=features/external/$name
mkdir -p $out
# Each step is skipped when its output already exists (restart after an interruption).
[[ -s $out/train_faces_report.json ]] || ${PY_AV:-python} FLAG_2027/feature_extraction/extract_face_features_farl.py \
  --manifest $manifest --image-root $root \
  --checkpoint models/farl/FaRL-Base-Patch16-LAIONFace20M-ep16.pth \
  --output-csv $out/train_faces.csv > $out/faces.log 2>&1
[[ -s $out/train_layers.npy ]] || HF_HUB_OFFLINE=1 ${PY_AV:-python} FLAG_2027/feature_extraction/extract_voice_features_wavlm_layers.py \
  --manifest $manifest --audio-root $root \
  --output-npy $out/train_layers.npy --revision $REV > $out/voices.log 2>&1
${PY_TRAIN:-python} FLAG_2027/feature_extraction/build_voice_feature_csv.py \
  --manifest $manifest --layers-npy $out/train_layers.npy --layers 0-12 --pooling mean \
  --output-csv $out/train_voices.csv > /dev/null
echo "$name features done"
