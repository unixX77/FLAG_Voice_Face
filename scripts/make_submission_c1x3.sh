#!/usr/bin/env bash
# 1) Fit the fixed age/gender terms on the 3-seed C1 fold models (raw-mean ensemble), training data only.
# 2) Score the evaluation trials with the three full-data models: mean of (1 - cosine) + terms, per trial.
# Fold models come from: SEEDS="1 2 3" scripts/runs/phase2_run_all.sh g3_cat_farl
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY_TRAIN:-python}
$PY FLAG_2027/analysis/finalize_attribute_terms.py --ensemble --seeds 1 2 3 \
  --face-csv features/face_v2_farl/train_English_faces.csv --voice-csv features/voice_v4_wavlm_allmean/train_English_voices.csv \
  --voice-attributes features/voice_v7_agegender/train_English_attributes.csv --face-attributes features/face_age/train_English_face_age.csv \
  --checkpoints 'runs/phase2/g3_cat_farl/fold_{fold}_seed_{seed}/checkpoint_last.pth.tar' \
  --output-json checkpoints/c1x3/attribute_terms.json
$PY -m FLAG_2027.generate_cross_modal_submission \
  --checkpoint checkpoints/c1x3/seed_{1,2,3}/full_data/checkpoint_full.pth.tar \
  --dev-root features/eval_manifests --split valid --feature-root features/eval_v10 \
  --attribute-terms checkpoints/c1x3/attribute_terms.json \
  --voice-attribute-root features/eval_agegender --face-attribute-root features/eval_face_age \
  --output-root submissions/c1x3 --zip-path submissions/c1x3.zip
