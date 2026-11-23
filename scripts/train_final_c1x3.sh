#!/usr/bin/env bash
# Final system C1 x 3: three full-data models (seeds 1-3) with the B0 recipe and the 1,048-d voice
# input [WavLM all-layer mean 768 | eGeMAPS 88 | ECAPA 192]. These are the exact settings used for
# checkpoints v5_ecapa_concat (seed 1) and v6_ecapa_concat_ens (seeds 2, 3).
# Requires the feature caches produced by scripts/extract_features.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY_TRAIN:-python}
EXT=features/external/combined_all
for seed in 1 2 3; do
  $PY FLAG_2027/train_cross_modal_full.py --seed $seed \
    --face-csv features/face_v2_farl/train_English_faces.csv \
    --voice-csv features/voice_v10_wavlm_egemaps_ecapa/train_English_voices.csv \
    --save-dir checkpoints/c1x3/seed_$seed/full_data --epochs 90 \
    --architecture-name cross_modal_identity_network_v5_ecapa_concat_full \
    --contrastive-mode supcon --identity-weight 0 --face-hidden-dims 256 --voice-hidden-dims 256 --embedding-dim 128 \
    --hidden-dropout 0.3 --standardize-inputs --input-dropout 0.2 --input-noise-std 0.1 --batches-per-epoch 50 --official-share 0.5 \
    --extra-face-csv $EXT/train_faces.csv --extra-voice-csv $EXT/train_voices_egemaps_ecapa.csv \
    --extra-manifest $EXT/train_manifest.txt --extra-gender-metadata $EXT/gender.csv \
    --trait-csv features/traits/train_English_traits.csv --extra-trait-csv features/traits/combined_all_traits.csv --trait-weight 0.2 \
    --language-adversary-weight 0.1 --cross-lingual-weight 0.1
done
