#!/usr/bin/env bash
# External data: V2 (config E) + external identities on the training side of every fold.
# Same number of optimisation steps as V2 (40 batches x 90 epochs); validation unchanged.
# Usage: runs/external/run_all.sh <set-name> [configs...]   (set = features/external/combined_<set>)
set -euo pipefail
cd "$(dirname "$0")/../.."
SET=$1; shift
CONFIGS=${*:-"share50 natural"}
PY=${PY_TRAIN:-python}
FACE=features/face_v2_farl/train_English_faces.csv
VOICE=features/voice_v4_wavlm_allmean/train_English_voices.csv
EXT=features/external/combined_$SET
E_SETTINGS="--contrastive-mode supcon --identity-weight 0 \
  --face-hidden-dims 256 --voice-hidden-dims 256 --embedding-dim 128 --hidden-dropout 0.3 \
  --standardize-inputs --input-dropout 0.2 --input-noise-std 0.1 --max-epochs 90 --patience 90 \
  --batches-per-epoch 40 --extra-face-csv $EXT/train_faces.csv --extra-voice-csv $EXT/train_voices.csv \
  --extra-manifest $EXT/train_manifest.txt --extra-gender-metadata $EXT/gender.csv"
jobs() { for seed in 1 2 3; do for fold in 0 1 2 3 4; do for c in $CONFIGS; do echo "$c $fold $seed"; done; done; done; }
run_one() {
  local c=$1 fold=$2 seed=$3 extra=""
  case $c in share50) extra="--official-share 0.5";; share25) extra="--official-share 0.25";; natural) extra="";;
              wide512) extra="--official-share 0.5 --face-hidden-dims 512 --voice-hidden-dims 512";; esac
  local out=runs/external/${SET}_$c/fold_${fold}_seed_${seed}
  [[ -f $out/training_log.csv ]] && [[ $(($(wc -l < $out/training_log.csv) - 1)) -ge 90 ]] && return 0
  rm -rf "$out"
  $PY FLAG_2027/train_cross_modal.py --fold "$fold" --seed "$seed" --face-csv $FACE --voice-csv $VOICE \
    --save-dir "$out" $E_SETTINGS $extra > /dev/null
  echo "${SET}_$c fold $fold seed $seed done"
}
export -f run_one; export PY FACE VOICE E_SETTINGS SET
jobs | xargs -P 4 -L 1 bash -c 'run_one "$@"' _
echo ALL DONE
