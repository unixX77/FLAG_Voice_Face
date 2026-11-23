#!/usr/bin/env bash
# Voice -> traits -> face architecture on top of eval #3 (V2 + all external data, share50).
# Every config = all_share50 settings + the listed additions; 3 seeds x 5 folds x 90 epochs.
# Usage: runs/traits/run_all.sh <config> [<config> ...]
#   P1 (official five-fold, decides submissions): lang, traits, egemaps, full, traits05
#   P2 (unheard-language proxy: 1/5 of MAV-Celeb v1 people held out per fold, no Urdu in training):
#      p2_base, p2_lang, p2_full
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY_TRAIN:-python}
EXT=features/external/combined_all
BASE="--face-csv features/face_v2_farl/train_English_faces.csv --contrastive-mode supcon --identity-weight 0 \
  --face-hidden-dims 256 --voice-hidden-dims 256 --embedding-dim 128 --hidden-dropout 0.3 \
  --standardize-inputs --input-dropout 0.2 --input-noise-std 0.1 --max-epochs 90 --patience 90 \
  --batches-per-epoch 40 --official-share 0.5 --no-best-checkpoint --extra-face-csv $EXT/train_faces.csv \
  --extra-manifest $EXT/train_manifest.txt --extra-gender-metadata $EXT/gender.csv"
WAVLM="--voice-csv features/voice_v4_wavlm_allmean/train_English_voices.csv --extra-voice-csv $EXT/train_voices.csv"
EGEMAPS="--voice-csv features/voice_v8_wavlm_egemaps/train_English_voices.csv --extra-voice-csv $EXT/train_voices_egemaps.csv"
TRAITS="--trait-csv features/traits/train_English_traits.csv --extra-trait-csv features/traits/combined_all_traits.csv --trait-weight 0.2"
LANG="--language-adversary-weight 0.1 --cross-lingual-weight 0.1"
P2="--holdout-external-prefix mav1 --exclude-train-languages Urdu"
args_for() {
  case $1 in
    lang) echo "$WAVLM $LANG";;
    traits) echo "$WAVLM $TRAITS";;
    traits05) echo "$WAVLM ${TRAITS/--trait-weight 0.2/--trait-weight 0.5}";;
    egemaps) echo "$EGEMAPS";;
    full) echo "$EGEMAPS $TRAITS $LANG";;
    p2_base) echo "$WAVLM $P2";;
    p2_lang) echo "$WAVLM $LANG $P2";;
    p2_full) echo "$EGEMAPS $TRAITS $LANG $P2";;
    *) echo "unknown config $1" >&2; exit 1;;
  esac
}
run_one() {
  local c=$1 fold=$2 seed=$3
  local out=runs/traits/$c/fold_${fold}_seed_${seed}
  [[ -f $out/training_log.csv ]] && [[ $(($(wc -l < $out/training_log.csv) - 1)) -ge 90 ]] && return 0
  rm -rf "$out"
  OMP_NUM_THREADS=2 $PY FLAG_2027/train_cross_modal.py --fold "$fold" --seed "$seed" --save-dir "$out" \
    $BASE $(args_for "$c") > /dev/null
  echo "$c fold $fold seed $seed done"
}
export -f run_one args_for; export PY BASE WAVLM EGEMAPS TRAITS LANG P2 EXT
for c in "$@"; do args_for "$c" > /dev/null; done
for seed in 1 2 3; do for fold in 0 1 2 3 4; do for c in "$@"; do echo "$c $fold $seed"; done; done; done \
  | xargs -P ${PARALLEL:-4} -L 1 bash -c 'run_one "$@"' _
echo "ALL DONE: $*"
