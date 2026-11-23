#!/usr/bin/env bash
# Phase III runs. Name = <protocol>_<variant>:
#   protocol: o (official five-fold) | p2 (Urdu holdout: mav1 people, no Urdu) | p3 (German holdout: mav3 people, no German)
#   variant : b0, c1 (= runs/phase2/g3_cat_farl recipe), rand (control D), shuf (control E),
#             k32/k64/k128 (bottleneck), bl2 (block L2), res (residual adapter), tr0/tr01 (trait weight 0 / 0.1)
# Everything else = B0 recipe (runs/traits/run_all.sh). DO NOT EDIT WHILE RUNNING.
# Usage: SEEDS="1" PARALLEL=7 runs/phase3/run_all.sh o_rand p2_rand ...
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY_TRAIN:-python}
EXT=features/external/combined_all
COMMON="--contrastive-mode supcon --identity-weight 0 \
  --face-hidden-dims 256 --voice-hidden-dims 256 --embedding-dim 128 --hidden-dropout 0.3 \
  --standardize-inputs --input-dropout 0.2 --input-noise-std 0.1 --max-epochs 90 --patience 90 \
  --batches-per-epoch 40 --official-share 0.5 --no-best-checkpoint \
  --extra-manifest $EXT/train_manifest.txt --extra-gender-metadata $EXT/gender.csv \
  --trait-csv features/traits/train_English_traits.csv --extra-trait-csv features/traits/combined_all_traits.csv \
  --trait-weight 0.2 --language-adversary-weight 0.1 --cross-lingual-weight 0.1 \
  --face-csv features/face_v2_farl/train_English_faces.csv --extra-face-csv $EXT/train_faces.csv"
V_B0="--voice-csv features/voice_v8_wavlm_egemaps/train_English_voices.csv --extra-voice-csv $EXT/train_voices_egemaps.csv"
V_CAT="--voice-csv features/voice_v10_wavlm_egemaps_ecapa/train_English_voices.csv --extra-voice-csv $EXT/train_voices_egemaps_ecapa.csv"
V_RAND="--voice-csv features/voice_v11_wavlm_egemaps_random192/train_English_voices.csv --extra-voice-csv $EXT/train_voices_egemaps_random192.csv"
FUSE="--voice-fusion-split 856"
variant() {
  case $1 in
    b0) echo "$V_B0";;
    c1) echo "$V_CAT";;
    rand) echo "$V_RAND";;
    shuf) echo "$V_CAT --shuffle-voice-block-start 856";;
    k32) echo "$V_CAT --voice-fusion bottleneck $FUSE --voice-fusion-dim 32";;
    k64) echo "$V_CAT --voice-fusion bottleneck $FUSE --voice-fusion-dim 64";;
    k128) echo "$V_CAT --voice-fusion bottleneck $FUSE --voice-fusion-dim 128";;
    bl2) echo "$V_CAT --voice-fusion block_l2 $FUSE";;
    res) echo "$V_CAT --voice-fusion residual $FUSE --voice-fusion-dim 64";;
    tr0) echo "$V_CAT --trait-weight 0";;
    tr01) echo "$V_CAT --trait-weight 0.1";;
    *) echo "unknown variant $1" >&2; exit 1;;
  esac
}
args_for() {
  local proto=${1%%_*} v=${1#*_}
  case $proto in
    o) variant "$v";;
    p2) echo "$(variant "$v") --holdout-external-prefix mav1 --exclude-train-languages Urdu";;
    p3) echo "$(variant "$v") --holdout-external-prefix mav3 --exclude-train-languages German";;
    *) echo "unknown protocol in $1" >&2; exit 1;;
  esac
}
run_one() {
  local c=$1 fold=$2 seed=$3
  local out=runs/phase3/$c/fold_${fold}_seed_${seed}
  [[ -f $out/training_log.csv ]] && [[ $(($(wc -l < $out/training_log.csv) - 1)) -ge 90 ]] && return 0
  rm -rf "$out"; mkdir -p "$(dirname "$out")"
  OMP_NUM_THREADS=2 $PY FLAG_2027/train_cross_modal.py --fold "$fold" --seed "$seed" --save-dir "$out" \
    $COMMON $(args_for "$c") > "$out.log" 2>&1 || { echo "FAILED $c fold $fold seed $seed (see $out.log)"; return 0; }
  rm -f "$out.log"; echo "$c fold $fold seed $seed done"
}
export -f run_one args_for variant; export PY COMMON V_B0 V_CAT V_RAND FUSE EXT
for c in "$@"; do args_for "$c" > /dev/null; done
for seed in ${SEEDS:-1 2 3}; do for fold in 0 1 2 3 4; do for c in "$@"; do echo "$c $fold $seed"; done; done; done \
  | xargs -P ${PARALLEL:-7} -L 1 bash -c 'run_one "$@"' _
echo "ALL DONE: $*"
