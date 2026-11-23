#!/usr/bin/env bash
# Phase II Gate 3: controlled encoder substitution. Every config = B0 (`full`, runs/traits/run_all.sh)
# with ONLY the voice and/or face input replaced. 90 epochs, checkpoint_last only.
# Usage: SEEDS="1 2 3" PARALLEL=7 runs/phase2/run_all.sh <config> ...
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
  --trait-weight 0.2 --language-adversary-weight 0.1 --cross-lingual-weight 0.1"
V_B0="--voice-csv features/voice_v8_wavlm_egemaps/train_English_voices.csv --extra-voice-csv $EXT/train_voices_egemaps.csv"
V_ECAPA="--voice-csv features/voice_v6_ecapa/train_English_voices.csv --extra-voice-csv $EXT/train_voices_ecapa.csv"
V_CAT="--voice-csv features/voice_v10_wavlm_egemaps_ecapa/train_English_voices.csv --extra-voice-csv $EXT/train_voices_egemaps_ecapa.csv"
F_FARL="--face-csv features/face_v2_farl/train_English_faces.csv --extra-face-csv $EXT/train_faces.csv"
F_ARC="--face-csv features/face_v1_insightface/train_English_faces.csv --extra-face-csv $EXT/train_faces_arcface.csv"
P2="--holdout-external-prefix mav1 --exclude-train-languages Urdu"
args_for() {
  case $1 in
    g3_b0_farl) echo "$V_B0 $F_FARL";;           # = B0 (control; runs/traits/full already exists)
    g3_ecapa_farl) echo "$V_ECAPA $F_FARL";;
    g3_b0_arcface) echo "$V_B0 $F_ARC";;
    g3_ecapa_arcface) echo "$V_ECAPA $F_ARC";;
    g3_cat_farl) echo "$V_CAT $F_FARL";;          # B0 voice + ECAPA (complementarity)
    p2_g3_cat_farl) echo "$V_CAT $F_FARL $P2";;
    # 3.3 projection capacity (B0 inputs; the later --face/--voice-hidden-dims override COMMON's)
    pj_linear) echo "$V_B0 $F_FARL --face-hidden-dims none --voice-hidden-dims none";;
    pj_deep) echo "$V_B0 $F_FARL --face-hidden-dims 512,256 --voice-hidden-dims 512,256";;
    pj_drop01) echo "$V_B0 $F_FARL --hidden-dropout 0.1";;
    # 3.4 losses (B0 inputs)
    ls_multipos) echo "$V_B0 $F_FARL --contrastive-mode multi_positive";;
    ls_margin) echo "$V_B0 $F_FARL --hard-negative-weight 0.25 --hard-negative-margin 0.1 --hard-negative-top-k 5";;
    p2_g3_ecapa_farl) echo "$V_ECAPA $F_FARL $P2";;
    p2_g3_b0_arcface) echo "$V_B0 $F_ARC $P2";;
    p2_g3_ecapa_arcface) echo "$V_ECAPA $F_ARC $P2";;
    *) echo "unknown config $1" >&2; exit 1;;
  esac
}
run_one() {
  local c=$1 fold=$2 seed=$3
  local out=runs/phase2/$c/fold_${fold}_seed_${seed}
  [[ -f $out/training_log.csv ]] && [[ $(($(wc -l < $out/training_log.csv) - 1)) -ge 90 ]] && return 0
  rm -rf "$out"; mkdir -p "$(dirname "$out")"
  OMP_NUM_THREADS=2 $PY FLAG_2027/train_cross_modal.py --fold "$fold" --seed "$seed" --save-dir "$out" \
    $COMMON $(args_for "$c") > "$out.log" 2>&1 || { echo "FAILED $c fold $fold seed $seed (see $out.log)"; return 0; }
  rm -f "$out.log"; echo "$c fold $fold seed $seed done"
}
export -f run_one args_for; export PY COMMON V_B0 V_CAT V_ECAPA F_FARL F_ARC P2 EXT
for c in "$@"; do args_for "$c" > /dev/null; done
for seed in ${SEEDS:-1 2 3}; do for fold in 0 1 2 3 4; do for c in "$@"; do echo "$c $fold $seed"; done; done; done \
  | xargs -P ${PARALLEL:-7} -L 1 bash -c 'run_one "$@"' _
echo "ALL DONE: $*"
