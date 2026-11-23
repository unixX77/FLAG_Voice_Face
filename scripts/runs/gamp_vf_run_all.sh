#!/usr/bin/env bash
# GAMP-VF ablations. Every config = B0 (`full`: runs/traits/run_all.sh) + the listed additions;
# official five-fold protocol, fixed 90 epochs, checkpoint_last only.
# Usage: SEEDS="1" PARALLEL=6 runs/gamp_vf/run_all.sh <config> [<config> ...]
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY_TRAIN:-python}
EXT=features/external/combined_all
E=results/gamp_vf/stage_E
B0="--face-csv features/face_v2_farl/train_English_faces.csv --contrastive-mode supcon --identity-weight 0 \
  --face-hidden-dims 256 --voice-hidden-dims 256 --embedding-dim 128 --hidden-dropout 0.3 \
  --standardize-inputs --input-dropout 0.2 --input-noise-std 0.1 --max-epochs 90 --patience 90 \
  --batches-per-epoch 40 --official-share 0.5 --no-best-checkpoint --extra-face-csv $EXT/train_faces.csv \
  --extra-manifest $EXT/train_manifest.txt --extra-gender-metadata $EXT/gender.csv \
  --voice-csv features/voice_v8_wavlm_egemaps/train_English_voices.csv --extra-voice-csv $EXT/train_voices_egemaps.csv \
  --trait-csv features/traits/train_English_traits.csv --extra-trait-csv features/traits/combined_all_traits.csv \
  --trait-weight 0.2 --language-adversary-weight 0.1 --cross-lingual-weight 0.1"
PROTO="--prototype-contrastive-weight 0.5"
BAL="--prototype-language-balanced --language-balanced-k"
args_for() {
  case $1 in
    b1_proto) echo "$PROTO";;                                     # B1 speaker/face prototypes
    b2_protobal) echo "$PROTO $BAL";;                             # B2 language-balanced prototypes
    b3_sg) echo "--batch-strategy same_gender --hard-ratio 0.5";; # B3 variants (strategy x ratio)
    b3_nn) echo "--batch-strategy nearest --hard-ratio 0.5";;
    b3_sgn) echo "--batch-strategy same_gender_nearest --hard-ratio 0.5";;
    b3_sgn100) echo "--batch-strategy same_gender_nearest --hard-ratio 1.0";;
    b3_sga) echo "--batch-strategy same_gender_age --hard-ratio 0.5";;
    p2_b3_sg) echo "--holdout-external-prefix mav1 --exclude-train-languages Urdu --batch-strategy same_gender --hard-ratio 0.5";;  # Urdu-holdout proxy
    b4_*) echo "$PROTO $BAL --batch-strategy ${B4_STRATEGY:?set B4_STRATEGY} --hard-ratio ${B4_RATIO:-0.5}";;
    # Hidden-association simulation (Stages E-G); identity lists in results/gamp_vf/stage_E
    sim_teacher) echo "--exclude-external-identities $E/hidden_plus_calibration.txt";;   # lower bound / teacher
    sim_oracle) echo "--exclude-external-identities $E/calibration_identities.txt";;     # upper bound: H with true labels
    sim_student_it*) echo "--exclude-external-identities $E/hidden_plus_calibration.txt \
      --pseudo-pairs-csv $E/${1#sim_student_}/fold_{fold}/pseudo_pairs.csv";;
    *) echo "unknown config $1" >&2; exit 1;;
  esac
}
run_one() {
  local c=$1 fold=$2 seed=$3
  local out=runs/gamp_vf/$c/fold_${fold}_seed_${seed}
  [[ -f $out/training_log.csv ]] && [[ $(($(wc -l < $out/training_log.csv) - 1)) -ge 90 ]] && return 0
  rm -rf "$out"
  local extra; extra=$(args_for "$c"); extra=${extra//\{fold\}/$fold}
  OMP_NUM_THREADS=2 $PY FLAG_2027/train_cross_modal.py --fold "$fold" --seed "$seed" --save-dir "$out" \
    $B0 $extra > "$out.log" 2>&1 || { echo "FAILED $c fold $fold seed $seed (see $out.log)"; return 0; }
  rm -f "$out.log"
  echo "$c fold $fold seed $seed done"
}
export -f run_one args_for; export PY B0 PROTO BAL EXT E B4_STRATEGY B4_RATIO
for c in "$@"; do args_for "$c" > /dev/null; done
mkdir -p runs/gamp_vf
for seed in ${SEEDS:-1 2 3}; do for fold in 0 1 2 3 4; do for c in "$@"; do echo "$c $fold $seed"; done; done; done \
  | xargs -P ${PARALLEL:-6} -L 1 bash -c 'mkdir -p runs/gamp_vf/$0; run_one "$0" "$1" "$2"'
echo "ALL DONE: $*"
