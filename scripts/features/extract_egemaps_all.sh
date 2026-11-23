#!/usr/bin/env bash
# eGeMAPS v02 functionals for official train, every external source and the eval trials.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY_TRAIN:-python}
X="$PY FLAG_2027/feature_extraction/extract_egemaps.py --workers 12"
$X --manifest train_set/train_set/train_English.txt --audio-root train_set/train_set --output-csv features/egemaps/train_English.csv
$X --manifest features/external/manifests/mav3_train.txt --audio-root external_data/mavceleb/work/v3 --output-csv features/egemaps/mav3_train.csv
$X --manifest features/external/manifests/vox2test_train.txt --audio-root external_data/voxceleb2/work/test --output-csv features/egemaps/vox2test_train.csv
$X --manifest features/external/manifests/mav1_train.txt --audio-root external_data/mavceleb/work/v1 --output-csv features/egemaps/mav1_train.csv
for t in no_gender gender; do for l in English Bangla; do
  $X --manifest features/eval_manifests/$t/${l}_valid.txt --audio-root evaluation_phase/$t --output-csv features/egemaps/eval_${t}_${l}_valid.csv
done; done
echo "egemaps all done"
