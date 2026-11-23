#!/usr/bin/env bash
# Language ID + V2 features for one external source. Usage: process_source.sh <name> <work-dir>
set -euo pipefail
cd "$(dirname "$0")/../.."
name=$1; root=$2
HF_HUB_OFFLINE=1 ${PY_AV:-python} FLAG_2027/external_data/language_id.py \
  --manifest features/external/manifests/${name}_train.txt --audio-root $root \
  --output-csv features/external/${name}_train_language.csv > features/external/${name}_language.log 2>&1
tail -1 features/external/${name}_language.log
scripts/features/extract_external_source.sh $name $root
