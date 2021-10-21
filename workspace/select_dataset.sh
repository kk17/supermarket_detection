#!/usr/bin/env bash

set -euo pipefail
XTRACE=${XTRACE:-false}
if [[ "$XTRACE" = "true" ]]; then
    set -x
fi
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

DATASETS=()
for tfrecord_path in $(find ./data -name "train.tfrecord"); do 
    DATASET_DIR=$(dirname $tfrecord_path)
    DATASETS+=($DATASET_DIR)
done

echo "Avaliable DATASETS:"
printf "%3s %-30s\n" "No." "Name" 
for i in "${!DATASETS[@]}"; do
    printf "%3d %-30s\n" ${i} ${DATASETS[$i]};
done

read -p "Please input select DATASET No. [0]: " SELECTED
SELECTED=${SELECTED:-0}
export DATASET_DIR=${DATASETS[$SELECTED]}
echo "Select DATASET is $DATASET_DIR"

if [ ! -f .env ]; then
    cat <<EOT >> .env
DATASET_DIR=${DATASETS[$SELECTED]}
EOT
else
    sed -i -E "s|DATASET_DIR=.+|DATASET_DIR=${DATASETS[$SELECTED]}|g" .env
fi