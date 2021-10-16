#!/usr/bin/env bash

set -euo pipefail
XTRACE=${XTRACE:-false}
if [[ "$XTRACE" = "true" ]]; then
    set -x
fi
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR/models

MODELS=()
VERSIONS=()
for config_path in $(find . -name "pipeline.config"); do 
    MODEL_DIR=$(dirname $config_path)
    MODEL_VERSION=$(basename $MODEL_DIR)
    MODEL_NAME=$(basename $(dirname $MODEL_DIR))
    MODELS+=($MODEL_NAME)
    VERSIONS+=($MODEL_VERSION)
done

echo "Avaliable models:"
printf "%3s %-30s %-3s\n" "No." "Name" "Version"
for i in "${!MODELS[@]}"; do
    printf "%3d %-30s %-3s\n" ${i} ${MODELS[$i]} ${VERSIONS[$i]};
done

read -p "Please input select model No. [0]: " SELECTED
SELECTED=${SELECTED:-0}
export MODEL_NAME=${MODELS[$SELECTED]}
export MODEL_VERSION=${VERSIONS[$SELECTED]}
echo "Select model is $MODEL_NAME $MODEL_VERSION"

cd $DIR
if [ ! -f .env ]; then
    cat <<EOT >> .env
MODEL_NAME=${MODELS[$SELECTED]}
MODEL_VERSION=${VERSIONS[$SELECTED]}
EOT
else
    sed -i -E "s|MODEL_NAME=.+|MODEL_NAME=${MODELS[$SELECTED]}|g" .env
    sed -i -E "s|MODEL_VERSION=.+|MODEL_VERSION=${VERSIONS[$SELECTED]}|g" .env
fi