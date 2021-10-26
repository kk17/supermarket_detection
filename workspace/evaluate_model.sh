#!/usr/bin/env bash

set -euox pipefail
XTRACE=${XTRACE:-false}
if [[ "$XTRACE" = "true" ]]; then
    set -x
fi
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR
if [ -f .env ]; then
    source .env
fi

MODEL_DIR=models/${MODEL_NAME}/${MODEL_VERSION}
PIPELINE_CONFIG_PATH=$MODEL_DIR/pipeline.config

USE_TPU=${USE_TPU:-false}
while [[ $# > 0 ]]; do
    case "$1" in
    --tpu)
        USE_TPU=true
        MODEL_DIR=$GS_PATH_PREFIX/$MODEL_DIR
        shift
        ;;
    *) # unknown flag/switch
        POSITION_ARGS+=("$1")
        shift
        ;;
    esac
done
set -- "${POSITION_ARGS[@]}"

# SCRIPT_PATH=$DIR/../tensorflow_model_garden/research/object_detection/model_main_tf2.py
SCRIPT_PATH=$DIR/model_main_tf2.py
python ${SCRIPT_PATH} \
  --model_dir=${MODEL_DIR} \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --checkpoint_dir=${MODEL_DIR} \
  --google_login $USE_TPU
