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

RESTART_CHECKPOINT=${RESTART_CHECKPOINT:-false}
POSITION_ARGS=("")
SYNC_TO_DRIVE=${SYNC_TO_DRIVE:-false}
USE_TPU=${USE_TPU:-false}

while [[ $# > 0 ]]; do
    case "$1" in
    --restart)
        RESTART_CHECKPOINT=true
        shift
        ;;
    --sync)
        SYNC_TO_DRIVE=true
        shift
        ;;
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

if [ "$RESTART_CHECKPOINT" = "true" ]; then
    if [ "$USE_TPU" = "true" ]; then
        # gsutil TODO
        echo TODO
    else
        cd models/${MODEL_NAME}/${MODEL_VERSION}
        rm -rf checkpoint train ckpt-*
        cd $DIR
    fi
fi


# SCRIPT_PATH=$DIR/../tensorflow_model_garden/research/object_detection/model_main_tf2.py
SCRIPT_PATH=$DIR/model_main_tf2.py
python ${SCRIPT_PATH} \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --model_dir=${MODEL_DIR}\
  --checkpoint_every_n=100 \
  --alsologtostderr \
  --use_tpu=$USE_TPU

if [ "${SYNC_TO_DRIVE}" = "true" ]; then
    ./sync_workspace_with_drive.sh $MODEL_DIR --to
fi