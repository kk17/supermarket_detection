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

RESTART_CHECKPOINT=${RESTART_CHECKPOINT:-false}
POSITION_ARGS=("")
SYNC_TO_DRIVE=${SYNC_TO_DRIVE:-false}
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
    *) # unknown flag/switch
        POSITION_ARGS+=("$1")
        shift
        ;;
    esac
done
set -- "${POSITION_ARGS[@]}"

if [ "$RESTART_CHECKPOINT" = "true" ]; then
    cd models/${MODEL_NAME}/${MODEL_VERSION}
    rm -rf checkpoint train ckpt-*
fi
cd $DIR

MODEL_DIR=models/${MODEL_NAME}/${MODEL_VERSION}

SCRIPT_PATH=$DIR/../tensorflow_model_garden/research/object_detection/model_main_tf2.py
python ${SCRIPT_PATH} \
  --pipeline_config_path=models/${MODEL_NAME}/${MODEL_VERSION}/pipeline.config\
  --model_dir=${MODEL_DIR}\
  --checkpoint_every_n=100 \
  --num_workers=3 \
  --alsologtostderr

if [ "${SYNC_TO_DRIVE}" = "true" ]; then
    ./sync_workspace_with_drive.sh $MODEL_DIR --to
fi