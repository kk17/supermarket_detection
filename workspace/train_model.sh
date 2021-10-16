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
MODEL_NAME=${MODEL_NAME:-efficientdet_d0_coco17_tpu}
if [ $# -gt 1 ]; then
    MODEL_NAME=$1
fi

MODEL_VERSION=${MODEL_VERSION:-v1}
if [ $# -gt 2 ]; then
    MODEL_VERSION=$2
fi

RESTART_CHECKPOINT=${RESTART_CHECKPOINT:-false}
if [ "$RESTART_CHECKPOINT" = "true" ]; then
    cd models/${MODEL_NAME}/${MODEL_VERSION}
    rm -rf checkpoint train ckpt-*
fi
cd $DIR

SCRIPT_PATH=$DIR/../tensorflow_model_garden/research/object_detection/model_main_tf2.py
python ${SCRIPT_PATH} \
  --pipeline_config_path=models/${MODEL_NAME}/${MODEL_VERSION}/pipeline.config\
  --model_dir=models/${MODEL_NAME}/${MODEL_VERSION}\
  --checkpoint_every_n=100 \
  --num_workers=3 \
  --alsologtostderr