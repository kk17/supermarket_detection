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
if [ -n "${1:-}" ]; then
    MODEL_NAME=$1
fi

MODEL_VERSION=${MODEL_VERSION:-v1}
if [ -n "${2:-}" ]; then
    MODEL_VERSION=$2
fi

SCRIPT_PATH=$DIR/../tensorflow_model_garden/research/object_detection/model_main_tf2.py
python ${SCRIPT_PATH} \
  --model_dir=models/${MODEL_NAME}/${MODEL_VERSION}\
  --pipeline_config_path=models/${MODEL_NAME}/${MODEL_VERSION}/pipeline.config\
  --checkpoint_dir=models/${MODEL_NAME}/${MODEL_VERSION}
