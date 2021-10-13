#!/usr/bin/env bash

set -euox pipefail
XTRACE=${XTRACE:-false}
if [[ "$XTRACE" = "true" ]]; then
    set -x
fi
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR

MODEL_NAME=${MODEL_NAME:-efficientdet_d0_coco17_tpu}
if [ -n "${1:-}" ]; then
    MODEL_NAME=$1
fi

MODEL_VERSION=${MODEL_VERSION:-v1}
if [ -n "${2:-}" ]; then
    MODEL_VERSION=$2
fi

python model_main_tf2.py \
  --pipeline_config_path=models/${MODEL_NAME}/${MODEL_VERSION}/pipeline.config\
  --model_dir=models/${MODEL_NAME}/${MODEL_VERSION}\
  --checkpoint_every_n=100 \
  --num_workers=3 \
  --alsologtostderr