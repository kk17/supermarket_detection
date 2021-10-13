#!/usr/bin/env bash

set -euo pipefail
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

python exporter_main_v2.py \
  --pipeline_config_path=models/${MODEL_NAME}/${MODEL_VERSION}/pipeline.config \
  --trained_checkpoint_dir=models/${MODEL_NAME}/${MODEL_VERSION}/ \
  --output_directory=exported_models/${MODEL_NAME}/${MODEL_VERSION} \
  --input_type=image_tensor