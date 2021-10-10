#!/usr/bin/env bash

set -euo pipefail
XTRACE=${XTRACE:-false}
if [[ "$XTRACE" = "true" ]]; then
    set -x
fi
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


python exporter_main_v2.py \
  --pipeline_config_path=models/efficientdet_d0_coco17_tpu/v1/pipeline.config \
  --trained_checkpoint_dir=models/efficientdet_d0_coco17_tpu/v1/ \
  --output_directory=exported_models/efficientdet_d0 \
  --input_type=image_tensor