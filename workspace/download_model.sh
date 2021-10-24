#!/usr/bin/env bash

set -euox pipefail
XTRACE=${XTRACE:-false}
if [[ "$XTRACE" = "true" ]]; then
    set -x
fi
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd $DIR/pre_trained_models

MODEL_URL=${MODEL_URL:-http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz}

if [ -n $1 ]; then
    MODEL_URL=$1
fi

MODEL_FILE=$(basename $MODEL_URL)
if [ ! -f $MODEL_FILE ]; then
    wget $MODEL_URL
    tar -xvf ${MODEL_FILE}
    rm ${MODEL_FILE}
    MODEL_NAME=$(basename $MODEL_FILE .tar.gz)
    SIMPLE_MODEL_NAME=${MODEL_FILE%_coco17*} # retain the part before the _coco17
    mv $MODEL_NAME $SIMPLE_MODEL_NAME
fi

cd $DIR
PRETRAINED_MODEL_DIR=pre_trained_models/$SIMPLE_MODEL_NAME
MODEL_DIR=models/$SIMPLE_MODEL_NAME/v1
mkdir -p $MODEL_DIR
cp $PRETRAINED_MODEL_DIR/pipeline.config $MODEL_DIR/pipeline.config

cd $DIR
python ./update_pipeline_config.py -n $SIMPLE_MODEL_NAME -v "v1"
