#!/usr/bin/env bash

set -euox pipefail
XTRACE=${XTRACE:-false}
if [[ "$XTRACE" = "true" ]]; then
    set -x
fi
IFS=$'\n\t'
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


# Install python packages
pip install -r requirements.txt


mkdir -p workspace
cd workspace
mkdir -p exported_models
# The model we are using requires [TensorFlow Models](https://github.com/tensorflow/models), 
# which we can easily install using [ETA](https://github.com/voxel51/eta), a package bundled with FiftyOne:
eta install models
ETA_MODULE_DIR=$(dirname $(python -c "import eta; print(eta.__file__)"))
TF_MODELS_DIR=tensorflow/models

cd ${ETA_MODULE_DIR}/${TF_MODELS_DIR}/research/
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
python -m pip install .

SCRIPT_PATH=research/object_detection/model_main_tf2.py
cp ${ETA_MODULE_DIR}/${TF_MODELS_DIR}/${SCRIPT_PATH} .

SCRIPT_PATH=research/object_detection/exporter_main_v2.py
cp ${ETA_MODULE_DIR}/${TF_MODELS_DIR}/${SCRIPT_PATH} .

