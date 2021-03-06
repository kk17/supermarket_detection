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

USE_TPU=${USE_TPU:-false}

MODEL_DIR=models/$MODEL_NAME/$MODEL_VERSION
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

echo "run the follow to set a tunnel for viewing tensorboard on colab:"
echo "ngrok http 6006 --region ap"

google_login=$USE_TPU python $DIR/my_tensorboard.py --logdir $MODEL_DIR
