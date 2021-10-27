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
GS_PATH_PREFIX=${GS_PATH_PREFIX:-}
SYNC_FROM_GS=${SYNC_FROM_GS:-"false"}
SYNC_TO_GS=${SYNC_TO_GS:-"false"}
SYNC_SUB_DIR=${SYNC_SUB_DIR:-.}

SYNC_SUB_DIRS=()
while [[ $# > 0 ]]; do
    case "$1" in
    --from)
        SYNC_FROM_GS=true
        shift
        ;;
    --to)
        SYNC_TO_GS=true
        shift
        ;;
    --current-model)
        MODEL_DIR=models/${MODEL_NAME}/${MODEL_VERSION}
        SYNC_SUB_DIRS+=($MODEL_DIR)
        shift
        ;;
    *) # unknown flag/switch
        SYNC_SUB_DIRS+=("$1")
        shift
        ;;
    esac
done
set -- "${SYNC_SUB_DIRS[@]}"
if [ ${#SYNC_SUB_DIRS[@]} -eq 0 ]; then
    SYNC_SUB_DIRS=($SYNC_SUB_DIR)
fi


if [ "$SYNC_FROM_GS" = "true" ]; then
    for DIR in ${SYNC_SUB_DIRS[@]}; do
        echo "sync  $DIR from gcs"
        gsutil cp -r $GS_PATH_PREFIX/$DIR $(dirname ./$DIR)
    done
fi

if [ "$SYNC_TO_GS" = "true" ]; then
    for DIR in ${SYNC_SUB_DIRS[@]}; do
        echo "sync  $DIR to gcs"
        gsutil cp -r ./$DIR $(dirname $GS_PATH_PREFIX/$DIR)
    done
fi
