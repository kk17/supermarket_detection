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
DRIVER_DIR_PATH=${DRIVER_DIR_PATH:-/content/drive/MyDrive/supermarket_detection_workspace}
SYNC_FROM_DRIVE=${SYNC_FROM_DRIVE:-"false"}
SYNC_TO_DRIVE=${SYNC_TO_DRIVE:-"false"}
DELETE_NO_EXIST=${DELETE_NO_EXIST:-"true"}
SYNC_SUB_DIR=${SYNC_SUB_DIR:-.}

SYNC_SUB_DIRS=()
while [[ $# > 0 ]]; do
    case "$1" in
    --from)
        SYNC_FROM_DRIVE=true
        shift
        ;;
    --to)
        SYNC_TO_DRIVE=true
        shift
        ;;
    --no-delete)
        DELETE_NO_EXIST=false
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

RSYNC_ARGS=(-rtuv --exclude '.env')
if [ $DELETE_NO_EXIST = "true" ]; then
    RSYNC_ARGS+=(--delete)
fi

if [ "$SYNC_FROM_DRIVE" = "true" ]; then
    for DIR in ${SYNC_SUB_DIRS[@]}; do
        echo "sync  $DIR from dirve"
        rsync  ${RSYNC_ARGS[@]} $DRIVER_DIR_PATH/$DIR/ ./$DIR/
    done
fi

if [ "$SYNC_TO_DRIVE" = "true" ]; then
    for DIR in ${SYNC_SUB_DIRS[@]}; do
        echo "sync  $DIR to dirve"
        rsync ${RSYNC_ARGS[@]} ./$DIR/ $DRIVER_DIR_PATH/$DIR/
    done
fi
