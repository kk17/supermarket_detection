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
SYNC_SUB_DIR=${SYNC_SUB_DIR:-.}

rsync -rtuv --exclude '.env' ./$SYNC_SUB_DIR $DRIVER_DIR_PATH/$SYNC_SUB_DIR
rsync -rtuv --exclude '.env' --delete  $DRIVER_DIR_PATH/$SYNC_SUB_DIR ./$SYNC_SUB_DIR