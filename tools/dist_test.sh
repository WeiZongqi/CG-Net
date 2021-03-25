#!/usr/bin/env bash

PYTHON=/home/cver/anaconda3/envs/Aerial/bin/python

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS  --master_port 2100\
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
