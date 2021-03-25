#!/usr/bin/env bash

PYTHON=/home/cver/anaconda3/envs/Aerial/bin/python

CONFIG=$1
GPUS=$2

$PYTHON -m torch.distributed.launch --nproc_per_node=$GPUS --master_port 2200\
    $(dirname "$0")/train1.py $CONFIG --launcher pytorch ${@:3}
