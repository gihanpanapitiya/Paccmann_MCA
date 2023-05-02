#!/bin/bash

# Launches model for preprocessing from within the container


####################################################################
## Exceute preprocess script for your model with the CMD arguments ##
####################################################################
preprocess="/usr/local/Paccmann_MCA/preprocess.py"


CUDA_VISIBLE_DEVICES=$1
CANDLE_DATA_DIR=$2

CMD="python3 ${preprocess}"


echo "using container "
echo "using CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}"
echo "using CANDLE_DATA_DIR ${CANDLE_DATA_DIR}"
echo "running command ${CMD}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} CANDLE_DATA_DIR=${CANDLE_DATA_DIR} $CMD
