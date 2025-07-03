#!/bin/bash

# Configuration
DATA_PATH=${DATA_PATH:-"path/to/your/dataset"}
OUTPUT_PATH=${OUTPUT_PATH:-"path/to/your/output"}
CONFIG_PATH=${CONFIG_PATH:-"ldit/lightningdit.yaml"}
DATA_SPLIT=${DATA_SPLIT:-"imagenet_train"}
IMAGE_SIZE=${IMAGE_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-20}
NUM_WORKERS=${NUM_WORKERS:-8}

# Distributed settings
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1234}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Launch feature extraction
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    utils/extract.py \
    --data_path $DATA_PATH \
    --output_path $OUTPUT_PATH \
    --config $CONFIG_PATH \
    --data_split $DATA_SPLIT \
    --image_size $IMAGE_SIZE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS