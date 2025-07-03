#!/bin/bash

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29502}
torchrun --nproc_per_node=$GPUS_PER_NODE --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=$MASTER_PORT \
    test.py \
    "$@"

echo "test.py finished."