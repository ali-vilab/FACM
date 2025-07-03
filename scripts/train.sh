# CONFIG_PATH=$1

PRECISION=${PRECISION:-bf16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NNODES=${WORLD_SIZE:-8}
NODE_RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-1235}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
RESULTS_PATH=${RESULTS_PATH:-"output"}
DATA_PATH=${DATA_PATH:-"path/to/imagenet"}
CKPT_PATH=${CKPT_PATH:-"cache/800ep-stg1.pt"}

accelerate launch \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $NODE_RANK \
    --num_processes  $(($GPUS_PER_NODE*$NNODES)) \
    --mixed_precision $PRECISION \
    --num_machines $NNODES \
    train.py \
    --results-dir $RESULTS_PATH --data-dir $DATA_PATH --ckpt-path $CKPT_PATH --distill \
    --ckpt-every 10000 --eval-every 10000 --fid-every 10000 --global-batch-size 1024 --accumulation 1 \
    --t-type default --p 0.5 --cfgw 1.75 --mean 0.8 --std 1.6 --glow 0.125