#!/bin/bash

# Unified launch script for SPES distributed training
# Usage: ./launch_node.sh <node_id> [options]
# Example: ./launch_node.sh 0
#          ./launch_node.sh 1 --resume

set -e

# Check if node_id is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <node_id> [--resume]"
    echo "  node_id: 0, 1, 2, or 3"
    echo "  --resume: Resume from latest checkpoint (optional)"
    exit 1
fi

NODE_ID=$1
RESUME_MODE=false

# Check for resume flag
if [ "$2" = "--resume" ]; then
    RESUME_MODE=true
fi

# Validate node_id
if [[ ! "$NODE_ID" =~ ^[0-3]$ ]]; then
    echo "Error: node_id must be 0, 1, 2, or 3"
    exit 1
fi

# Configuration
CONFIG_FILE="configs/spes_experiments_scaling/spes_moe_3b_9b_4nodes_norm_router.yaml"
RUN_NAME="spes_moe_3b_9b_4nodes_norm_router"
LOAD_PATH="output/OLMoE-A3B-9B-from-Qwen3-noise05-unsharded"
NPROC_PER_NODE=8

# Data seeds for each node
declare -A DATA_SEEDS
DATA_SEEDS[0]=6198
DATA_SEEDS[1]=7198
DATA_SEEDS[2]=8198
DATA_SEEDS[3]=9198

DATA_SEED=${DATA_SEEDS[$NODE_ID]}

# Environment variables
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_MODE="offline"
export WANDB_API_KEY=74d6d35108ddf5e89fafb7b119508765893f2698

echo "=========================================="
echo "SPES Distributed Training - Node $NODE_ID"
echo "=========================================="
echo "Config: $CONFIG_FILE"
echo "Run Name: ${RUN_NAME}_node_${NODE_ID}"
echo "Data Seed: $DATA_SEED"
echo "Resume Mode: $RESUME_MODE"
echo "=========================================="

if [ "$RESUME_MODE" = true ]; then
    # Resume training from checkpoint
    torchrun --nproc_per_node=$NPROC_PER_NODE scripts/train.py $CONFIG_FILE \
        --run_name=${RUN_NAME}_node_${NODE_ID}_resume \
        --spes_config.peer_id=$NODE_ID \
        --data.seed=$DATA_SEED \
        --try_load_latest_save=true \
        --save_overwrite=true
else
    # Start fresh training
    torchrun --nproc_per_node=$NPROC_PER_NODE scripts/train.py $CONFIG_FILE \
        --run_name=${RUN_NAME}_node_${NODE_ID} \
        --spes_config.peer_id=$NODE_ID \
        --data.seed=$DATA_SEED \
        --reset_optimizer_state=true \
        --reset_trainer_state=true \
        --restore_dataloader=false \
        --no_pre_train_checkpoint=true \
        --load_path=$LOAD_PATH \
        --save_overwrite=true
fi
