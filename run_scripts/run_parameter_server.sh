#!/bin/bash
# SPES Parameter Server 启动脚本

set -e

# ============================
# 配置参数 (可根据需要修改)
# ============================
TOTAL_PEERS=${TOTAL_PEERS:-2}                           # 总的 client 数量
NUM_TRAIN_EXPERTS_PER_NODE=${NUM_TRAIN_EXPERTS_PER_NODE:-8}  # 每个节点训练的 experts 数量
PORT=${PORT:-50051}                                     # 服务器监听端口
MERGE_INTERVAL=${MERGE_INTERVAL:-500}                   # 合并间隔步数
MERGE_ALPHA_START=${MERGE_ALPHA_START:-0.01}            # 合并起始 alpha 值
MERGE_DECAY_STEPS=${MERGE_DECAY_STEPS:-10000}           # alpha 衰减步数

# ============================
# 脚本路径
# ============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SERVER_SCRIPT="${PROJECT_ROOT}/spes/spes/spes_server_knowledge_transfer.py"

# ============================
# 打印配置信息
# ============================
echo "========================================"
echo "SPES Parameter Server Configuration"
echo "========================================"
echo "Total Peers:              ${TOTAL_PEERS}"
echo "Experts per Node:         ${NUM_TRAIN_EXPERTS_PER_NODE}"
echo "Server Port:              ${PORT}"
echo "Merge Interval:           ${MERGE_INTERVAL}"
echo "Merge Alpha Start:        ${MERGE_ALPHA_START}"
echo "Merge Decay Steps:        ${MERGE_DECAY_STEPS}"
echo "========================================"
echo ""

# ============================
# 启动服务器
# ============================
echo "[INFO] Starting SPES Parameter Server..."

python "${SERVER_SCRIPT}" \
    --total_peers "${TOTAL_PEERS}" \
    --num_train_experts_per_node "${NUM_TRAIN_EXPERTS_PER_NODE}" \
    --port "${PORT}" \
    --merge_interval "${MERGE_INTERVAL}" \
    --merge_alpha_start "${MERGE_ALPHA_START}" \
    --merge_decay_steps "${MERGE_DECAY_STEPS}"
