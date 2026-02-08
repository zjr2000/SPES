# ==========================================
# 2. 捕获调度系统传入的环境变量 (兼容单机模式)
# ==========================================
# 如果系统没有提供 RANK，默认为 0 (单机调试模式)
CURRENT_RANK=${RANK:-0}
# 如果系统没有提供 MASTER_ADDR，默认为 localhost
RAW_MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
GPUS_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi -L | wc -l)}

# ==========================================
# [解析] SPES Server 通信用的真实 IP
# ==========================================
# 注意：这个 IP 是给 SPES Server 用的，不是给 torchrun DDP 用的
if [ "$RAW_MASTER_ADDR" == "127.0.0.1" ] || [ "$RAW_MASTER_ADDR" == "localhost" ]; then
    NODE0_IP="127.0.0.1"
else
    # 方法 1: 使用 getent
    NODE0_IP=$(getent hosts "$RAW_MASTER_ADDR" | awk '{ print $1 }' | head -n 1)
    
    # 方法 2: Python 备选
    if [ -z "$NODE0_IP" ]; then
        NODE0_IP=$(python -c "import socket; print(socket.gethostbyname('$RAW_MASTER_ADDR'))")
    fi
    
    # 保底
    if [ -z "$NODE0_IP" ]; then
        NODE0_IP=$RAW_MASTER_ADDR
    fi
fi

echo "------------------------------------------------"
echo "Node Initialized:"
echo "  RANK (Peer ID) : ${CURRENT_RANK}"
echo "  SPES Server IP : ${NODE0_IP}"
echo "  GPUs per Node  : ${GPUS_PER_NODE}"
echo "------------------------------------------------"

# ==========================================
# 3. Node 0 特殊逻辑：启动 SPES Server
# ==========================================
SERVER_PID=""
if [ "$CURRENT_RANK" -eq 0 ]; then
    echo "[Node 0] Starting SPES Server in background..."
    
    # 启动 SPES Server
    nohup python olmo/spes/spes_server_knowledge_transfer.py \
        --total_peers 4 \
        --num_train_experts_per_node 2 \
        --merge_interval 500 \
        --merge_alpha_start 0.05 \
        --merge_decay_steps 20000 > spes_server.log 2>&1 &
    
    SERVER_PID=$!
    echo "[Node 0] SPES Server started with PID: ${SERVER_PID}"
    
    # 等待几秒确保 Server 端口已打开
    sleep 5
fi

# ==========================================
# 4. 准备训练参数
# ==========================================

# 4.1 计算 Seed
DATA_SEED=$((6198 + CURRENT_RANK * 1000))

# 4.2 拼接 SPES Server 地址 (应用层通信)
SPES_FULL_ADDR="${NODE0_IP}:50051"

# ==========================================
# 5. 关键步骤：设置本地 DDP 隔离环境
# ==========================================
# 清除可能干扰 DDP 的全局变量
unset MASTER_ADDR
unset MASTER_PORT
unset WORLD_SIZE
unset RANK

# 生成一个本地随机端口，用于本机的 GPU 之间通信
# 这确保了即使是在同一台机器上多次运行，也不会冲突
LOCAL_MASTER_PORT=$(shuf -i 20000-60000 -n 1)
LOCAL_MASTER_ADDR="127.0.0.1"

echo "------------------------------------------------"
echo "Starting Torchrun (Local DDP Group):"
echo "  Local Master   : ${LOCAL_MASTER_ADDR}"
echo "  Local Port     : ${LOCAL_MASTER_PORT}"
echo "  SPES Target    : ${SPES_FULL_ADDR}"
echo "------------------------------------------------"

# ==========================================
# 6. 启动训练 (Client)
# ==========================================

# 核心修改：
# 1. 显式传入 --master_addr=127.0.0.1，强制 DDP 只在本地回环，不走网卡。
# 2. 显式传入 --master_port，避免端口冲突。
# 3. 保留 --nnodes=1 --node_rank=0，因为在 SPES 架构下，每个节点都是独立的 DDP 组。

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_addr=${LOCAL_MASTER_ADDR} \
    --master_port=${LOCAL_MASTER_PORT} \
    scripts/train.py \
    configs/spes_experiments_scaling/spes_moe_3b_9b_bsz1024_32a800_4nodes_norm_router.yaml \
    --run_name=spes_moe_3b_9b_bsz1024_32a800_4nodes_norm_router_${CURRENT_RANK} \
    --spes_config.server_addr=${SPES_FULL_ADDR} \
    --spes_config.peer_id=${CURRENT_RANK} \
    --data.seed=${DATA_SEED} \
    --try_load_latest_save=true \
    --save_overwrite=true

# ==========================================
# 7. 收尾工作
# ==========================================
# 只有启动了 Server 的节点才需要杀进程
if [ -n "$SERVER_PID" ]; then
    echo "[Node 0] Training finished. Shutting down SPES Server (PID: ${SERVER_PID})..."
    kill $SERVER_PID
fi