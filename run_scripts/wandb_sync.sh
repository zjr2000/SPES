#!/bin/bash

# ============================================
# WandB 离线日志同步脚本 (仅同步最新 Run)
# ============================================

export WANDB_API_KEY=your_wandb_api_key_here  # 替换为你的 WandB API Key
WANDB_DIR="${WANDB_DIR:-/path/to/your/wandb/logs}"  # wandb 离线日志路径
SYNC_INTERVAL="${SYNC_INTERVAL:-60}"                # 同步间隔（秒），默认改为60秒，为了更及时看到Loss
LOG_FILE="${LOG_FILE:-/var/log/wandb_sync.log}"     # 日志文件路径
MAX_RETRIES=3                                        # 最大重试次数

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo -e "${timestamp} [${level}] ${message}" | tee -a "$LOG_FILE"
}

# 检查 wandb 是否安装
check_wandb() {
    if ! command -v wandb &> /dev/null; then
        log "ERROR" "${RED}wandb 未安装，请先运行: pip install wandb${NC}"
        exit 1
    fi
}

# 检查目录是否存在
check_directory() {
    if [ ! -d "$WANDB_DIR" ]; then
        log "ERROR" "${RED}目录不存在: $WANDB_DIR${NC}"
        exit 1
    fi
}

# 同步单个 run
sync_run() {
    local run_dir=$1
    local retry_count=0
    
    # 注意：wandb sync 会同步所有数据（metrics, loss, system stats等）
    # 对于正在运行的任务，wandb sync 会同步当前已有的数据然后退出
    
    while [ $retry_count -lt $MAX_RETRIES ]; do
        log "INFO" "正在同步最新 Run: $(basename "$run_dir") (尝试 $((retry_count + 1))/$MAX_RETRIES)"
        
        # 这里的 sync 命令会上传最新的 loss 数据
        if wandb sync "$run_dir" 2>&1 | tee -a "$LOG_FILE"; then
            log "INFO" "${GREEN}同步成功: $run_dir${NC}"
            return 0
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $MAX_RETRIES ]; then
                log "WARN" "${YELLOW}同步失败，等待 5 秒后重试...${NC}"
                sleep 5
            fi
        fi
    done
    
    log "ERROR" "${RED}同步失败（已重试 $MAX_RETRIES 次）: $run_dir${NC}"
    return 1
}

# 【核心修改】查找并同步最新的离线 run
sync_latest_run() {
    log "INFO" "正在扫描最新生成的 offline-run..."
    
    # 使用 ls -td 按时间倒序排列，取第一个
    # 2>/dev/null 屏蔽如果没有文件时的报错
    local latest_run=$(ls -td "$WANDB_DIR"/offline-run-* 2>/dev/null | head -n 1)
    
    if [ -z "$latest_run" ]; then
        log "WARN" "${YELLOW}在 $WANDB_DIR 下未找到任何 offline-run 目录${NC}"
        return 1
    fi

    log "INFO" "定位到最新目录: $latest_run"
    sync_run "$latest_run"
}

# 单次同步模式
sync_once() {
    log "INFO" "========== 开始单次同步 =========="
    sync_latest_run
    log "INFO" "========== 单次同步完成 =========="
}

# 守护进程模式
daemon_mode() {
    log "INFO" "========== 启动守护进程模式 (只监控最新 Run) =========="
    log "INFO" "监控目录: $WANDB_DIR"
    log "INFO" "同步间隔: ${SYNC_INTERVAL}秒"
    
    while true; do
        sync_latest_run
        log "INFO" "等待 ${SYNC_INTERVAL} 秒后刷新 Loss..."
        sleep "$SYNC_INTERVAL"
    done
}

# 显示帮助
show_help() {
    cat << EOF
WandB 离线日志同步脚本 (Latest Run Only)

功能: 自动寻找目录下最新的 offline-run 文件夹并同步，适合查看正在训练任务的 Loss。

用法: $0 [选项]

选项:
    -d, --dir PATH       指定 wandb 日志目录 (默认: $WANDB_DIR)
    -i, --interval SEC   同步间隔秒数 (默认: $SYNC_INTERVAL)
    -l, --log PATH       日志文件路径 (默认: $LOG_FILE)
    -o, --once           单次同步模式（同步后退出）
    -D, --daemon         守护进程模式（持续运行，推荐用于实时看 Loss）
    -h, --help           显示此帮助信息

示例:
    $0 -d ./wandb -D            # 守护模式，持续同步最新的 run
    $0 -d ./wandb -o            # 只同步一次最新的 run
EOF
}

# 主函数
main() {
    local mode="daemon"
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -d|--dir)
                WANDB_DIR="$2"
                shift 2
                ;;
            -i|--interval)
                SYNC_INTERVAL="$2"
                shift 2
                ;;
            -l|--log)
                LOG_FILE="$2"
                shift 2
                ;;
            -o|--once)
                mode="once"
                shift
                ;;
            -D|--daemon)
                mode="daemon"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                echo "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 检查环境
    check_wandb
    check_directory
    
    # 确保日志目录存在
    mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true
    
    # 根据模式运行
    case $mode in
        once)
            sync_once
            ;;
        daemon)
            daemon_mode
            ;;
    esac
}

# 捕获退出信号
trap 'log "INFO" "收到退出信号，正在停止..."; exit 0' SIGINT SIGTERM

# 运行主函数
main "$@"