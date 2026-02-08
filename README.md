<div align="center">

# ⚡ SPES: SParse Expert Synchronization
### Pretraining A Large Language Model using Distributed GPUs: A Memory-Efficient Decentralized Paradigm

[![GitHub](https://img.shields.io/badge/GitHub-SPES-blue?logo=github)](https://github.com/zjr2000/SPES)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/xxxx.xxxxx)

</div>

---

## 📖 Introduction

**SPES** (**SP**arse **E**xpert **S**ync) is a cutting-edge, memory-efficient decentralized training framework designed for pretraining MoE LLMs across geographically distributed GPU nodes.

Unlike conventional paradigms that demand high-bandwidth interconnects, SPES enables the collaborative pretraining of **Mixture-of-Experts (MoE)** models where nodes operate semi-independently.

### 🌟 Key Features

| Feature | Description |
| :--- | :--- |
| 🌐 **Decentralized Training** | Operates without high-speed cross-node interconnects. Each node functions as an independent training unit with local DDP. |
| 💾 **Memory Efficiency** | Nodes only maintain gradients/optimizer states for their *local* subset of experts, drastically reducing memory footprint. |
| ⚡ **Sparse Sync** | Utilizes a lightweight gRPC parameter server to synchronize *only* trained parameters periodically. |
| 🔀 **Smart Merging** | Implements intelligent weighted merging with a decaying alpha schedule to ensure stable convergence during knowledge transfer. |

---

## 🚧 Roadmap & Status

- [x] **Release Training Code**
- [ ] Release pretrained model checkpoints & training logs
- [ ] Add detailed documentation for training and evaluation scripts

---

## 🔧 Installation

### Prerequisites
*   **Python:** `>= 3.10`
*   **CUDA:** `>= 12.1` (Tested on 12.4)
*   **PyTorch:** `2.5.1`
*   **Hardware:** NVIDIA GPUs (Tested on A100/A800/L40S)

### Quick Install

```bash
# 1. Clone the repository
git clone https://github.com/zjr2000/SPES.git
cd SPES

# 2. Install PyTorch (Adjust CUDA version if necessary)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 3. Install SPES and core dependencies
pip install -e '.[all]'

# 4. Install gRPC components
pip install grpcio==1.73.1 grpcio-tools==1.73.1 protobuf==6.31.0
```

### Evaluation Dependencies
To run benchmarks using the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness):
```bash
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
pip install "lm_eval[hf]"
```

---

## 📦 Data Preparation

SPES utilizes tokenized numpy memmap files (`.npy`) for high-performance data loading.

### 1. Tokenize Raw Data
Convert your `.jsonl` or `.parquet` files using the provided script:

```bash
python data_process_scripts/tokenize_data.py \
    --file_glob "/path/to/your/data/*.jsonl" \
    --tokenizer_name_or_path "Qwen/Qwen2.5-0.5B" \
    --output_prefix "/path/to/output/tokenized_" \
    --text_field "text" \
    --processes 8 \
    --batch_size 500 \
    --max_shard_bytes 4294967296 \
    --dtype "uint32"
```

### 2. Generate File List
Create a manifest file for the training configuration:
```bash
bash data_process_scripts/list_processed_files.sh /path/to/tokenized/data /path/to/output/file_list.txt
```

### 3. Update Config
Point your YAML configuration file (in `configs/`) to `file_list.txt`.

---

## 🚀 How to Run

SPES uses a **Client-Server** architecture:
1.  **Parameter Server:** Manages expert synchronization.
2.  **Training Clients:** Independent nodes performing local training.

### ⚙️ Configuration
Key SPES parameters in your YAML config:

```yaml
using_spes: true
spes_config:
  num_peers: 4                  # Total training nodes
  peer_id: 0                    # Current node ID (0-indexed)
  num_train_experts_per_node: 2 # Local experts per node
  sync_steps: 100               # Sync frequency
  server_addr: 127.0.0.1:50051  # Parameter Server Address
```

### Option A: Manual Launch (Step-by-Step)

**1. Start Parameter Server**
```bash
bash run_scripts/run_parameter_server.sh
```

**2. Start Training Clients (On each node)**
```bash
# Example: Launching on Node 1
bash run_scripts/run_single_node.sh 1

# Optional: Resume from checkpoint
bash run_scripts/run_single_node.sh 0 --resume
```

### Option B: Cluster Launch (Automated)

For SLURM or other schedulers where `RANK`, `MASTER_ADDR`, and `NPROC_PER_NODE` are set automatically:

```bash
bash run_scripts/run_cluster.sh
```
*This script automatically handles server startup on Rank 0 and isolates DDP to the local node.*

---

## 📊 Evaluation

### 1. Convert Checkpoints
Convert the sharded FSDP checkpoints to HuggingFace format:

```bash
# Syntax: <RUN_DIR> <SAVE_STEP> <MODEL_SIZE>
bash eval_scripts/convet_model_to_hf_unshard.sh output/spes_moe_3b_9b/node0 10000 A3B-9B
```

### 2. Run Benchmarks
Evaluate using `lm-evaluation-harness`:

```bash
bash eval_scripts/eval_full.sh <MODEL_PATH> <MODEL_NAME>
```

---

## 🙏 Acknowledgements

This project stands on the shoulders of giants. We explicitly thank the following projects and teams:

*   **[OLMo (Allen Institute for AI)](https://github.com/allenai/OLMo):** Our codebase is built upon the excellent modeling, training, and inference code provided by the [Ai2](https://allenai.org/) team.
*   **[MegaBlocks (Databricks)](https://github.com/databricks/megablocks):** We utilize MegaBlocks for efficient "dropless" Mixture-of-Experts (MoE) training and sparse operations.
*   **[LM Evaluation Harness (EleutherAI)](https://github.com/EleutherAI/lm-evaluation-harness):** Used for our few-shot evaluation framework and benchmarking.

## 📄 License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.