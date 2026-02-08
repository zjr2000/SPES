#!/bin/bash
# Example script for tokenizing a dataset using tokenize_data.py
# This script demonstrates how to tokenize text data into numpy memmap format
# for efficient training data loading.

# Usage:
#   1. Modify the parameters below according to your dataset and tokenizer
#   2. Run: bash data_process_scripts/tokenize_example.sh

python data_process_scripts/tokenize_data.py \
    --file_glob "/path/to/your/data/*.jsonl" \
    --tokenizer_name_or_path "Qwen/Qwen2.5-0.5B" \
    --output_prefix "/path/to/output/tokenized_" \
    --text_field "text" \
    --processes 8 \
    --batch_size 500 \
    --max_shard_bytes 4294967296 \
    --dtype "uint32"

# Parameters:
#   --file_glob: Glob pattern to match input files (supports .jsonl, .jsonl.gz, .jsonl.zst, .parquet)
#   --tokenizer_name_or_path: HuggingFace tokenizer name or local path
#   --output_prefix: Prefix for output .npy files (will be appended with process_id and shard_id)
#   --text_field: JSON field containing the text to tokenize (default: "text")
#   --processes: Number of parallel processes (default: 1)
#   --files_per_process: Optional, number of files to assign per process
#   --batch_size: Number of texts to batch before tokenizing (default: 500)
#   --max_shard_bytes: Maximum bytes per output shard file (default: 4GB)
#   --dtype: Data type for token IDs, "uint16" or "uint32" (default: "uint32")
