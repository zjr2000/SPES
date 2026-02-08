#!/bin/bash

# 指定要查找的目录
SEARCH_DIR=$1

# 输出文件
OUTPUT_FILE=$2

# 查找所有文件并输出绝对路径到文件
find "$SEARCH_DIR" -type f > "$OUTPUT_FILE"
