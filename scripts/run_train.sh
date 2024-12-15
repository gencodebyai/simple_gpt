#!/bin/bash

# 设置项目根目录
PROJECT_ROOT=$(pwd)

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 创建必要的目录
mkdir -p data/raw
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs

# 设置训练参数
EPOCHS=1
BATCH_SIZE=8
LEARNING_RATE=3e-4
BLOCK_SIZE=128
DATA_PATH="data/processed/train.txt"

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "训练数据文件不存在，请先运行数据预处理脚本"
    exit 1
fi

# 运行训练脚本
PYTHONPATH="${PROJECT_ROOT}" python3 src/train.py \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --block_size $BLOCK_SIZE \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log