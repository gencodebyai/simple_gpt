#!/bin/bash

# 创建必要的目录
mkdir -p data/raw
mkdir -p data/processed
mkdir -p checkpoints
mkdir -p logs

# 设置训练参数 - 减小 batch_size 和 block_size
EPOCHS=10
BATCH_SIZE=8  # 从32减小到8
LEARNING_RATE=3e-4
BLOCK_SIZE=128  # 从256减小到128
DATA_PATH="data/processed/train.txt"

# 运行训练脚本
python3 src/train.py \
    --data_path $DATA_PATH \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --block_size $BLOCK_SIZE \
    2>&1 | tee logs/train_$(date +%Y%m%d_%H%M%S).log 