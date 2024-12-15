#!/bin/bash

# 设置项目根目录
PROJECT_ROOT=$(pwd)

# 设置 PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# 创建必要的目录
mkdir -p logs
mkdir -p checkpoints
mkdir -p data/processed

# 设置推理参数
MODEL_PATH="${PROJECT_ROOT}/checkpoints/model_epoch_10.pt"
DATA_PATH="${PROJECT_ROOT}/data/processed/train.txt"
VOCAB_PATH="${PROJECT_ROOT}/data/processed/vocab.txt"
BLOCK_SIZE=128
MAX_LENGTH=100
TEMPERATURE=0.8

# 检查词表文件是否存在
if [ ! -f "$VOCAB_PATH" ]; then
    echo "词表文件不存在: $VOCAB_PATH"
    echo "正在运行数据预处理..."
    bash scripts/process_data.sh
fi

# 检查数据文件是否存在
if [ ! -f "$DATA_PATH" ]; then
    echo "训练数据文件不存在: $DATA_PATH"
    echo "正在运行数据预处理..."
    bash scripts/process_data.sh
fi

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "模型文件不存在: $MODEL_PATH"
    echo "正在训练模型..."
    bash scripts/run_train.sh
fi

echo "使用以下参数进行推理："
echo "模型路径: $MODEL_PATH"
echo "数据路径: $DATA_PATH"
echo "词表路径: $VOCAB_PATH"
echo "序列长度: $BLOCK_SIZE"
echo "生成长度: $MAX_LENGTH"
echo "采样温度: $TEMPERATURE"

# 运行推理脚本
python3 src/inference.py \
    --prompt "穆爾基普萊斯" \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --block_size $BLOCK_SIZE \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    2>&1 | tee logs/inference_$(date +%Y%m%d_%H%M%S).log