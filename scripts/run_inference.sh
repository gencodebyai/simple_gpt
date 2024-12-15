#!/bin/bash

# 设置推理参数
MODEL_PATH="checkpoints/model_epoch_10.pt"
MAX_LENGTH=100
TEMPERATURE=0.8

# 运行推理脚本
python3 src/inference.py \
    --prompt "你好，请问" \
    --model_path $MODEL_PATH \
    --max_length $MAX_LENGTH \
    --temperature $TEMPERATURE \
    2>&1 | tee logs/inference_$(date +%Y%m%d_%H%M%S).log 