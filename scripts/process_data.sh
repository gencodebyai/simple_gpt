#!/bin/bash

# 设置参数
NUM_PAGES=${1:-10}  # 默认值为10
TEST_SIZE=${2:-0.1} # 默认值为0.1
VAL_SIZE=${3:-0.1}  # 默认值为0.1

# 创建日志目录
mkdir -p logs

# 获取当前时间戳
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/data_processing_${TIMESTAMP}.log"

echo "开始数据处理: $(date)" | tee $LOG_FILE

# 1. 爬取数据
echo "开始爬取数据..." | tee -a $LOG_FILE
python3 src/crawler.py \
    --num_pages $NUM_PAGES \
    2>&1 | tee -a $LOG_FILE

# 2. 预处理数据
echo "开始预处理数据..." | tee -a $LOG_FILE
python3 src/preprocessor.py \
    --test_size $TEST_SIZE \
    --val_size $VAL_SIZE \
    2>&1 | tee -a $LOG_FILE

echo "数据处理完成: $(date)" | tee -a $LOG_FILE 