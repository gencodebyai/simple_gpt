#!/bin/bash

# 设置参数
NUM_PAGES=10
TEST_SIZE=0.1
VAL_SIZE=0.1
EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=3e-4

# 创建日志目录
mkdir -p logs

# 记录开始时间
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/pipeline_${TIMESTAMP}.log"

echo "开始完整处理流程: $(date)" | tee $LOG_FILE

# 1. 数据处理
echo "开始数据处理阶段..." | tee -a $LOG_FILE
bash scripts/process_data.sh $NUM_PAGES $TEST_SIZE $VAL_SIZE

# 2. 训练模型
echo "开始训练模型..." | tee -a $LOG_FILE
bash scripts/run_train.sh

# 3. 测试模型
echo "开始测试模型..." | tee -a $LOG_FILE
bash scripts/run_inference.sh

echo "完整流程结束: $(date)" | tee -a $LOG_FILE