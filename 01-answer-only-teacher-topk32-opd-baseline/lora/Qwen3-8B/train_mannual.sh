#!/usr/bin/env bash
set -euo pipefail

# 启动 Qwen3-8B LoRA 训练。
mkdir -p /root/train_output/logs
nohup bash /root/lora_config/Qwen3-8B/nohup_lora_train.sh \
    > /root/train_output/logs/Qwen3-8B-LoRA-nothink.log 2>&1 &

# 查看训练日志：
# tail -f /root/train_output/logs/Qwen3-8B-LoRA-nothink.log

# 训练完成后融合 LoRA adapter：
# export CUDA_VISIBLE_DEVICES=0
# mkdir -p /root/merge_result
# llamafactory-cli export /root/lora_config/Qwen3-8B/merge_config.yaml
