#!/usr/bin/env bash
set -euo pipefail

# 后台启动 Qwen3-1.7B Brief + Program + Answer 实验 C LoRA 训练。
mkdir -p /root/train_output/logs
nohup bash /root/lora_config/Qwen3-1.7B/nohup_lora_train.sh > /root/train_output/logs/Qwen3-1.7B-LoRA-BPA-C-0826.log 2>&1 &

echo "Qwen3-1.7B LoRA training started, pid=$!"
echo "Log: /root/train_output/logs/Qwen3-1.7B-LoRA-BPA-C-0826.log"

# 查看训练日志：
tail -f /root/train_output/logs/Qwen3-1.7B-LoRA-BPA-C-0826.log

# 训练完成后融合最终 output_dir 中的 LoRA adapter：
# export CUDA_VISIBLE_DEVICES=0
# mkdir -p /root/merge_result
# llamafactory-cli export /root/lora_config/Qwen3-1.7B/merge_config.yaml
