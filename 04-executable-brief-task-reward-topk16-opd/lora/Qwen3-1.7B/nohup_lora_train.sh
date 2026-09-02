#!/usr/bin/env bash
set -euo pipefail

# Qwen3-1.7B 使用 GPU 0、1、2。
export CUDA_VISIBLE_DEVICES=0,1,2
export FORCE_TORCHRUN=1

llamafactory-cli train /root/lora_config/Qwen3-1.7B/lora_sft.yaml
