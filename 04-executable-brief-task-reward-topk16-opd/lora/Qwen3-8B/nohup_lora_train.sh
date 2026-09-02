#!/usr/bin/env bash
set -euo pipefail

# Qwen3-8B 使用 GPU 0、1、2、3。
export CUDA_VISIBLE_DEVICES=0,1,2,3
export FORCE_TORCHRUN=1

llamafactory-cli train /root/lora_config/Qwen3-8B/lora_sft.yaml
