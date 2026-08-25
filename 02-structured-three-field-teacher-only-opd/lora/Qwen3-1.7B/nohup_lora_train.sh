#!/usr/bin/env bash
set -euo pipefail

# Qwen3-1.7B 位于独立容器，本容器使用 GPU 0。
export CUDA_VISIBLE_DEVICES=0

llamafactory-cli train /root/lora_config/Qwen3-1.7B/lora_sft.yaml
