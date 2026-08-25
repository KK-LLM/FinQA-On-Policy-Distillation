#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

llamafactory-cli train /root/lora_config/Qwen3-8B/lora_sft.yaml
