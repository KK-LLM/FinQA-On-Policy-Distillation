#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${MODEL_ROOT:-/root/model}
DATA_ROOT=${DATA_ROOT:-/root/data}
TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-/root/train_output}
CACHE_ROOT=${CACHE_ROOT:-/root/autodl-tmp/opd_finqa_lora/cache}


# ==================== 模型与数据路径 ====================
export STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"${MODEL_ROOT}/Qwen3-1.7B-Lora"}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${MODEL_ROOT}/Qwen3-8B-Lora"}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/train.parquet"}
export VAL_FILE=${VAL_FILE:-"${DATA_ROOT}/valid.parquet"}


# ==================== 常用训练参数 ====================
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-48}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-24}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-128}
export ACTOR_LR=${ACTOR_LR:-1e-6}
export N_RESPONSES=${N_RESPONSES:-4}
export DISTILLATION_TOP_K=${DISTILLATION_TOP_K:-32}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-8}
export SAVE_FREQ=${SAVE_FREQ:-91}
export TEST_FREQ=${TEST_FREQ:-91}


# ==================== Rollout 与显存 ====================
export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
export VAL_N_RESPONSES=${VAL_N_RESPONSES:-1}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.9}
export VAL_TOP_P=${VAL_TOP_P:-1.0}
export VAL_TOP_K=${VAL_TOP_K:--1}
export ACTOR_MAX_TOKENS=${ACTOR_MAX_TOKENS:-32768}
export VLLM_MAX_BATCHED_TOKENS=${VLLM_MAX_BATCHED_TOKENS:-98304}
export TEACHER_MAX_BATCHED_TOKENS=${TEACHER_MAX_BATCHED_TOKENS:-131072}
export VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-192}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}


# ==================== 输出与续训 ====================
export PROJECT_NAME=${PROJECT_NAME:-FinQA-LoRA-GPU-OPD}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-finqa_qwen3_1p7b_lora_1400_from_qwen3_8b_lora_5750_scheme_c_opd_n4_e8_rtxpro6000x3}
export OUTPUT_DIR=${OUTPUT_DIR:-"${TRAIN_OUTPUT_ROOT}/${EXPERIMENT_NAME}"}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${OUTPUT_DIR}/validation"}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${OUTPUT_DIR}/swanlab"}
export SWANLAB_MODE=${SWANLAB_MODE:-online}
export RESUME_MODE=${RESUME_MODE:-auto}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-4}


# ==================== 固定环境 ====================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2}
export DATA_SEED=${DATA_SEED:-42}
CACHE_DIR=${CACHE_DIR:-"${CACHE_ROOT}"}
export HF_HOME=${HF_HOME:-"${CACHE_DIR}/huggingface"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${CACHE_DIR}"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"${CACHE_DIR}/torchinductor"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"${CACHE_DIR}/triton"}
export TMPDIR=${TRAIN_TMPDIR:-"${CACHE_DIR}/tmp"}


# ==================== 启动前准备 ====================
[[ -f "${TRAIN_FILE}" ]] || { echo "找不到训练集：${TRAIN_FILE}" >&2; exit 1; }
[[ -f "${VAL_FILE}" ]] || { echo "找不到验证集：${VAL_FILE}" >&2; exit 1; }
[[ -d "${STUDENT_MODEL_PATH}" ]] || { echo "找不到 Student：${STUDENT_MODEL_PATH}" >&2; exit 1; }
[[ -d "${TEACHER_MODEL_PATH}" ]] || { echo "找不到 Teacher：${TEACHER_MODEL_PATH}" >&2; exit 1; }

mkdir -p "${OUTPUT_DIR}" "${VALIDATION_DATA_DIR}" "${SWANLAB_LOG_DIR}" "${HF_HOME}" \
    "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TMPDIR}"


# ==================== 启动 OPD 训练 ====================
exec bash "${SCRIPT_DIR}/on_policy_distillation_gpu.sh" "$@"
