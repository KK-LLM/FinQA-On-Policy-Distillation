#!/usr/bin/env bash

# Four-GPU launcher: 3× RTX PRO 6000 96G Student + 1× Teacher.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_ROOT=${MODEL_ROOT:-/root/model}
DATA_ROOT=${DATA_ROOT:-/root/data}
TRAIN_OUTPUT_ROOT=${TRAIN_OUTPUT_ROOT:-/root/train_output}
CACHE_ROOT=${CACHE_ROOT:-/root/autodl-tmp/opd_finqa_bpa/cache}

# Complete merged Hugging Face models, not adapter-only checkpoint directories.
export STUDENT_MODEL_PATH=${STUDENT_MODEL_PATH:-"${MODEL_ROOT}/Qwen3-1.7B-LoRA-BPA-0813-ckp-3400"}
export TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-"${MODEL_ROOT}/Qwen3-8B-LoRA-BPA-0813-ckp-15550"}
export FINQA_TOKENIZER_PATH=${FINQA_TOKENIZER_PATH:-"${STUDENT_MODEL_PATH}"}
export TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/train_opd_scheme_a.parquet"}
export VAL_FILE=${VAL_FILE:-"${DATA_ROOT}/valid_opd_scheme_a.parquet"}

export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-24}
export PPO_EPOCHS=${PPO_EPOCHS:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3000}
export MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-192}
export ACTOR_LR=${ACTOR_LR:-1e-5}
export LR_SCHEDULER_TYPE=${LR_SCHEDULER_TYPE:-cosine}
export LR_WARMUP_STEPS=${LR_WARMUP_STEPS:--1}
export LR_WARMUP_STEPS_RATIO=${LR_WARMUP_STEPS_RATIO:-0.10}
export MIN_LR_RATIO=${MIN_LR_RATIO:-0.10}
export NUM_CYCLES=${NUM_CYCLES:-0.5}
export N_RESPONSES=${N_RESPONSES:-4}
export TOTAL_EPOCHS=${TOTAL_EPOCHS:-4}
# floor(3515 / 96) = 36 outer steps/epoch; save and validate every 9 steps.
export SAVE_FREQ=${SAVE_FREQ:-9}
export TEST_FREQ=${TEST_FREQ:-9}

export ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1.0}
export ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1.0}
export ROLLOUT_TOP_K=${ROLLOUT_TOP_K:--1}
export VAL_N_RESPONSES=${VAL_N_RESPONSES:-1}
export VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
export VAL_TOP_P=${VAL_TOP_P:-1.0}
export VAL_TOP_K=${VAL_TOP_K:--1}
export VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-False}
export ACTOR_MAX_TOKENS=${ACTOR_MAX_TOKENS:-49152}
export ROLLOUT_LOGPROB_MAX_TOKENS=${ROLLOUT_LOGPROB_MAX_TOKENS:-98304}
export VLLM_MAX_BATCHED_TOKENS=${VLLM_MAX_BATCHED_TOKENS:-131072}
export TEACHER_MAX_BATCHED_TOKENS=${TEACHER_MAX_BATCHED_TOKENS:-163840}
export ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-128}
export TEACHER_MAX_NUM_SEQS=${TEACHER_MAX_NUM_SEQS:-224}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.70}
export TEACHER_GPU_MEMORY_UTILIZATION=${TEACHER_GPU_MEMORY_UTILIZATION:-0.85}
export DISTILLATION_TOPK=${DISTILLATION_TOPK:-16}
export DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}

export PROJECT_NAME=${PROJECT_NAME:-FinQA-BPA-OPD}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-finqa_bpa_opd_reward_coef1p0_topk16}
export OUTPUT_DIR=${OUTPUT_DIR:-"${TRAIN_OUTPUT_ROOT}/${EXPERIMENT_NAME}"}
export VALIDATION_DATA_DIR=${VALIDATION_DATA_DIR:-"${OUTPUT_DIR}/validation"}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-"${OUTPUT_DIR}/swanlab"}
export SWANLAB_MODE=${SWANLAB_MODE:-online}
export RESUME_MODE=${RESUME_MODE:-auto}
export MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-10}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export DATA_SEED=${DATA_SEED:-42}
CACHE_DIR=${CACHE_DIR:-"${CACHE_ROOT}"}
export HF_HOME=${HF_HOME:-"${CACHE_DIR}/huggingface"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"${CACHE_DIR}"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"${CACHE_DIR}/torchinductor"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"${CACHE_DIR}/triton"}
export TMPDIR=${TRAIN_TMPDIR:-"${CACHE_DIR}/tmp"}

require_full_model() {
    local model_path=$1
    local label=$2
    [[ -d ${model_path} ]] || { echo "Missing ${label}: ${model_path}" >&2; exit 1; }
    [[ -f ${model_path}/config.json ]] || {
        echo "${label} has no config.json: ${model_path}" >&2
        exit 1
    }
    [[ -f ${model_path}/tokenizer_config.json ]] || {
        echo "${label} has no tokenizer_config.json: ${model_path}" >&2
        exit 1
    }
    if ! find "${model_path}" -maxdepth 1 -type f \
        \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) -print -quit | grep -q .; then
        echo "${label} has no full model weights and may be adapter-only: ${model_path}" >&2
        exit 1
    fi
}

[[ -f ${TRAIN_FILE} ]] || { echo "Missing train Parquet: ${TRAIN_FILE}" >&2; exit 1; }
[[ -f ${VAL_FILE} ]] || { echo "Missing valid Parquet: ${VAL_FILE}" >&2; exit 1; }
[[ -f ${SCRIPT_DIR}/finqa_three_field_reward.py ]] || {
    echo "Missing custom reward beside launcher." >&2
    exit 1
}
require_full_model "${STUDENT_MODEL_PATH}" "Student model"
require_full_model "${TEACHER_MODEL_PATH}" "Teacher model"

IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
(( ${#visible_gpus[@]} == 4 )) || {
    echo "CUDA_VISIBLE_DEVICES must expose exactly four GPUs; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
}
(( TRAIN_BATCH_SIZE >= PPO_MINI_BATCH_SIZE && TRAIN_BATCH_SIZE % PPO_MINI_BATCH_SIZE == 0 )) || {
    echo "TRAIN_BATCH_SIZE must be divisible by PPO_MINI_BATCH_SIZE." >&2
    exit 1
}
(( (PPO_MINI_BATCH_SIZE * N_RESPONSES) % 3 == 0 )) || {
    echo "PPO_MINI_BATCH_SIZE × N_RESPONSES must be divisible by three Student GPUs." >&2
    exit 1
}
(( (TRAIN_BATCH_SIZE * N_RESPONSES) % 3 == 0 )) || {
    echo "TRAIN_BATCH_SIZE × N_RESPONSES must be divisible by three Student GPUs." >&2
    exit 1
}
(( N_RESPONSES > 1 && VAL_N_RESPONSES > 0 && PPO_EPOCHS > 0 && TOTAL_EPOCHS > 0 )) || {
    echo "Rollout samples, validation samples, PPO epochs, and total epochs must be positive." >&2
    exit 1
}
(( SAVE_FREQ > 0 && TEST_FREQ > 0 && DISTILLATION_TOPK > 0 )) || {
    echo "Save/test frequencies and distillation Top-K must be positive." >&2
    exit 1
}
[[ ${LR_SCHEDULER_TYPE} == cosine ]] || {
    echo "This v2 package requires LR_SCHEDULER_TYPE=cosine." >&2
    exit 1
}
[[ ${VAL_DO_SAMPLE} == False ]] || {
    echo "This v2 package requires deterministic validation with VAL_DO_SAMPLE=False." >&2
    exit 1
}
(( ACTOR_MAX_TOKENS >= 2 * (MAX_PROMPT_LENGTH + MAX_RESP_LENGTH + 1) )) || {
    echo "Actor token budget must be at least twice the max model length." >&2
    exit 1
}
(( ROLLOUT_LOGPROB_MAX_TOKENS >= ACTOR_MAX_TOKENS )) || {
    echo "Forward-only rollout log-prob budget must not be smaller than the Actor budget." >&2
    exit 1
}
(( VLLM_MAX_BATCHED_TOKENS >= MAX_PROMPT_LENGTH + MAX_RESP_LENGTH + 1 )) || {
    echo "Student vLLM token budget is smaller than max model length." >&2
    exit 1
}
(( TEACHER_MAX_BATCHED_TOKENS >= MAX_PROMPT_LENGTH + MAX_RESP_LENGTH + 1 )) || {
    echo "Teacher vLLM token budget is smaller than max model length." >&2
    exit 1
}

mkdir -p "${OUTPUT_DIR}" "${VALIDATION_DATA_DIR}" "${SWANLAB_LOG_DIR}" \
    "${HF_HOME}" "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

printf '%s\n' \
    "Student: ${STUDENT_MODEL_PATH}" \
    "Teacher: ${TEACHER_MODEL_PATH}" \
    "Train/valid: ${TRAIN_FILE} | ${VAL_FILE}" \
    "GPU split: 3 Student + 1 Teacher (${CUDA_VISIBLE_DEVICES})" \
    "Batch: train/${TRAIN_BATCH_SIZE}, PPO mini/${PPO_MINI_BATCH_SIZE}, PPO epochs/${PPO_EPOCHS}, rollout n/${N_RESPONSES}" \
    "Token batches: actor/${ACTOR_MAX_TOKENS}, rollout log-prob/${ROLLOUT_LOGPROB_MAX_TOKENS}" \
    "Objective: forward_kl_topk/${DISTILLATION_TOPK} + task reward, coef/${DISTILLATION_LOSS_COEF}" \
    "LR: peak/${ACTOR_LR}, ${LR_SCHEDULER_TYPE}, warmup ratio/${LR_WARMUP_STEPS_RATIO}, min ratio/${MIN_LR_RATIO}" \
    "Schedule: ${TOTAL_EPOCHS} epochs, save/${SAVE_FREQ}, validate/${TEST_FREQ}" \
    "Output: ${OUTPUT_DIR}"

exec bash "${SCRIPT_DIR}/on_policy_distillation_gpu.sh" "$@"
