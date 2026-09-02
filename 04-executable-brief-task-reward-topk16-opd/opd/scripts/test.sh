#!/usr/bin/env bash

# ==================== 激活环境 ====================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate finqa-opd


# ==================== 首次开启训练 ====================
mkdir -p /root/train_output/logs

RUN_NAME="finqa_c_qwen3_1p7b_7920_from_qwen3_8b_14800_scheme_a_opd_topk16_taskreward_coef1p0_lr1e5_b96_m24_ppo1_n4_e4_$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="/root/train_output/${RUN_NAME}"
RUN_LOG="/root/train_output/logs/${RUN_NAME}.log"
TRAIN_PID_FILE="${OUTPUT_DIR}/train.pid"
mkdir -p "${OUTPUT_DIR}"

# 物理 GPU 1/2/3 运行 Student，物理 GPU 0 运行 Teacher。
nohup setsid env \
    CUDA_VISIBLE_DEVICES=1,2,3,0 \
    PYTHONUNBUFFERED=1 \
    SWANLAB_MODE=online \
    RESUME_MODE=disable \
    EXPERIMENT_NAME="${RUN_NAME}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    bash /root/scripts/run_OPD_FinQA_C_1P7B_from_8B_topk16_coef1p0.sh \
    >"${RUN_LOG}" 2>&1 &

TRAIN_PID=$!
printf '%s\n' "${TRAIN_PID}" > "${TRAIN_PID_FILE}"


# ==================== 查看训练日志 ====================
tail -f "${RUN_LOG}"


# ==================== 模型合并 ====================
# 每 9 step 保存一次；先从与上一轮可比的 step 72 开始合并测试。
source /root/miniconda3/etc/profile.d/conda.sh
conda activate finqa-opd

RUN_NAME=""
CKPT_STEP=72
OUTPUT_DIR="/root/train_output/${RUN_NAME}"
CKPT_DIR="${OUTPUT_DIR}/global_step_${CKPT_STEP}"
MERGED_DIR="/root/merge_result/${RUN_NAME}-global_step_${CKPT_STEP}"

mkdir -p /root/merge_result

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}/actor" \
    --target_dir "${MERGED_DIR}"
