# ==================== 激活环境 ====================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate finqa-opd

# ==================== 开启训练 ====================
mkdir -p /root/train_output/logs

RUN_NAME="finqa_qwen3_1p7b_lora_1400_from_qwen3_8b_lora_5750_scheme_c_opd_n2_e12_rtxpro6000x3_$(date +%Y%m%d_%H%M%S)"
RUN_LOG="/root/train_output/logs/${RUN_NAME}.log"

nohup env \
    CUDA_VISIBLE_DEVICES=0,1,2 \
    PYTHONUNBUFFERED=1 \
    SWANLAB_MODE=online \
    RESUME_MODE=disable \
    TRAIN_BATCH_SIZE=48 \
    PPO_MINI_BATCH_SIZE=48 \
    N_RESPONSES=2 \
    VAL_N_RESPONSES=1 \
    ACTOR_MAX_TOKENS=32768 \
    VLLM_MAX_BATCHED_TOKENS=98304 \
    TEACHER_MAX_BATCHED_TOKENS=131072 \
    VLLM_MAX_NUM_SEQS=192 \
    ROLLOUT_GPU_MEMORY_UTILIZATION=0.70 \
    TEACHER_GPU_MEMORY_UTILIZATION=0.85 \
    TOTAL_EPOCHS=12 \
    EXPERIMENT_NAME="${RUN_NAME}" \
    bash /root/scripts/run_OPD_FinQA_Qwen3_1P7B_LoRA_from_Qwen3_8B_LoRA_GPU.sh \
    >"${RUN_LOG}" 2>&1 &

# ==================== 查看训练日志 ====================
tail -f "${RUN_LOG}"

# ==================== 模型合并 ====================
source /root/miniconda3/etc/profile.d/conda.sh
conda activate finqa-opd

export RUN_NAME="finqa_qwen3_1p7b_lora_1400_from_qwen3_8b_lora_5750_scheme_c_opd_n2_e12_rtxpro6000x3_20260809_064540"
# 按实际需要修改 checkpoint step。
export CKPT_STEP=728
export OUTPUT_DIR="/root/train_output/${RUN_NAME}"
export CKPT_DIR="${OUTPUT_DIR}/global_step_${CKPT_STEP}"
export MERGED_DIR="/root/merge_result/Qwen3-1.7B-Lora-OPD-0808-${CKPT_STEP}"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${CKPT_DIR}/actor" \
    --target_dir "${MERGED_DIR}"
