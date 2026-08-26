#!/usr/bin/env bash

# VERL 0.8 native OPD command. Launch through run_OPD_*.sh.
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUSTOM_REWARD_PATH="${SCRIPT_DIR}/finqa_three_field_reward.py"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH + 1))

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESP_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation=error
    data.shuffle=True
    data.validation_shuffle=False
    data.seed=${DATA_SEED}
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=False
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.enable_activation_offload=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}
    actor_rollout_ref.actor.optim.lr_warmup_steps=${LR_WARMUP_STEPS}
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=${LR_WARMUP_STEPS_RATIO}
    actor_rollout_ref.actor.optim.min_lr_ratio=${MIN_LR_RATIO}
    actor_rollout_ref.actor.optim.num_cycles=${NUM_CYCLES}
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_epochs=${PPO_EPOCHS}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKENS}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.dtype=bfloat16
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}
    actor_rollout_ref.rollout.n=${N_RESPONSES}
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_batched_tokens=${VLLM_MAX_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N_RESPONSES}
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ROLLOUT_LOGPROB_MAX_TOKENS}
)

REWARD=(
    reward.custom_reward_function.path="${CUSTOM_REWARD_PATH}"
    reward.custom_reward_function.name=compute_score
    reward.reward_model.enable=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","swanlab"]'
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${OUTPUT_DIR}
    trainer.n_gpus_per_node=3
    trainer.nnodes=1
    trainer.resume_mode=${RESUME_MODE}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    trainer.val_before_train=True
    trainer.validation_data_dir="${VALIDATION_DATA_DIR}"
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

DISTILLATION=(
    distillation.enabled=True
    distillation.n_gpus_per_node=1
    distillation.nnodes=1
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL_PATH}"
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.dtype=bfloat16
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_BATCHED_TOKENS}
    distillation.teacher_models.teacher_model.inference.max_num_seqs=${TEACHER_MAX_NUM_SEQS}
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True
    distillation.teacher_models.teacher_model.inference.temperature=1.0
    distillation.distillation_loss.loss_mode=forward_kl_topk
    distillation.distillation_loss.topk=${DISTILLATION_TOPK}
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

if [[ ${OPD_DRY_RUN:-0} == 1 ]]; then
    printf '%s\n' "OPD dry-run passed: configuration arrays were built; training was not started."
    exit 0
fi

exec python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${DISTILLATION[@]}" \
    "$@"
