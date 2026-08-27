#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CUSTOM_REWARD_PATH="${SCRIPT_DIR}/finqa_reward.py"

MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH + 1))

export VLLM_USE_V1=${VLLM_USE_V1:-1}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

DATA=(
    # VERL 要求一个合法的 advantage estimator；纯 OPD 不使用其任务损失。
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="['${TRAIN_FILE}']"
    data.val_files="['${VAL_FILE}']"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESP_LENGTH}
    data.filter_overlong_prompts=True
    # 若仍遇到超长输入则直接报错，不进行静默截断。
    data.truncation='error'
    # 在训练时打乱数据顺序。
    data.shuffle=True
    data.validation_shuffle=False
    data.seed=${DATA_SEED}
    # 保留应用 chat template 前的原始对话，供教师模型等流程使用。
    data.return_raw_chat=True
    +data.apply_chat_template_kwargs.enable_thinking=False
)

MODEL=(
    actor_rollout_ref.model.path="${STUDENT_MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    # 用反向传播时重算激活换取更低显存占用。
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # 不把激活卸载到 CPU，避免额外的数据传输开销。
    actor_rollout_ref.model.enable_activation_offload=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.weight_decay=0.01
    actor_rollout_ref.actor.optim.clip_grad=1.0
    # 设置每次 actor 更新使用的 PPO mini-batch 大小。
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    # 每批 rollout 数据只用于一轮 actor 更新。
    actor_rollout_ref.actor.ppo_epochs=1
    # 按实际 token 数动态拆分 micro-batch，以提高显存利用率。
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ACTOR_MAX_TOKENS}
    actor_rollout_ref.actor.loss_agg_mode=token-mean
    # 不额外加入 actor 与参考模型之间的 KL loss。
    actor_rollout_ref.actor.use_kl_loss=False
    # 使用 torch.compile 编译 actor 计算图，以提升稳定阶段吞吐。
    actor_rollout_ref.actor.use_torch_compile=True
    # actor 参数常驻 GPU，不卸载到 CPU。
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    # 优化器状态常驻 GPU，不卸载到 CPU。
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
    actor_rollout_ref.actor.fsdp_config.dtype=bfloat16
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
    # 序列并行大小为 1，表示不启用 Ulysses 序列并行。
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
    actor_rollout_ref.rollout.max_num_seqs=${VLLM_MAX_NUM_SEQS}
    # 将长 prompt 分块执行 prefill，降低峰值显存并改善调度。
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    # 缓存并复用相同 prompt 前缀的 KV cache。
    actor_rollout_ref.rollout.enable_prefix_caching=True
    # rollout 完成后释放 vLLM cache，为 actor 更新腾出显存。
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K}
    # 每条验证样本只生成一次，避免 K=8 验证显著增加训练时间。
    actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}
    actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}
    actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N_RESPONSES}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    # 计算 rollout log probability 时也按 token 数动态拆分批次。
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    # 限制单张 GPU 一次计算 rollout log probability 的 token 总数。
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ACTOR_MAX_TOKENS}
)

# 当前实验仅用于记录 FinQA rollout 正确率；use_task_rewards=False 时不参与最终损失。
REWARD=(
    # 指定自定义 FinQA 奖励函数所在的 Python 文件。
    reward.custom_reward_function.path="${CUSTOM_REWARD_PATH}"
    # 指定从该文件中加载的奖励计算函数名。
    reward.custom_reward_function.name=compute_score
    # 不启动额外的神经网络 reward model。
    reward.reward_model.enable=False
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","swanlab"]'
    trainer.project_name=${PROJECT_NAME}
    # 设置日志平台和 checkpoint 中显示的实验名称。
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.default_local_dir=${OUTPUT_DIR}
    # 指定资源池中 2 张卡负责运行 Student，另 1 张卡独占 Teacher。
    trainer.n_gpus_per_node=2
    trainer.nnodes=1
    # 控制是否自动续训、禁用续训或从指定路径恢复。
    trainer.resume_mode=${RESUME_MODE}
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP}
    # 训练前先记录 LoRA Student 基线，以便衡量 OPD 真实增益。
    trainer.val_before_train=True
    # 保存每次单次采样验证的生成结果，用于 checkpoint 复核。
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
    # 使用 vLLM 作为教师模型的推理后端。
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.dtype=bfloat16
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=1
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${TEACHER_GPU_MEMORY_UTILIZATION}
    distillation.teacher_models.teacher_model.inference.max_model_len=${MAX_MODEL_LEN}
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens=${TEACHER_MAX_BATCHED_TOKENS}
    distillation.teacher_models.teacher_model.inference.max_num_seqs=${VLLM_MAX_NUM_SEQS}
    # 教师模型也使用分块 prefill，以降低长输入的峰值显存。
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill=True
    # 教师模型复用相同 prompt 前缀的 KV cache。
    distillation.teacher_models.teacher_model.inference.enable_prefix_caching=True
    distillation.teacher_models.teacher_model.inference.temperature=1.0
    distillation.distillation_loss.loss_mode=forward_kl_topk
    distillation.distillation_loss.topk=${DISTILLATION_TOP_K}
    # 直接反向传播蒸馏损失，不把它转成 policy-gradient reward。
    distillation.distillation_loss.use_policy_gradient=False
    # 只优化蒸馏目标，FinQA 任务奖励仅用于诊断记录。
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)


python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "${DISTILLATION[@]}" \
    "$@"
