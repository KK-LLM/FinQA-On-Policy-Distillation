# FinQA Brief–Program–Answer Teacher-only TopK32 OPD

本轮以 Qwen3-1.7B 三字段 LoRA 作为 Student 初始化模型，使用 Qwen3-8B 三字段 LoRA 作为 Teacher，继续验证 Teacher TopK32 On-Policy Distillation（OPD）能否提升 Student 的 FinQA 任务能力。

训练过程中，Student 在线生成 `Brief–Program–Answer` 响应，Teacher 在 Student 实际访问的 token 位置提供 Top-32 概率分布，再通过 Forward KL 直接更新 Student。本文中的 Teacher-only 表示 Actor Loss 只包含 Teacher 蒸馏损失，不包含 Task Reward 或 Policy Gradient；Reward 函数仍用于训练诊断和内部验证。

## 实验目标

- 验证三字段输出能否稳定接入 VERL OPD 训练链路；
- 验证纯 Teacher TopK32 Forward KL 能否在三字段 LoRA 基础上继续提升 Student；
- 同时观察 Answer 正确率、Program 正确性和 Program–Answer 一致性的变化；
- 为后续加入 FinQA Task Reward 建立无任务奖励的对照基线。

## 模型起点

| 角色 | 实际使用模型 | `avg@8` | `best@8` |
|---|---|---:|---:|
| Student | Qwen3-1.7B 三字段 LoRA checkpoint-3400 | 41.26% | 57.98% |
| Teacher | Qwen3-8B 三字段 LoRA checkpoint-15550 | 64.71% | 72.71% |

两项结果均来自 FinQA test 的 K=8 外部评测。Student 和 Teacher 使用同一套三字段输出协议、System Prompt 和 Qwen3 非思考模板。LoRA 数据构造、训练配置和模型选择见 [LoRA 实验说明](../lora/README.md)。

评估本轮 OPD 收益时，应以实际 Student 初始化模型的 41.26% `avg@8` 和 57.98% `best@8` 为基线，不将 Base Model 到三字段 LoRA 的提升计入 OPD 收益。

## OPD 数据

### 筛选口径

候选池由 6,240 条经过清洗的 FinQA 三字段训练数据组成。Student 和 Teacher 分别对每道题生成 8 次响应，再按题目 ID 对齐测试结果。

```text
S = Student 在 8 次采样中的成功次数
T = Teacher 在 8 次采样中的成功次数
```

`S` 和 `T` 的取值范围均为 0～8。单次输出只有同时满足以下条件才计为成功：

- Answer 格式正确且结果正确；
- Program 可解析、可执行且执行结果正确；
- Program 执行结果与 Answer 一致；
- Brief 满足格式与长度约束；
- 响应没有因达到生成上限而截断。

对少量缺少 `Brief:` 标签，但仍可唯一恢复为“Brief 内容行 + Program + Answer”的 Student 输出，筛选时使用固定规则恢复；其余标签或结构异常的输出不计为成功。

这一口径不只检查最终答案，还要求整条 `Brief–Program–Answer` 输出能够被解析和执行，以避免将 Teacher 偶然答对但中间结构错误的样本当作稳定教师信号。

### 三类基础数据

| 数据组 | 筛选条件 | 作用 | 唯一题目数 |
|---|---|---|---:|
| 严格核心组 | `T >= 7`、`S >= 1`、`T-S >= 2` | 保留 Teacher 稳定且明显强于 Student 的题目 | 1,978 |
| 高难度组 | `T >= 6`、`S = 0` | 保留 Student 完全未掌握但 Teacher 较稳定的题目 | 858 |
| 稳定锚点组 | `T = S = 8` | 保留两个模型都稳定掌握的区域 | 388 |
| 合计 | — | — | 3,224 |

稳定锚点组保留全部 123 条 2-step 题目和 8 条 3+ steps 题目，再使用随机种子 `42` 从 810 条 1-step 候选中抽取 257 条。这样可以保留已掌握区域，同时避免简单锚点占比过高。

### 内部验证集

内部验证集在定向重采样之前，从严格核心组、高难度组和稳定锚点组中分别抽取 5%。抽样在组内按 1-step、2-step 和 3+ steps 分层，并使用随机种子 `42` 固定结果。

| 数据组 | 1-step | 2-step | 3+ steps | 合计 |
|---|---:|---:|---:|---:|
| 严格核心组 | 62 | 32 | 4 | 98 |
| 高难度组 | 10 | 25 | 7 | 42 |
| 稳定锚点组 | 12 | 6 | 1 | 19 |
| 合计 | 84 | 63 | 12 | 159 |

该验证集来自 OPD 候选训练池，不是 FinQA 官方 valid 或 test。Train 与内部 valid 的源题目 ID 无交集，valid 内也没有重复题目。

### 定向重采样

划分内部 valid 后，Train 剩余 3,065 道唯一题目。本轮只对两类数据额外采样一次：

1. `2-step、T >= 7、S in {1,2}` 的高对比核心数据，增加 285 个实例；
2. 严格核心组的 3+ steps 数据，以及高难度组中 `T >= 7` 的 3+ steps 数据，增加 165 个实例。

`T=6、S=0` 的题目、稳定锚点和其余普通数据不重复。每个源题最多出现两次。

### 最终数据

| 数据 | 数量 |
|---|---:|
| Train 唯一题目 | 3,065 |
| 定向增加实例 | 450 |
| 最终 Train 实例 | 3,515 |
| 内部 Valid | 159 |

最终 Train 包含 1,651 条 1-step、1,512 条 2-step 和 352 条 3+ steps 实例。2-step 和高质量 3+ steps 数据得到定向增强，完成重采样后的锚点占比约为 10.50%。

本轮 VERL 实际读取的数据为 [`train_opd_scheme_a.parquet`](./data/train_opd_scheme_a.parquet) 和 [`valid_opd_scheme_a.parquet`](./data/valid_opd_scheme_a.parquet)。筛选使用的 Student 和 Teacher K=8 结果只用于决定哪些题目进入数据集；OPD 训练时，Student 仍会在线生成新的三字段响应。

## Reward 与诊断指标

本轮使用两个 Reward 相关文件：

- [`finqa_three_field_reward.py`](./scripts/finqa_three_field_reward.py) 是 VERL 通过 `compute_score` 直接调用的入口，负责解析框架传入的数据并管理 tokenizer；
- [`finqa_three_field_core.py`](./scripts/finqa_three_field_core.py) 保存三字段格式检查、Program 解析与执行、Answer 校验和分项诊断逻辑。

这样拆分可以将 VERL 调用接口与 FinQA 任务评测逻辑分开。Reward 会计算 Answer 正确性、Program 可解析性与可执行性、Program 结果正确性、Program–Answer 一致性、Brief 合规性和三字段格式。

本轮设置 `use_task_rewards=false`，因此 Reward 分数只写入训练日志并用于内部验证，不参与 Actor Loss 计算，也不会直接产生梯度。

## 训练资源

| GPU | 任务 | 资源 |
|---|---|---|
| GPU 1、2、3 | Student FSDP Actor 与 Student rollout | 3 × RTX PRO 6000 Blackwell 96 GB |
| GPU 0 | Teacher vLLM 推理 | 1 × RTX PRO 6000 Blackwell 96 GB |

Student 使用 3 张 GPU 进行全参数更新和在线采样，Teacher 固定参数并独占 1 张 GPU 提供在线 Top-32 概率分布。

## OPD 配置

以下配置以完整训练日志中的 resolved config 为准。

### 训练预算与优化器

| 参数 | 实际值 |
|---|---:|
| 训练框架 | VERL 0.8.0 |
| Train batch size | 96 |
| PPO mini-batch size | 24 |
| Student rollout 数量 | 4 |
| PPO epochs | 1 |
| Total epochs | 4 |
| 每个 epoch 的 step 数 | 36 |
| 预设总 step 数 | 144 |
| Learning rate | `1e-5` |
| LR scheduler | Cosine |
| Warmup ratio | 0.10 |
| Minimum LR ratio | 0.10 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Save frequency | 9 step |
| Validation frequency | 9 step |

### 序列与采样

| 参数 | 训练 rollout | 内部验证 |
|---|---:|---:|
| Max prompt length | 3,000 | 3,000 |
| Max response length | 192 | 192 |
| Responses per prompt | 4 | 1 |
| Temperature | 1.0 | 0.0 |
| Top-p | 1.0 | 1.0 |
| Sampling Top-k | -1 | -1 |
| Sampling | 是 | 否 |

训练 rollout 的 `Top-k=-1` 表示生成阶段不进行 Top-k 截断，与蒸馏损失使用的 Teacher TopK32 是两个不同参数。

### 蒸馏目标

| 参数 | 实际值 |
|---|---:|
| Loss mode | `forward_kl_topk` |
| Teacher Top-K | 32 |
| Policy Gradient | 否 |
| Task Reward 参与损失 | 否 |
| Actor reference KL loss | 关闭 |
| Loss max clamp | 10.0 |
| Log-prob min clamp | -10.0 |

`use_policy_gradient=false` 表示 Teacher 蒸馏损失直接反向传播，不转换为 Policy Gradient 奖励。本轮没有在启动脚本中配置 `distillation_loss_coef`。VERL resolved config 仍可能显示框架 schema 默认的 `distillation_loss_coef=1.0`，但该字段只在 `use_task_rewards=true` 的联合 Loss 分支中生效，不参与本轮 Actor Loss。

### 显存与吞吐配置

| 参数 | 实际值 |
|---|---:|
| Actor max tokens per GPU | 49,152 |
| Rollout log-prob max tokens per GPU | 98,304 |
| Student vLLM max batched tokens | 131,072 |
| Teacher vLLM max batched tokens | 163,840 |
| Student vLLM max sequences | 128 |
| Teacher vLLM max sequences | 224 |
| Student rollout GPU memory utilization | 0.70 |
| Teacher GPU memory utilization | 0.85 |
| Student / Teacher precision | BF16 |

## 启动与模型合并

本轮实际调用链为：

```text
test.sh
  → run_OPD_FinQA_1P7B_from_8B_scheme_a.sh
    → on_policy_distillation_gpu.sh
```

- [`test.sh`](./scripts/test.sh) 负责环境激活、GPU 分配、后台启动、状态查看、checkpoint 恢复与模型合并；该文件是按区块执行的容器操作手册，不应一次性执行整个文件；
- [`run_OPD_FinQA_1P7B_from_8B_scheme_a.sh`](./scripts/run_OPD_FinQA_1P7B_from_8B_scheme_a.sh) 负责模型、数据、训练参数、缓存和输出路径；
- [`on_policy_distillation_gpu.sh`](./scripts/on_policy_distillation_gpu.sh) 将环境变量转换为 VERL/Hydra 参数并启动 OPD 训练。

训练产生的 Actor checkpoint 使用 `verl.model_merger` 合并为完整 Hugging Face 模型，再使用实验根目录下的 [`finqa_three_field_eval.py`](../finqa_three_field_eval.py) 和 [`prompt.py`](../prompt.py) 进行外部 K=8 评测。该评测实现与三字段 LoRA 阶段保持一致。

## 训练过程

本轮按预设的 4 epoch 训练预算运行至 step 144，共完成 144 个 outer steps。训练主体未因 OOM、NaN 或梯度发散中断。

启动时的 resolved config 确认 `use_task_rewards=false`、`use_policy_gradient=false` 和 `forward_kl_topk/topk=32`，因此训练更新采用纯 Teacher TopK32 蒸馏路径。

内部 valid 只包含 159 道题，并且每道题只进行 1 次贪心生成。该结果只用于检查训练链路和三字段输出状态，不替代 1,147 题 K=8 外部测试，也不作为本轮结论的主要依据。

## 外部测试结果

外部评测与三字段 LoRA 使用同一评测脚本、同一 System Prompt 和相同的 K=8 测试口径。FinQA test 共 1,147 道题，每道题采样 8 次，每个 checkpoint 共生成 9,176 个响应。三组评测的请求错误均为 0。`avg@8` 表示全部生成结果的正确比例，`best@8` 表示至少有 1 次回答正确的题目比例。

| 模型 | `avg@8` | `best@8` |
|---|---:|---:|
| OPD-81 | 43.36% | 60.33% |
| OPD-117 | **43.70%** | 60.94% |
| **OPD-144** | **43.70%** | **61.55%** |

后两个 checkpoint 的 `avg@8` 均达到本轮最高值 43.70%；其中最后一个 checkpoint 的 `best@8` 进一步提高至 61.55%，也是本轮最高结果。因此，本轮 Teacher-only OPD 采用最后一个 checkpoint 作为代表模型。

### 与上一轮 OPD 及本轮初始化模型对比

| 指标 | 上一轮 Answer-only OPD | 三字段 LoRA Student | 本轮 Teacher-only OPD | 相比上一轮 OPD | 相比三字段 LoRA |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `avg@8` | 23.89% | 41.26% | **43.70%** | **+19.81 个百分点** | **+2.44 个百分点** |
| `best@8` | 36.44% | 57.98% | **61.55%** | **+25.11 个百分点** | **+3.57 个百分点** |

相比上一轮 Answer-only OPD，本轮 Teacher-only OPD 的 `avg@8` 提高了 19.81 个百分点，`best@8` 提高了 25.11 个百分点。这部分差异反映的是三字段数据、更强 LoRA 初始化及本轮 OPD 配置共同形成的整体提升。

在三字段 LoRA Student 的基础上，本轮 Teacher-only OPD 的 `avg@8` 和 `best@8` 又分别提高了 2.44 和 3.57 个百分点，说明 Teacher-only OPD 在更强的 Student 初始化上仍然带来了进一步增益。

## 结果分析

相较于实际 Student 初始化模型，本轮 Teacher-only OPD 的 `avg@8` 提升 2.44 个百分点，`best@8` 提升 3.57 个百分点。这说明纯 Teacher TopK32 分布监督本身能够在三字段 LoRA 基础上产生可测量的任务收益。

但从 step 117 到本轮训练终点，`avg@8` 已经完全相同，`best@8` 仅提高 0.61 个百分点。当前结果不支持继续增加相同配置的训练 epoch 能够形成新的能力台阶。

因此，本轮保留为三字段 Teacher-only TopK32 OPD 基线：它验证了更长的结构化响应可以稳定完成在线蒸馏，也证明不依赖 Task Reward 的 Teacher 分布信号能够进一步提高 FinQA 外部测试结果。

## 下一轮优化方向

本轮已经验证纯 Teacher TopK32 分布监督能够带来收益，但后期 Answer 平均正确率进入平台。下一轮将保持 Student、Teacher、三字段数据、TopK32 和主要训练配置不变，在 Teacher 蒸馏目标上加入 FinQA Task Reward，观察任务级反馈能否将已经改善的 Program 分布进一步转化为 Answer 正确率。

## 文件说明

```text
opd/
├── README.md
├── data/
│   ├── train_opd_scheme_a.parquet
│   └── valid_opd_scheme_a.parquet
└── scripts/
    ├── finqa_three_field_core.py
    ├── finqa_three_field_reward.py
    ├── on_policy_distillation_gpu.sh
    ├── run_OPD_FinQA_1P7B_from_8B_scheme_a.sh
    └── test.sh
```

- `data/` 保存本轮 VERL 实际读取的训练集和内部验证集；
- `scripts/` 保存实际训练入口、VERL OPD 命令和三字段 Reward 逻辑；
- LoRA 与 OPD 共用的 [`finqa_three_field_eval.py`](../finqa_three_field_eval.py) 和 [`prompt.py`](../prompt.py) 位于上一级实验目录，用于保持两阶段的外部评测口径一致；
- 本 README 汇总数据筛选、训练配置、外部测试结果和后续优化依据。

[返回项目首页](../../README.md)
