# FinQA Answer-only Teacher TopK32 OPD Baseline

Answer-only LoRA 已经显著提高 Qwen3-1.7B 和 Qwen3-8B 的 FinQA 基础能力。本轮以 Qwen3-1.7B LoRA checkpoint-1400 初始化 Student，使用 Qwen3-8B LoRA checkpoint-5750 作为 Teacher，继续验证 Teacher-token TopK32 On-Policy Distillation（OPD）能否在 LoRA 基础上提升 Student。

训练过程中，Student 在线生成 Answer-only 响应，Teacher 在 Student 访问到的 token 位置提供 Top-32 概率分布，再通过 Forward KL 直接更新 Student。FinQA 任务奖励只用于记录训练期间的答案正确率，不参与参数更新。

## 实验目标

- 验证 VERL 原生 OPD 能否稳定完成 Qwen3-8B LoRA 到 Qwen3-1.7B LoRA 的在线蒸馏；
- 验证基于 Teacher–Student 能力差筛选的 FinQA 数据能否用于 OPD 训练；
- 观察纯 TopK32 Forward KL 在 Answer-only 输出上的训练动态与外部测试收益；
- 为后续调整蒸馏内容和训练目标建立可复现的基线。

## 模型与数据

### 模型起点

| 角色 | 模型 | `avg@8` | `best@8` |
|---|---|---:|---:|
| Student | Qwen3-1.7B LoRA checkpoint-1400 | 21.17% | 34.44% |
| Teacher | Qwen3-8B LoRA checkpoint-5750 | 59.20% | 63.73% |

两项结果均来自 FinQA test 的 K=8 外部评测。LoRA 数据、训练配置和 checkpoint 选择见 [LoRA 实验说明](../lora/README.md)。

### OPD 数据

原始 FinQA train 共有 6,251 条。本轮没有直接使用全部训练数据，而是根据 Teacher 和 Student 在训练集上的 K=8 测试结果筛选样本：两个模型对每道题分别采样 8 次，再按照题目 ID 对齐结果。

```text
T = Teacher 在 8 次采样中的正确次数
S = Student 在 8 次采样中的正确次数
```

`T` 和 `S` 的取值范围均为 0～8。筛选时没有直接使用 `best@8`，因为它只能表示一道题是否至少答对过一次，无法区分偶然答对和稳定答对。比较每道题的实际正确次数，可以同时衡量 Teacher 的稳定性以及 Teacher 与 Student 之间的能力差。

这套筛选结果与本轮使用的 Teacher checkpoint-5750 和 Student checkpoint-1400 直接绑定。更换任一模型后，需要重新生成 K=8 结果并重新筛选数据。

#### 核心迁移数据

核心数据的筛选条件为：

```text
T > S 且 T >= 5
```

- `T > S`：只保留 Teacher 明确强于 Student 的题目；
- `T >= 5`：Teacher 至少在 8 次采样中答对 5 次，避免将仅偶然答对的题目纳入纯 KL 蒸馏。

按照这项规则，共筛选出 3,498 条核心迁移数据。

#### 共同掌握锚点

训练集同时加入少量 Teacher 和 Student 均稳定答对的题目：

```text
T = S = 8
```

符合条件的锚点候选共有 1,786 条。锚点用于保留 Student 已经掌握区域的模型行为，但数量过多会稀释核心迁移数据，因此将锚点限制在最终训练集的 20% 以内：

```text
anchors / (core + anchors) <= 0.20
```

核心迁移数据共有 3,498 条，因此最多保留 `floor(3498 / 4) = 874` 条锚点。抽取时先对候选题目 ID 排序，再使用随机种子 `42` 固定抽取；只要输入结果不变，锚点选择即可稳定复现。

#### 最终数据组成

| 数据类型 | 筛选方式 | 数量 |
|---|---|---:|
| 核心迁移数据 | `T > S` 且 `T >= 5` | 3,498 |
| 共同掌握锚点 | 从 `T = S = 8` 的候选中固定抽取 | 874 |
| 最终训练集 | 核心迁移数据与锚点合并 | 4,372 |
| 验证集 | 保留完整 FinQA valid，不按能力差筛选 | 883 |

锚点在最终训练集中的占比为 19.9909%。入选训练集上的 K=8 统计如下：

| 统计项 | 结果 |
|---|---:|
| Teacher `avg@8` | 0.9923 |
| Student `avg@8` | 0.3510 |
| 平均 `T-S` | 5.1306 次 |

筛选结果仍按原始 FinQA train 的顺序保存，训练阶段再由 VERL 使用 `data.shuffle=True` 和 `data.seed=42` 打乱。

#### 未进入训练集的数据

以下题目没有进入本轮 OPD 训练集：

- `T < S`：Teacher 弱于 Student；
- `T > S` 但 `T < 5`：Teacher 没有达到多数正确；
- `T = S = 0`：Teacher 和 Student 均未答对；
- 除 `T = S = 8` 外的其他 `T = S` 题目；
- 超出锚点数量限制、未被固定抽中的 `T = S = 8` 题目。

这套规则只决定哪些题目进入训练集，Teacher 和 Student 的 K=8 测试结果没有作为固定答案写入数据。OPD 训练过程中，Student 仍然在线生成响应，Teacher 再在 Student 实际访问到的 token 位置提供 Top-32 概率分布。

最终训练集对应 [`data/train.parquet`](./data/train.parquet)，验证集对应 [`data/valid.parquet`](./data/valid.parquet)。

## 训练资源

| GPU | 任务 | 资源 |
|---|---|---|
| GPU 0、1 | Student FSDP Actor 与 Student rollout | 2 × RTX PRO 6000 Blackwell 96 GB |
| GPU 2 | Teacher vLLM 推理 | 1 × RTX PRO 6000 Blackwell 96 GB |

Student 使用两张 GPU 进行全参数更新，Teacher 固定参数并独占一张 GPU 提供在线概率分布。Student 与 Teacher 均使用 Qwen3 tokenizer 和词表。

## OPD 配置

以下配置以完整训练日志中的 resolved config 为准。

### 训练预算与优化器

| 参数 | 实际值 |
|---|---:|
| 训练框架 | VERL 0.8.0 |
| Train batch size | 48 |
| PPO mini-batch size | 48 |
| Student rollout 数量 | 2 |
| PPO epochs | 1 |
| Total epochs | 12 |
| 每个 epoch 的 step 数 | 91 |
| 总 step 数 | 1,092 |
| Learning rate | `1e-6` |
| LR scheduler | Constant |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Save frequency | 91 step |
| Validation frequency | 91 step |

### 序列与采样

| 参数 | 训练 rollout | 内部验证 |
|---|---:|---:|
| Max prompt length | 4,096 | 4,096 |
| Max response length | 128 | 128 |
| Responses per prompt | 2 | 1 |
| Temperature | 1.0 | 0.9 |
| Top-p | 1.0 | 1.0 |
| Sampling Top-k | -1 | -1 |

训练 rollout 的 `Top-k=-1` 表示生成阶段不进行 Top-k 截断，与蒸馏损失使用的 Teacher TopK32 是两个不同参数。

### 蒸馏目标

| 参数 | 实际值 |
|---|---:|
| Loss mode | `forward_kl_topk` |
| Teacher Top-K | 32 |
| Distillation coefficient | 1.0 |
| Policy Gradient | 否 |
| Task Reward 参与损失 | 否 |
| Actor reference KL loss | 关闭 |
| Loss max clamp | 10.0 |
| Log-prob min clamp | -10.0 |

`use_policy_gradient=false` 表示蒸馏损失直接反向传播，不转换为 Policy Gradient 奖励。

[`finqa_reward.py`](./scripts/finqa_reward.py) 用于判断 Student rollout 的最终答案是否正确，并计算训练日志中的 Train FinQA score 和内部验证 Reward。本轮设置 `use_task_rewards=false`，因此这些分数只用于观察训练效果，不参与 Loss 计算，也不会直接产生梯度。保留该文件是为了完整复现本轮训练中的 FinQA 判分和诊断指标。

### 显存与吞吐配置

| 参数 | 实际值 |
|---|---:|
| Actor max tokens per GPU | 32,768 |
| Student vLLM max batched tokens | 98,304 |
| Teacher vLLM max batched tokens | 131,072 |
| vLLM max sequences | 192 |
| Student rollout GPU memory utilization | 0.70 |
| Teacher GPU memory utilization | 0.85 |
| Student / Teacher precision | BF16 |

## 启动与模型合并

本轮实际调用链为：

```text
test.sh
  → run_OPD_FinQA_Qwen3_1P7B_LoRA_from_Qwen3_8B_LoRA_GPU.sh
    → on_policy_distillation_gpu.sh
```

- [`test.sh`](./scripts/test.sh) 设置本轮覆盖参数，启动后台训练，并提供日志查看和 GPU checkpoint 合并命令；
- [`run_OPD_FinQA_Qwen3_1P7B_LoRA_from_Qwen3_8B_LoRA_GPU.sh`](./scripts/run_OPD_FinQA_Qwen3_1P7B_LoRA_from_Qwen3_8B_LoRA_GPU.sh) 负责模型、数据、缓存和输出路径，以及路径检查与运行环境准备；
- [`on_policy_distillation_gpu.sh`](./scripts/on_policy_distillation_gpu.sh) 将环境变量转换为 VERL/Hydra 参数并启动训练。

原始容器中，脚本放置在 `/root/scripts/`。启动训练：

```bash
bash /root/scripts/test.sh
```

训练结束后，在 `test.sh` 的模型合并区块中设置 `RUN_NAME` 和 `CKPT_STEP`，再使用 VERL FSDP merger 将对应 Actor checkpoint 合并为完整 Hugging Face 模型。外部测试使用的 OPD-910、OPD-1001 和 OPD-1092 均经过该合并流程。

合并后的 OPD 模型与 Answer-only LoRA 模型统一使用实验根目录下的 [`finqa_answer_only_eval.py`](../finqa_answer_only_eval.py) 进行外部评测。

## 训练过程

训练完整执行至 `1092/1092` step。

| 训练指标 | Epoch 1 | Epoch 12 | 变化 |
|---|---:|---:|---|
| Distillation loss | 0.7850 | 0.6644 | 下降 15.36% |
| Grad norm | 37.4341 | 20.1271 | 整体下降 |
| Student entropy | 0.7774 | 0.9242 | 输出分布变得更分散 |
| Teacher–Student TopK overlap | 0.3828 | 0.3827 | 基本不变 |
| Train FinQA score | 0.3151 | 0.2981 | 未随 Loss 同步提升 |

Distillation loss 在前四个 epoch 下降较快，之后逐渐进入平台期。与此同时，Teacher–Student TopK overlap 始终维持在约 0.383，训练 rollout 的 FinQA score 也没有形成持续上升趋势。这说明 Student 对 Teacher 局部概率分布的拟合在改善，但这种变化没有稳定转换为更多正确答案。

本轮平均 response 长度约为 7.44 tokens。内部验证每道题只采样一次，`reward mean@1` 波动较大，因此只用于训练状态观察和 checkpoint 初筛，最终结论以独立的 K=8 外部评测为准。

## 外部测试结果

外部评测与 Answer-only LoRA 使用同一评测脚本、同一 Answer-only System Prompt、相同的 `temperature=0.5` 和 K=8 测试口径。FinQA test 共 1,147 道题，每道题采样 8 次，共生成 9,176 个回答。`avg@8` 表示全部生成的正确比例，`best@8` 表示至少有一次回答正确的问题比例。

| 模型 | 与本轮 OPD 的关系 | `avg@8` | `best@8` |
|---|---|---:|---:|
| Qwen3-1.7B LoRA checkpoint-1400 | 实际 Student 初始化模型 | 21.17% | 34.44% |
| Qwen3-1.7B LoRA checkpoint-800 | 同口径 LoRA checkpoint 参考 | 20.73% | 35.92% |
| OPD-910 | OPD checkpoint | 23.61% | **36.79%** |
| OPD-1001 | OPD checkpoint | 23.56% | 36.62% |
| OPD-1092 | OPD 最终 checkpoint | **23.89%** | 36.44% |

三个 OPD checkpoint 的 `avg@8` 均在 23.56%–23.89% 之间，`best@8` 均在 36.44%–36.79% 之间。OPD-1092 取得最高 `avg@8`，OPD-910 取得最高 `best@8`。

### 相对实际初始化模型

| OPD checkpoint | `avg@8` 绝对提升 | `best@8` 绝对提升 |
|---|---:|---:|
| OPD-910 | +2.44 个百分点 | +2.35 个百分点 |
| OPD-1001 | +2.39 个百分点 | +2.18 个百分点 |
| OPD-1092 | **+2.72 个百分点** | +2.00 个百分点 |

相较于实际用于初始化 Student 的 checkpoint-1400，三个 OPD checkpoint 均取得提升，说明本轮 OPD 训练确实产生了可测量的任务收益。

### 相对 LoRA checkpoint-800

checkpoint-800 没有参与本轮 OPD 训练，只作为同口径 LoRA 参考。OPD-1092 相比 checkpoint-800 的 `avg@8` 提升 3.16 个百分点，`best@8` 提升 0.52 个百分点；若采用本轮最高 `best@8` 的 OPD-910，则提升 0.87 个百分点。

这组比较表明，OPD 对平均生成正确率的改善较为明确，但对问题覆盖率的进一步提升较小。

## 结果分析

本轮训练完成了从数据筛选、Student 在线 rollout、Teacher TopK32 分布计算、Forward KL 更新到 checkpoint 合并和外部测试的完整链路。相较于 checkpoint-1400，OPD-1092 的 `avg@8` 提升 2.72 个百分点，OPD-910 的 `best@8` 提升 2.35 个百分点，因此这轮训练并非无效。

但从训练动态看，Distillation loss 的持续下降没有伴随 TopK overlap、Train FinQA score 和内部验证结果的同步改善；从外部测试看，训练后期三个 checkpoint 的差异也很小。继续单纯增加训练 epoch 缺少有效依据。

因此，本轮被保留为“有效但收益有限的失败基线”：它证明了当前 OPD 工程链路和纯 TopK32 Forward KL 能够工作，也说明仅依靠 Answer-only token 分布蒸馏，很难充分释放 Teacher 与 Student 之间较大的能力差。

上述结果反映的是本轮完整配置的整体表现，不能将收益有限严格归因于某一个超参数。Student 初始化、Answer-only 输出、数据难度结构、Teacher TopK32 分布和无任务奖励的纯蒸馏目标在本轮同时存在，单项因素的实际影响仍需要后续对照实验确认。

## 本轮反思与下一轮优化方向

Answer-only 输出适合统一判分，却压缩了 Teacher 能够传递的信息。本轮平均 response 只有约 7.44 tokens，蒸馏位置主要集中在最终数字、符号和结束标记上。即使 Teacher 在 FinQA 上明显强于 Student，当前输出也没有显式呈现表格定位、公式选择、运算顺序和百分比换算等中间过程。

本轮数据还集中保留了 Teacher 稳定正确、Student 明显较弱的题目。核心迁移数据中有 2,072 条为 `S=0`，占 3,498 条核心数据的 59.23%。这些题具有较大的能力迁移空间，但当 Student 在 Answer-only 轨迹中生成错误数字前缀后，Teacher 只能沿着 Student 已经访问到的状态提供后续 token 分布，纠错空间仍然有限。

结合“Distillation loss 下降而任务正确率提升有限”的现象，下一轮优先调整蒸馏内容，而不是继续简单增加 epoch。具体方向是将输出从单一 Answer 扩展为 `Brief–Program–Answer` 三字段：

- `Brief` 提炼与问题直接相关的财务信息和计算关系；
- `Program` 显式表示可执行或可检查的运算过程；
- `Answer` 保留归一化后的最终答案。

Student 和 Teacher 需要先通过三字段 LoRA 学习相同的输出结构，再进行 OPD。下一轮首先保留纯分布蒸馏的基本训练框架，重点观察更长、更具任务信息的响应能否改善蒸馏效果。Answer-only 信息不足是基于本轮训练现象提出的分析判断，三字段输出的实际收益仍需通过后续实验验证。

## 文件说明

```text
opd/
├── README.md
├── data/
│   ├── train.parquet
│   └── valid.parquet
└── scripts/
    ├── test.sh
    ├── run_OPD_FinQA_Qwen3_1P7B_LoRA_from_Qwen3_8B_LoRA_GPU.sh
    ├── on_policy_distillation_gpu.sh
    └── finqa_reward.py
```

- `data/` 保存本轮 VERL 实际读取的训练集和验证集；
- `scripts/` 保存真实训练入口、启动器和 VERL OPD 命令，其中 `finqa_reward.py` 负责 FinQA 答案判分和训练诊断，不参与本轮参数优化；
- LoRA 与 OPD 共用的 [`finqa_answer_only_eval.py`](../finqa_answer_only_eval.py) 位于上一级实验目录，用于保持两阶段的 Answer-only 外部评测口径一致；
- 本 README 汇总实际训练配置、训练过程、外部测试结果和下一轮优化依据。

[返回项目首页](../../../README.md)
