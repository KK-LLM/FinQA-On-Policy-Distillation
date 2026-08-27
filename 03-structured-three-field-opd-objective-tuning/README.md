# 三字段数据下的 OPD 目标函数与蒸馏参数优化

上一轮 Teacher-only TopK32 OPD 证明了 Teacher 分布监督能够继续提升三字段 LoRA Student，但外部测试结果在训练后期逐渐进入平台。本轮保留相同的模型、数据和主要训练配置，先加入 Task Reward，再依次调整蒸馏系数和 Teacher TopK，继续优化 FinQA 正确率与训练效率。

## 优化起点

[实验 2：三字段 Teacher-only TopK32 OPD](../02-structured-three-field-teacher-only-opd/opd/README.md) 是本轮优化的起点。该实验将三字段 LoRA Student 的 `avg@8` 从 41.26% 提高到 43.70%，说明纯 Teacher TopK32 分布监督能够产生有效增益；但后期 `avg@8` 不再继续增长，因此下一步不再单纯延长相同配置的训练周期，而是开始调整训练目标。

| 角色 | 模型 | 实验 2 外部测试 `avg@8` | 实验 2 外部测试 `best@8` |
|:---|:---|---:|---:|
| Student | Qwen3-1.7B 三字段 LoRA checkpoint-3400 | 41.26% | 57.98% |
| Teacher | Qwen3-8B 三字段 LoRA checkpoint-15550 | 64.71% | 72.71% |

三轮 OPD 均使用 Qwen3-1.7B 三字段 LoRA checkpoint-3400 初始化 Student，使用 Qwen3-8B 三字段 LoRA checkpoint-15550 作为 Teacher。每轮训练都从同一个 LoRA Student 重新开始；逐步保留的是上一阶段已经确定的配置，不是上一阶段训练得到的 OPD checkpoint。

训练集和内部验证集分别包含 3,515 和 159 条数据，对应 [`train_opd_scheme_a.parquet`](./opd/data/train_opd_scheme_a.parquet) 和 [`valid_opd_scheme_a.parquet`](./opd/data/valid_opd_scheme_a.parquet)。三字段数据构造、LoRA 配置和初始化模型结果见 [实验 2 LoRA 说明](../02-structured-three-field-teacher-only-opd/lora/README.md)。

外部评测统一使用本目录下的 [`finqa_three_field_eval.py`](./finqa_three_field_eval.py) 和 [`prompt.py`](./prompt.py)。FinQA test 共 1,147 道题，每道题生成 8 次，`avg@8` 表示全部 9,176 个响应的 Answer 正确率，`best@8` 表示至少有 1 次回答正确的题目比例。外测采样温度统一为 `0.5`。

## 优化路线

```text
实验 2：Teacher-only TopK32
              ↓
加入 Task Reward，coef=0.5
              ↓ 保留 Task Reward
将 coef 从 0.5 提高到 1.0
              ↓ 保留 Task Reward 和 coef=1.0
将 Teacher TopK 从 32 缩小到 16
```

### 第一步：加入 Task Reward

上一轮 Teacher-only OPD 只使用 Teacher 蒸馏损失更新 Student，Reward 仅用于记录 Answer、Program 和格式诊断结果。第一步保留 Teacher TopK32 和其他训练配置，让 Task Reward 正式参与 Actor 优化，并将蒸馏系数设置为 `0.5`：

```text
Actor Loss = PG Loss + 0.5 × Distillation Loss
```

Task Reward 由四部分组成：

```text
0.50 × Answer 正确
+ 0.45 × Program 结果正确且与 Answer 一致
+ 0.025 × Brief 合规
+ 0.025 × 三字段格式正确
```

其中，[`finqa_three_field_reward.py`](./opd/scripts/finqa_three_field_reward.py) 是 VERL 调用入口，[`finqa_three_field_core.py`](./opd/scripts/finqa_three_field_core.py) 负责三字段解析、Program 执行、Answer 校验和分项 Reward 计算。

加入 Task Reward 后，代表性 `avg@8` 从 43.70% 提高到 44.75%。联合目标在当前结果上优于 Teacher-only，因此后续继续保留 Task Reward，并在此基础上调整 Teacher 蒸馏权重。

### 第二步：提高蒸馏系数

第二步保留 Task Reward、Teacher TopK32 以及第一步的其他训练配置，只将蒸馏系数从 `0.5` 提高到 `1.0`：

```text
DISTILLATION_LOSS_COEF: 0.5 → 1.0
```

对应的联合目标为：

```text
Actor Loss = PG Loss + 1.0 × Distillation Loss
```

这一组对比只改变蒸馏系数，是整条优化路线中最接近严格单变量的实验。代表性 `avg@8` 从 44.75% 提高到 45.12%，提升幅度较小，但训练过程保持稳定。第三步因此保留 Task Reward 和 `coef=1.0`，继续优化蒸馏候选集大小。

### 第三步：缩小 Teacher 候选集

第三步保留 Task Reward、`DISTILLATION_LOSS_COEF=1.0` 和其他训练配置，只将 Teacher 蒸馏候选 token 数量从 32 缩小到 16：

```text
distillation.topk: 32 → 16
```

Teacher TopK 控制的是蒸馏候选 token 数量。训练 rollout 的采样 `top_k` 仍为 `-1`，没有启用 `only_stu` 或 Student-support；第三步仍然采用 Teacher-support Forward-KL。

TopK16 的代表性 `avg@8` 为 45.31%，与 TopK32 的 45.12% 基本处于同一水平；稳态平均单步耗时则从 175.67 秒降至 133.36 秒。TopK16 因而保留了前两步形成的主要性能，同时降低了蒸馏训练开销。

## 核心配置

### 三轮公共配置

| 参数 | 实际值 |
|:---|:---:|
| 训练框架 | VERL 0.8.0 |
| Student / Teacher GPU | 3 / 1 |
| Train batch size | 96 |
| PPO mini-batch size | 24 |
| PPO epochs | 1 |
| Student rollout 数量 | 4 |
| Max prompt / response length | 3,000 / 192 |
| Actor learning rate | `1e-5` |
| LR scheduler | Cosine |
| Warmup ratio / minimum LR ratio | 0.10 / 0.10 |
| Weight decay / gradient clipping | 0.01 / 1.0 |
| 计划训练周期 | 4 epoch，36 outer steps/epoch |
| 保存 / 内部验证间隔 | 每 9 step |
| Rollout temperature / top-p / top-k | 1.0 / 1.0 / -1 |
| 蒸馏目标 | Teacher-support `forward_kl_topk` |
| Distillation Policy Gradient | 关闭 |

GPU 1、2、3 用于 Student FSDP Actor 与 Student rollout，GPU 0 用于 Teacher vLLM 推理；4 张 GPU 均为 RTX PRO 6000 Blackwell 96 GB。

关闭 Distillation Policy Gradient 仅表示不将 Teacher 蒸馏信号转换为 Policy Gradient；Task Reward 对应的 PG Loss 仍参与三轮联合 OPD 的 Actor Loss。

### 配置递进

| 配置项 | 优化起点：Teacher-only OPD | 第一步：加入 Reward | 第二步：提高 coef | 第三步：缩小 TopK |
|:---|:---:|:---:|:---:|:---:|
| 相对上一步的修改 | — | Task Reward 参与训练 | `coef: 0.5 → 1.0` | `Teacher TopK: 32 → 16` |
| Task Reward | 关闭 | **开启** | **开启** | **开启** |
| 联合 Loss 中的蒸馏系数 | — | **0.5** | **1.0** | 保留 `1.0` |
| Teacher TopK | 32 | 保留 `32` | 保留 `32` | **16** |

实验 2 未启用联合 Loss，因此不使用蒸馏系数对 Task Reward 与蒸馏 Loss 进行加权。

三轮训练分别由以下脚本启动：

- [`run_reward_coef_0p5_topk32.sh`](./opd/scripts/run_reward_coef_0p5_topk32.sh)；
- [`run_reward_coef_1p0_topk32.sh`](./opd/scripts/run_reward_coef_1p0_topk32.sh)；
- [`run_reward_coef_1p0_topk16.sh`](./opd/scripts/run_reward_coef_1p0_topk16.sh)。

三个启动脚本共用 [`on_policy_distillation_gpu.sh`](./opd/scripts/on_policy_distillation_gpu.sh) 作为 VERL OPD 训练入口。脚本之间只保留当前步骤需要调整的参数差异。

## 外部测试结果

### 第一步：Task Reward、coef=0.5、TopK32

| Checkpoint | `avg@8` | `best@8` |
|:---:|---:|---:|
| step 72 | 44.50% | **61.55%** |
| step 90 | **44.75%** | 60.51% |
| step 108 | 44.26% | 59.46% |
| step 144 | 44.59% | 60.24% |

第一步以 step 90 的 `44.75% avg@8` 作为代表结果。step 72 的 `best@8` 更高，说明两个指标的最佳 checkpoint 并不相同。

### 第二步：Task Reward、coef=1.0、TopK32

| Checkpoint | `avg@8` | `best@8` |
|:---:|---:|---:|
| step 90 | 44.98% | **61.55%** |
| step 108 | **45.12%** | 60.85% |

第二步以 step 108 的 `45.12% avg@8` 作为代表结果。该轮保留了训练日志和外测汇总指标，但没有保留原始逐题外测文件。

### 第三步：Task Reward、coef=1.0、TopK16

| Checkpoint | `avg@8` | `best@8` |
|:---:|---:|---:|
| step 72（第一次） | **45.31%** | **61.64%** |
| step 90 | 44.88% | 61.38% |
| step 108 | 44.99% | 60.94% |

第三步的最佳 `avg@8` 较早出现在 step 72，后续 step 90 和 step 108 均保持在约 45% 附近。

### 整体递进结果

| 对比项 | 优化起点：Teacher-only OPD | 第一步：加入 Reward | 第二步：提高 coef | 第三步：缩小 TopK |
|:---|:---:|:---:|:---:|:---:|
| Task Reward | 关闭 | **开启** | 保留开启 | 保留开启 |
| 联合 Loss 中的蒸馏系数 | — | 0.5 | **1.0** | 保留 `1.0` |
| Teacher TopK | 32 | 保留 `32` | 保留 `32` | **16** |
| 相对上一步的修改 | — | 引入任务正确性奖励 | 提高蒸馏 Loss 权重 | 减少候选 token 数量 |
| 代表性 `avg@8` | 43.70% | **44.75%** | **45.12%** | **45.31%** |
| 相比上一阶段 | — | **+1.05pp** | **+0.37pp** | **+0.19pp** |
| 相比实验 2 起点 | — | **+1.05pp** | **+1.42pp** | **+1.61pp** |
| 稳态平均耗时（秒/step，越低越好） | 172.38 | 169.85 | 175.67 | **133.36** |
| 相比上一阶段耗时变化 | — | **下降 1.47%** | 上升 3.43% | **下降 24.08%** |

```text
43.70% → 44.75% → 45.12% → 45.31%
         +1.05pp    +0.37pp    +0.19pp
```

从实验 2 的 Teacher-only OPD 到 Teacher TopK16，最佳单次 `avg@8` 累计提高 1.61 个百分点。

## 单步训练耗时

单步训练耗时用于观察三次参数调整对训练效率的影响。

| 配置 | 平均耗时（秒/step） | 中位数（秒/step） |
|:---|---:|---:|
| 优化起点：Teacher-only OPD | 172.38 | 173.46 |
| 第一步：加入 Reward | 169.85 | 170.30 |
| 第二步：提高 coef | 175.67 | 175.82 |
| 第三步：缩小 TopK | **133.36** | **133.35** |

优化起点、第一步和第二步均使用 TopK32，平均单步耗时分布在 169.85—175.67 秒；这一级别的波动不足以支持 Task Reward 或蒸馏系数会稳定改变训练速度。第二步和第三步只改变 Teacher TopK，更接近单变量比较：TopK 从 32 缩小到 16 后，平均单步耗时下降 24.08%，中位数也呈现一致变化。

TopK16 的正确率与 TopK32 基本持平，但平均单步耗时从 175.67 秒降至 133.36 秒。后续实验将优先使用 TopK16，在保持主要性能的同时提高训练效率。

## TopK16 最优结果稳定性

为确认 TopK16 最优结果是否稳定，使用相同测试集、Prompt、温度和 K=8 配置进行了 3 次外部测试：

| 重复测试 | `avg@8` | `best@8` |
|:---:|---:|---:|
| 第一次 | **45.31%** | 61.64% |
| 第二次 | 45.27% | 62.16% |
| 第三次 | 45.22% | **62.77%** |
| 三次 `avg@8` 均值 | **45.27%** | — |

三次 `avg@8` 集中在 45.22%—45.31%，极差为 0.09 个百分点。核心递进表使用最佳单次结果 45.31%，稳定性分析使用三次均值 45.27%，两个口径不混用。

## 优化结果与后续选择

第一步加入 Task Reward 后，代表性 `avg@8` 比 Teacher-only 起点提高 1.05 个百分点，当前外部结果说明联合方案整体更强，因此后续两步继续保留 Reward。由于 Teacher-only 没有外测完全配对的 checkpoint，而且当前只有 1 个训练 seed，这一差值只用于描述现有结果层级，不作为 Task Reward 的精确因果收益。

保留 Reward 后，第二步只将蒸馏系数从 `0.5` 提高到 `1.0`。最佳 `avg@8` 增加 0.37 个百分点，更高系数没有破坏训练，但结果仍处于约 45% 的平台。这说明 `coef=1.0` 可以保留，继续单纯增大蒸馏权重则缺少明确依据。

在继续保留 Reward 和 `coef=1.0` 的情况下，第三步将 Teacher TopK 从 32 缩小到 16。最佳单次 `avg@8` 再增加 0.19 个百分点，TopK16 配置的三次复测均值为 45.27%。该差异不足以证明 TopK16 在正确率上显著优于 TopK32，但可以确认主要性能没有下降；同时，稳态平均单步耗时下降 24.08%，因此 TopK16 更适合作为后续同类训练的效率起点。

整条路线最终保留了 Task Reward、`coef=1.0` 和 Teacher TopK16，代表性 `avg@8` 从 43.70% 提高到 45.31%。后两步的正确率增量已经明显收窄，后续优化重点应从继续放大蒸馏系数，转向提高 Task Reward 对 Program 执行结果、Answer 一致性和困难样本的区分能力；若需要进一步压缩候选集，可在当前配置上单独测试 TopK8，并继续同时观察正确率和训练耗时。

## 文件说明

```text
03-structured-three-field-opd-objective-tuning/
├── README.md
├── finqa_three_field_eval.py
├── prompt.py
└── opd/
    ├── data/
    │   ├── train_opd_scheme_a.parquet
    │   └── valid_opd_scheme_a.parquet
    └── scripts/
        ├── finqa_three_field_core.py
        ├── finqa_three_field_reward.py
        ├── on_policy_distillation_gpu.sh
        ├── run_reward_coef_0p5_topk32.sh
        ├── run_reward_coef_1p0_topk32.sh
        └── run_reward_coef_1p0_topk16.sh
```

- `finqa_three_field_eval.py` 与 `prompt.py` 保存三轮训练共用的 FinQA 外部评测实现；
- `opd/data/` 保存三轮训练共用的 OPD 训练集与内部验证集；
- `opd/scripts/` 保存公共 OPD 主脚本、三轮启动脚本和 Task Reward 实现。

[返回项目首页](../README.md)
