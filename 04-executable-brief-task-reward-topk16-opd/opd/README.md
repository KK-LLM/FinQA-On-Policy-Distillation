# FinQA Trace-Enhanced BPA OPD

前一阶段的 Brief–Program–Answer（BPA）LoRA 已经明显增强了 Qwen3-1.7B 的 FinQA 能力，但 [Teacher-only TopK32 OPD](../../02-structured-three-field-teacher-only-opd/opd/README.md) 只将 Student 的 Answer `avg@8` 从 41.26% 进一步提高到 43.70%，增量为 2.44 个百分点。继续加入 Task Reward、调整蒸馏系数并将 Teacher TopK32 缩小到 TopK16 后，代表性结果提高到 45.31%，后续增量仍在逐步收窄。

结合测试中出现的 Brief（计算描述）和 Program 基本正确、最终 Answer 却错误的情况，后续优化开始从训练数据本身寻找原因。仅仅增加三字段输出长度，可以扩大 Teacher 向 Student 提供分布监督的范围，但如果 Brief 没有明确写出中间结果及其依赖关系，新增 token 仍可能只让模型学到字段形式、相关数字和局部计算模式，难以稳定建立从计算过程到最终答案的完整映射。

本轮因此将 Brief 重构为可核验的显式计算轨迹，并同步强化 `Brief → Program → Answer` 的三阶段对应关系。使用这套数据训练后，Qwen3-1.7B 和 Qwen3-8B Trace-Enhanced BPA LoRA 的 Answer `avg@8` 分别达到 48.812% 和 70.031%。在此基础上继续进行 OPD，目的是检验信息密度更高、字段关系更完整的监督表示，能否进一步改善 Teacher 向非思考 Student 的能力迁移。

LoRA 数据重构方法、训练结果和初始化模型选择见 [Trace-Enhanced BPA LoRA](../lora/README.md)。本轮沿用此前目标函数优化后保留的 Teacher TopK16、Task Reward 与 `coef=1.0`，不再同时引入新的 OPD 目标变量。

## OPD 数据

OPD 候选数据来自 5,951 条 Trace-Enhanced BPA LoRA 训练数据。全部记录先经过与本轮 Reward 一致的解析、Program 执行、Answer 校验和显式轨迹检查；修复 74 条、排除 1 条后，共有 5,950 条进入 Student 与 Teacher 的 K=8 候选评测。

为了减少数据选择策略变化对前后结果比较的影响，本轮继续沿用 [Teacher-only TopK32 OPD](../../02-structured-three-field-teacher-only-opd/opd/README.md) 的数据筛选方法。Student 和 Teacher 分别对每道候选题生成 8 次响应；一次响应只有同时满足 Answer 正确、Program 可执行且其结果与 Answer 一致，以及 Brief 和三字段输出符合当前协议，才计为一次成功。

```text
S = Student 在 8 次生成中的成功次数
T = Teacher 在 8 次生成中的成功次数
```

`S` 和 `T` 的取值范围均为 0～8。根据两个模型的成功次数及其差距，候选题被划分为以下三类：

| 数据组 | 筛选条件 | 作用 |
|---|:---:|---|
| 严格核心组 | `T >= 7`、`S >= 1`、`T-S >= 2` | 保留 Teacher 表现稳定、明显强于 Student，同时 Student 已具备一定基础的题目 |
| 高难度组 | `T >= 6`、`S = 0` | 保留 Student 尚未掌握、但 Teacher 已能稳定完成的题目 |
| 稳定锚点组 | `T = S = 8` | 保留两个模型均已稳定掌握的题目，维持已有能力 |

本轮同时保留了上一轮的 5% 分层内部验证集、随机种子 `42`、稳定锚点选择方式，以及针对 2-step 高对比题和 3+ steps 题目的定向补充规则。由于候选数据和 Student、Teacher 均已更新为本轮版本，最终入选数量与上一轮不同；保持数据选择方法不变，可以避免同时引入新的样本筛选策略，使前后两轮的数据构造口径更容易对应。

最终从候选池中选择 2,947 道唯一题目。内部验证集包含 145 道题；训练集包含 2,802 个唯一 ID，并针对 2-step 与 3+ steps 样本进行定向补充，形成 3,422 个训练实例。Train 与内部 valid 的题目 ID 交集为 0。

| 数据划分 | 唯一题目 | 最终实例 |
|---|---:|---:|
| Train | 2,802 | 3,422 |
| Internal valid | 145 | 145 |

本轮 OPD 实际使用的数据为：

- [`train_finqa_c_scheme_a_0828.parquet`](./data/train_finqa_c_scheme_a_0828.parquet)：3,422 条训练数据；
- [`valid_finqa_c_scheme_a_0828.parquet`](./data/valid_finqa_c_scheme_a_0828.parquet)：145 条内部验证数据。

## OPD 方法

Qwen3-1.7B Trace-Enhanced BPA LoRA 作为 Student，Qwen3-8B Trace-Enhanced BPA LoRA 作为 Teacher。Student 在线生成固定的 `Brief–Program–Answer` 三字段响应，Teacher 在 Student 实际访问的 token 位置提供 Top-16 概率分布，通过 Teacher-support Forward KL 更新 Student。

训练与内部验证均设置 `enable_thinking=false`，不生成开放式 `<think>` 内容。Teacher TopK16 控制的是蒸馏候选 token 数量；rollout `top_k=-1` 表示生成阶段不进行 Top-k 截断，两者不是同一个参数。

### 核心配置

| 配置项 | 实际值 |
|---|---:|
| Student / Teacher | Qwen3-1.7B / Qwen3-8B Trace-Enhanced BPA LoRA |
| 蒸馏目标 | Teacher-support `forward_kl_topk` |
| Teacher Top-K | 16 |
| 蒸馏系数 | 1.0 |
| Task Reward | 开启 |
| Distillation Policy Gradient | 关闭 |
| Train batch / PPO mini-batch | 96 / 24 |
| 每个 Prompt 的 rollout 数量 | 4 |
| PPO epochs | 1 |
| 计划训练周期 | 4 epochs，共 140 outer steps |
| Student / Teacher GPU | 3 / 1 |

### Task Reward

本轮训练数据将 Brief 从概括性的计算描述重构为包含中间结果和步骤依赖的显式计算轨迹，原有 Reward 所依据的三字段协议已无法完整判断这些新增信息是否与 Program 和 Answer 一致，因此需要同步更新对应的奖励判定。这一调整用于使在线生成的评价标准与新的监督目标保持一致，并非另行搜索新的 Reward 结构。为避免在更新判定内容的同时改变各项目标的相对作用，本轮继续保留原有的四项权重，使奖励总尺度和优化重点尽可能与上一阶段保持一致。

Task Reward 同时检查最终答案、Program 计算链、Brief 轨迹和三字段格式：

| Reward 项 | 权重 | 判定内容 |
|---|---:|---|
| Answer correct | 0.50 | Answer 格式有效且与 Gold Answer 一致 |
| Program chain correct | 0.45 | Program 步数、依赖图和操作数有效，执行结果正确且与输出 Answer 一致 |
| Brief trace consistent | 0.025 | Brief 满足长度要求，并与 Program 的实际执行轨迹和 Answer 一致 |
| Strict three-field format | 0.025 | 输出严格符合 `Brief`、`Program`、`Answer` 三字段格式 |

其中 0.45 的 Program 项是对整条 Program–Answer 终局链路的联合判定，不是按中间步骤分别发放的过程奖励。

实际训练入口与 Reward 实现保存在 [`scripts/`](./scripts/) 中。完整训练使用以下调用链：

```text
test.sh
  → run_OPD_FinQA_C_1P7B_from_8B_topk16_coef1p0.sh
    → on_policy_distillation_gpu_c.sh
```

## 外部评测

训练完成后，使用 [`finqa_three_field_eval.py`](../finqa_three_field_eval.py) 和 [`prompt.py`](../prompt.py) 在 FinQA test 的 1,147 道题上进行 K=8 外部评测。评测参数设为 `temperature=0.5`、`top_p=1.0`、`enable_thinking=false`；每个模型共生成 9,176 条响应。

### 计划内 4 epochs

计划内 4 epochs 的外部测试结果如下：

| Step | Answer `avg@8` | Answer `best@8` | Program | P–A consistency | Exact 3-field |
|---:|---:|---:|---:|---:|---:|
| 72 | 55.656% | 69.050% | 64.091% | 84.990% | 99.313% |
| 108 | 56.266% | 69.398% | 64.004% | **86.221%** | 99.172% |
| **117** | **56.430%** | **69.834%** | **64.146%** | 86.047% | **99.324%** |
| 140 | 56.255% | 69.660% | 64.124% | 85.786% | 99.270% |

step 117 同时取得计划内最高的 Answer `avg@8`、Answer `best@8`、Program 正确率和严格三字段格式率，因此被选为计划内最佳 checkpoint。最后一个 step 140 的训练已正常完成，但外部结果略低于 step 117，不以训练终点替代实际选出的最佳模型。

### 续训 2 epochs

计划内 4 epochs 在 step 140 完成。训练后期的 KD Loss 和 Actor Loss 仍在缓慢下降，训练 Reward 也保持上升趋势，因此继续增加 2 epochs，用于检验当前数据和训练目标下是否仍有进一步的能力提升空间。

| Step | Answer `avg@8` | Answer `best@8` | Program | Exact 3-field |
|---:|---:|---:|---:|---:|
| 156 | 56.528% | 69.398% | **64.440%** | 99.204% |
| 180 | 56.332% | 69.573% | 64.048% | **99.346%** |
| **196** | **56.648%** | **70.009%** | 64.266% | 99.085% |
| 208 | 56.517% | **70.009%** | 64.211% | 99.226% |

续训阶段的 Answer `avg@8` 集中在 56.332%～56.648%，没有形成明显高于计划内训练的新结果区间。其中，step 196 的 56.648% 是续训阶段的最高观察值，而 step 117 为 56.430%，两者的绝对差只有 0.218%，因此不能认为 step 196 已经稳定优于 step 117。

这说明训练 Loss 的缓慢下降没有继续转化为稳定的外部正确率提升。在当前数据和训练目标下，模型经过计划内 4 epochs 后已经进入平台期，可以认为训练较为充分；继续沿用相同配置增加 epoch，暂未表现出明确收益。因此，本轮仍选择计划内的 step 117 作为 Trace-Enhanced BPA OPD 的正式模型。

## OPD 效果与收益

本轮 OPD 收益以实际使用的 Qwen3-1.7B Trace-Enhanced BPA LoRA 为直接基线，两者使用同一 FinQA test、同一 Prompt、同一 K=8 评测脚本和相同采样参数。

### 核心效果收益

| 核心指标 | Trace-Enhanced BPA LoRA | Trace-Enhanced BPA OPD | OPD 收益 |
|---|:---:|:---:|---:|
| Answer `avg@8` | 48.812% | **56.430%** | **+7.618 pp** |
| Answer `best@8` | 62.424% | **69.834%** | **+7.411 pp** |

相较于作为 Student 初始化模型的 Trace-Enhanced BPA LoRA，本轮 OPD 将 Answer `avg@8` 从 48.812% 提高到 56.430%，将 Answer `best@8` 从 62.424% 提高到 69.834%，分别取得 7.618 和 7.411 个百分点的增益。两项核心指标同步提高，说明本轮 OPD 不仅提高了模型单次生成正确答案的平均能力，也扩大了多次采样下能够正确解决的问题范围。

### OPD 对三字段生成质量的提升

| 三字段指标 | Trace-Enhanced BPA LoRA | Trace-Enhanced BPA OPD | OPD 收益 |
|---|:---:|:---:|---:|
| Program | 56.811% | **64.146%** | **+7.334 pp** |
| Answer ∩ Program | 46.774% | **56.277%** | **+9.503 pp** |
| P–A consistency | 79.435% | **86.047%** | **+6.612 pp** |
| Exact 3-field | 0.447% | **99.324%** | **+98.878 pp** |

- **Program**：生成的 Program 可以正常解析和执行，并且执行结果与标准答案一致。
- **Answer ∩ Program**：同一条响应中的 Answer 正确，同时 Program 的执行结果也正确。该指标比单独检查 Answer 或 Program 更严格。
- **P–A consistency**：在 Program 可以执行的响应中，Program 的执行结果与模型输出的 Answer 是否一致。它衡量模型能否将计算结果正确传递到最终答案，但不代表两者一定与标准答案一致。
- **Exact 3-field**：输出是否严格符合 `Brief`、`Program`、`Answer` 三行结构及固定字段顺序。该指标反映格式稳定性，不单独等同于解题正确率。

OPD 不仅提高了最终 Answer 正确率，也同步改善了 Program 正确率、Answer 与 Program 的联合正确率以及两者的一致性。其中，Answer ∩ Program 提高 9.503 个百分点，是除严格格式率外增幅最大的三字段指标，说明 OPD 带来的收益更多体现在完整计算链路的共同正确，而不是某一个字段的孤立改善。Exact 3-field 从 0.447% 提高到 99.324%，则表明在线训练进一步稳定了模型对固定三字段协议的遵循。

### OPD 阶段增益对比

| 监督方案 | LoRA `avg@8` | OPD `avg@8` | OPD 阶段增益 |
|---|:---:|:---:|---:|
| Brief–Program–Answer | 41.26% | 45.31% | +4.05 pp |
| Trace-Enhanced BPA | **48.812%** | **56.430%** | **+7.618 pp** |

Trace-Enhanced BPA OPD 将 Answer `avg@8` 从 48.812% 提高到 56.430%，取得 7.618 个百分点的 OPD 增益。上一套 Brief–Program–Answer 最终方案的 OPD 增益为 4.05 个百分点，本轮增益约为其 1.88 倍，已接近两倍。这说明本轮优化带来的收益并未停留在 LoRA 阶段：提高 Brief 中的计算信息密度、强化 `Brief → Program → Answer` 映射后，Student 在在线蒸馏阶段获得了更大的提升空间，也进一步体现了本轮数据重构与 Reward 适配的实际作用。

## 多步问题

按 Gold Program 的运算步数拆分后，OPD 对 2-step 和 3+ steps 问题的提升明显高于单步问题：

| Program 步数 | Trace-Enhanced BPA LoRA | Trace-Enhanced BPA OPD | OPD 增益 |
|---|:---:|:---:|---:|
| 1 step | 54.320% | 58.180% | +3.860 pp |
| 2 steps | 44.835% | 57.091% | **+12.256 pp** |
| 3+ steps | 25.298% | 39.583% | **+14.286 pp** |

随着 Program 步数增加，Trace-Enhanced BPA OPD 相对 Trace-Enhanced BPA LoRA 的增量同步扩大。这一结果与显式计算轨迹强化中间结果传递、三字段映射更适合承载多步蒸馏信号的设计预期一致，也说明本轮的主要收益并非只来自简单题目的进一步拟合。

## 结果分析

Trace-Enhanced BPA OPD 将 Qwen3-1.7B Trace-Enhanced BPA LoRA 的 Answer `avg@8` 从 48.812% 提高到 56.430%，并同时改善 Answer、Program、二者联合正确率和 Program–Answer 一致性。

更值得关注的是，2-step 与 3+ steps 的增量分别达到 12.256 和 14.286 个百分点，远高于单步问题的 3.860 个百分点。显式计算轨迹不仅提高了 LoRA 阶段的监督信息密度，也为 OPD 提供了更连续的三字段蒸馏载体；当前结果进一步支持了通过增强计算信息、明确中间结果依赖和强化 `Brief → Program → Answer` 映射来提升非思考模型能力的思路。

本轮所有训练与评测过程均保持 `enable_thinking=false`，模型按照固定三字段协议直接生成结果，不依赖开放式思考过程。Trace-Enhanced BPA OPD 将 1.7B Student 的 Answer `avg@8` 提高到 56.430%，说明通过增强训练数据中的计算信息、明确中间结果依赖并结合在线蒸馏，可以在非思考生成方式下显著提升模型的 FinQA 任务能力，也为对推理时延敏感的实际场景提供了一条兼顾处理效率与模型能力的训练路径。

## 文件布局

本轮 OPD 的核心文件布局如下：

```text
opd/
├── README.md
├── data/
│   ├── train_finqa_c_scheme_a_0828.parquet
│   └── valid_finqa_c_scheme_a_0828.parquet
└── scripts/
    ├── finqa_three_field_core_c_0828.py
    ├── finqa_three_field_reward_c_0828.py
    ├── on_policy_distillation_gpu_c.sh
    ├── run_OPD_FinQA_C_1P7B_from_8B_topk16_coef1p0.sh
    └── test.sh
```

| 文件或目录 | 说明 |
|---|---|
| `data/` | VERL 实际读取的 3,422 条 OPD 训练数据和 145 条内部验证数据。 |
| `scripts/finqa_three_field_core_c_0828.py` | 三字段解析、Program 执行、Answer 校验和分项 Reward 逻辑。 |
| `scripts/finqa_three_field_reward_c_0828.py` | VERL 自定义 Reward 的 `compute_score` 调用入口。 |
| `scripts/on_policy_distillation_gpu_c.sh` | 将运行参数转换为 VERL/Hydra 配置并启动 OPD。 |
| `scripts/run_OPD_FinQA_C_1P7B_from_8B_topk16_coef1p0.sh` | 本轮模型、数据、训练目标与资源参数配置。 |
| `scripts/test.sh` | 容器内的训练启动、日志查看、恢复与 checkpoint 合并操作入口。 |
| [`../finqa_three_field_eval.py`](../finqa_three_field_eval.py)、[`../prompt.py`](../prompt.py) | LoRA 与 OPD 共用的外部评测脚本和 Prompt 定义。 |

[返回项目首页](../../README.md)
