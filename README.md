# FinQA On-Policy Distillation

本项目基于 FinQA，研究如何使用 Qwen3-8B 作为 Teacher，通过 On-Policy Distillation（OPD）提升 Qwen3-1.7B 的金融问答能力。实验从 Base Model 直接 OPD 出发，依次完成 Answer-only LoRA 与 OPD、Brief–Program–Answer 三字段 LoRA 与 Teacher-only OPD，并进一步引入 Task Reward，优化蒸馏系数和 Teacher TopK。在此基础上，项目继续提高监督数据中的有效计算信息密度，将 Brief 重构为与 Program 和 Answer 严格对应的显式计算轨迹，并重新训练非思考 Student 与 Teacher。

---

## 前置探索：Base Model 直接 OPD

最初的实验没有对 Qwen3-1.7B 和 Qwen3-8B 进行 FinQA LoRA 训练，而是直接以原始 Qwen3-1.7B 作为 Student、原始 Qwen3-8B 作为 Teacher 进行 Answer-only OPD。

### 核心配置与外部测试结果

- Student：Qwen3-1.7B Base
- Teacher：Qwen3-8B Base
- 输出模式：非思考模式，仅输出最终答案
- 外部评测：FinQA test，每道题采样 8 次，共 1,147 道题

| 模型 | `avg@8` | `best@8` |
|:---|---:|---:|
| Qwen3-1.7B Base | 3.48% | 4.97% |
| Qwen3-8B Base | 9.51% | 12.73% |
| Qwen3-1.7B OPD-392 | **4.98%** | **9.54%** |

相较于原始 Qwen3-1.7B，OPD-392 的 `avg@8` 提高 1.50 个百分点，`best@8` 提高 4.57 个百分点。Base Model 直接 OPD 已经能够产生相对增益，但训练后的绝对正确率仍然较低；Qwen3-8B Base 在相同测试中的 `avg@8` 也只有 9.51%，Teacher 的任务能力同样有限。

FinQA 需要理解长篇金融材料并完成多步计算，而 Answer-only 输出只能传递最终答案，无法直接监督中间计算过程。因此，后续训练先通过 LoRA 提高 Student 和 Teacher 的 FinQA 基础能力，再进行 OPD。

---

## 实验 1：Answer-only Teacher TopK32 OPD

本轮先对 Qwen3-1.7B 和 Qwen3-8B 进行 Answer-only LoRA，再以 LoRA 模型作为 Student 和 Teacher 开展 Teacher TopK32 OPD。

### LoRA 起点

| 角色 | 模型 | `avg@8` | `best@8` |
|:---|:---|---:|---:|
| Student | Qwen3-1.7B Answer-only LoRA | 21.17% | 34.44% |
| Teacher | Qwen3-8B Answer-only LoRA | 59.20% | 63.73% |

Answer-only LoRA 将 Student 的 `avg@8` 从 3.48% 提高到 21.17%，为后续 OPD 提供了更强的任务起点。

### OPD 配置与结果

- 输出格式：Answer-only
- 蒸馏目标：Teacher-token TopK32 Forward KL
- Task Reward 参与损失：否
- Policy Gradient：否
- 外部评测：FinQA test，每道题采样 8 次

| 对比对象 | `avg@8` | `best@8` |
|:---|---:|---:|
| OPD 初始化模型 | 21.17% | 34.44% |
| 同口径 LoRA `best@8` 参考 | 20.73% | 35.92% |
| 本轮 OPD（最高 `avg@8`） | **23.89%** | 36.44% |
| 本轮 OPD（最高 `best@8`） | 23.61% | **36.79%** |

### 结果分析

本轮 Answer-only OPD 的最高 `avg@8` 为 23.89%，相对初始化模型提高 2.72 个百分点；最高 `best@8` 为 36.79%，相对同口径 LoRA 参考结果提高 0.87 个百分点。

OPD 已经产生了有效提升，但整体收益仍然有限。主要问题是 Answer-only 输出缺少 Brief 和 Program，Teacher 只能围绕较短的最终答案分布进行蒸馏，难以向 Student 传递完整的问题理解和计算过程。这一结果推动了后续 Brief–Program–Answer 三字段数据与模型的构造。

具体训练保存点及完整测试结果见 [实验 1 README](./01-answer-only-teacher-topk32-opd-baseline/README.md)。

---

## 实验 2：Brief–Program–Answer Teacher-only OPD

实验 1 的主要限制是只监督最终答案。本轮将输出扩展为 Brief–Program–Answer 三字段：Brief 说明问题目标、关键数值和计算关系，Program 给出可执行的 FinQA Program，Answer 保存归一化结果。

### 三字段数据与 LoRA

Qwen3-1.7B 和 Qwen3-8B 均从原始 Base Model 开始训练，使用同一套三字段训练数据、System Prompt 和 `qwen3_nothink` 模板。最终训练集包含 6,240 条记录，所有 Program 均通过解析和执行校验，Program 执行结果与标准答案一致，Brief 均不超过 64 Qwen3 tokens。

| 角色 | Answer-only LoRA | 三字段 LoRA | `avg@8` 变化 | `best@8` 变化 |
|:---|---:|---:|---:|---:|
| Student | 21.17% / 34.44% | **41.26% / 57.98%** | **+20.09 个百分点** | **+23.54 个百分点** |
| Teacher | 59.20% / 63.73% | **64.71% / 72.71%** | **+5.51 个百分点** | **+8.98 个百分点** |

表中模型结果依次为 `avg@8 / best@8`。三字段 LoRA 同时提高了 Student 和 Teacher 的外部测试结果，其中 Student 的提升更加明显。

### Teacher-only OPD

Student 和 Teacher 分别使用 Qwen3-1.7B 与 Qwen3-8B 三字段 LoRA。Student 在线生成 Brief–Program–Answer 响应，Teacher 在 Student 实际访问的 token 位置提供 TopK32 概率分布，通过 Forward KL 更新 Student。

- OPD 训练框架：VERL 0.8.0
- 训练数据：3,515 条
- 内部验证数据：159 条
- 每条输入生成 4 个 response
- 蒸馏目标：Teacher TopK32 Forward KL
- Task Reward 参与损失：否
- Policy Gradient：否
- 外部评测：FinQA test，每道题采样 8 次

### 结果分析

| 指标 | 上一轮 Answer-only OPD | 三字段 LoRA Student | 本轮 Teacher-only OPD | 相比上一轮 OPD | 相比三字段 LoRA |
|:---:|---:|---:|---:|---:|---:|
| `avg@8` | 23.89% | 41.26% | **43.70%** | **+19.81 个百分点** | **+2.44 个百分点** |
| `best@8` | 36.44% | 57.98% | **61.55%** | **+25.11 个百分点** | **+3.57 个百分点** |

Brief–Program–Answer 三字段 LoRA 形成了明显更强的 Student 和 Teacher。在三字段 LoRA Student 的基础上，Teacher-only OPD 又将 `avg@8` 和 `best@8` 分别提高 2.44 和 3.57 个百分点，说明 Teacher 分布监督仍然能够带来进一步提升。

三字段 LoRA 的数据构造与评测结果见 [LoRA README](./02-structured-three-field-teacher-only-opd/lora/README.md)，Teacher-only OPD 的训练过程与结果分析见 [OPD README](./02-structured-three-field-teacher-only-opd/opd/README.md)。

---

## 实验 3：OPD 目标函数与蒸馏参数优化

Teacher-only TopK32 OPD 将三字段 LoRA Student 的 `avg@8` 从 41.26% 提高到 43.70%，但后期结果逐渐进入平台。本轮保留相同的 Student、Teacher、训练数据和主要配置，先加入 Task Reward，再依次调整蒸馏系数和 Teacher TopK。

### 优化起点与路线

```text
Teacher-only TopK32：43.70%
              ↓
加入 Task Reward，coef=0.5
              ↓ 开启 Task Reward
将 coef 从 0.5 提高到 1.0
              ↓ 开启 Task Reward，coef=1.0
将 Teacher TopK 从 32 缩小到 16
```

三轮训练均使用 Qwen3-1.7B 三字段 LoRA checkpoint-3400 初始化 Student，并使用 Qwen3-8B 三字段 LoRA checkpoint-15550 作为 Teacher。每轮训练都从同一个 LoRA Student 开始，逐步继承的是上一阶段已经确定的配置。

### Task Reward

Task Reward 正式参与 Actor 优化后，联合 Loss 为：

```text
Actor Loss = PG Loss + coef × Distillation Loss
```

Reward 由四部分组成：

```text
0.50 × Answer 正确
+ 0.45 × Program 结果正确且与 Answer 一致
+ 0.025 × Brief 合规
+ 0.025 × 三字段格式正确
```

Answer 和 Program 正确性占主要权重，Brief 与三字段格式用于约束输出结构。

### 配置与外部测试结果

| 阶段 | Task Reward | 蒸馏系数 | Teacher TopK | 代表性 `avg@8` | 相比上一阶段 |
|:---|:---:|:---:|:---:|---:|---:|
| Teacher-only 起点 | 关闭 | — | 32 | 43.70% | — |
| 加入 Task Reward | 开启 | 0.5 | 32 | **44.75%** | **+1.05 个百分点** |
| 提高蒸馏系数 | 开启 | 1.0 | 32 | **45.12%** | **+0.37 个百分点** |
| 缩小候选集 | 开启 | 1.0 | 16 | **45.31%** | **+0.19 个百分点** |

加入 Task Reward 后，代表性 `avg@8` 提高 1.05 个百分点，是三步优化中最主要的正确率增量。继续将蒸馏系数从 `0.5` 提高到 `1.0` 后，`avg@8` 再提高 0.37 个百分点。最后将 Teacher TopK 从 32 缩小到 16，最佳单次结果提高 0.19 个百分点，三次复测均值为 45.27%。

### 训练效率

| 配置 | 平均耗时（秒/step） | 中位数（秒/step） |
|:---|---:|---:|
| Teacher-only TopK32 | 172.38 | 173.46 |
| Task Reward、coef=0.5、TopK32 | 169.85 | 170.30 |
| Task Reward、coef=1.0、TopK32 | 175.67 | 175.82 |
| Task Reward、coef=1.0、TopK16 | **133.36** | **133.35** |

TopK16 的正确率与 TopK32 基本持平，但平均单步耗时从 175.67 秒降至 133.36 秒，下降 24.08%。TopK16 在保持主要性能的同时提高了训练效率。

### 最终选择

- Task Reward：开启
- `DISTILLATION_LOSS_COEF=1.0`
- Teacher TopK：16

三步优化将代表性 `avg@8` 从 43.70% 提高到 45.31%，累计提升 1.61 个百分点。后续同类训练优先采用 Task Reward、`coef=1.0` 和 Teacher TopK16。

完整配置、各 checkpoint 外部测试结果和重复测试结果见 [实验 3：OPD 目标函数与蒸馏参数优化](./03-structured-three-field-opd-objective-tuning/)。

---

## 实验 4：Trace-Enhanced BPA LoRA

实验 2 的 Teacher-only OPD 将三字段 LoRA Student 的 `avg@8` 从 41.26% 提高到 43.70%；实验 3 继续加入 Task Reward、提高蒸馏系数并缩小 Teacher TopK 后，代表性 `avg@8` 达到 45.31%，但后续增量逐渐收窄。测试中也仍然存在 Brief（计算描述）和 Program 基本正确、最终 Answer 却错误的情况。

这些结果表明，增加输出内容能够提供更多蒸馏信号，但原有 Brief 对中间结果及其依赖关系的表达仍不充分。为了让三字段监督更完整地承载同一条计算过程，本轮将 Brief 重构为显式计算轨迹：依次写出实际操作数、运算顺序、中间结果和最终结果，再由 Program 形式化同一过程，并由 Gold Program 的规范化执行结果和 FinQA 标准答案共同复核 Answer。

最终构造 5,951 条 Trace-Enhanced BPA 训练数据。Qwen3-1.7B 和 Qwen3-8B 均从原始 Base Model 开始训练，继续使用 `qwen3_nothink` 模板。训练完成后，使用本轮 `finqa_three_field_eval.py` 在 FinQA test 的 1,147 道题上进行 K=8 外部评测，评测参数为 `temperature=0.5`、`top_p=1.0`、`enable_thinking=false`。

### Qwen3-1.7B LoRA

| LoRA 监督形式 | Answer `avg@8` | Answer `best@8` | 较上一阶段 `avg@8` |
|:---|---:|---:|---:|
| Answer-only | 21.17% | 34.44% | — |
| Brief–Program–Answer | 41.26% | 57.98% | +20.09 个百分点 |
| Trace-Enhanced BPA | **48.812%** | **62.424%** | **+7.552 个百分点** |

### Qwen3-8B LoRA

| LoRA 监督形式 | Answer `avg@8` | Answer `best@8` | 较上一阶段 `avg@8` |
|:---|---:|---:|---:|
| Answer-only | 59.20% | 63.73% | — |
| Brief–Program–Answer | 64.71% | 72.71% | +5.51 个百分点 |
| Trace-Enhanced BPA | **70.031%** | **75.937%** | **+5.321 个百分点** |

在训练数据中引入显式计算轨迹后，两个模型的评测结果均较上一阶段有所提升，进一步支持了通过提高计算信息密度、强化三字段映射来增强非思考模型能力的思路。对于 Qwen3-1.7B，Answer `avg@8` 从 Answer-only LoRA 的 21.17% 提高到 48.812%；这一结果也表明，在无法为推理阶段引入额外思考过程的低时延场景中，针对任务结构优化监督数据，能够为小规模非思考模型提供更有效的能力增强路径。

本次 GitHub 更新仅包含 Trace-Enhanced BPA 的 LoRA 阶段，不包含后续 OPD 内容。完整的数据重构方法与 LoRA 评测结果见 [实验 4 LoRA README](./04-executable-brief-task-reward-topk16-opd/lora/README.md)。

---

## 目录结构

```text
FinQA-On-Policy-Distillation/
├── README.md
├── data/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── 01-answer-only-teacher-topk32-opd-baseline/
│   ├── README.md
│   ├── finqa_answer_only_eval.py
│   ├── lora/
│   └── opd/
├── 02-structured-three-field-teacher-only-opd/
│   ├── README.md
│   ├── finqa_three_field_eval.py
│   ├── prompt.py
│   ├── lora/
│   └── opd/
├── 03-structured-three-field-opd-objective-tuning/
│   ├── README.md
│   ├── finqa_three_field_eval.py
│   ├── prompt.py
│   └── opd/
└── 04-executable-brief-task-reward-topk16-opd/
    ├── finqa_three_field_eval.py
    ├── prompt.py
    └── lora/
```

- `data/` 保存 FinQA 官方数据。
- `01-answer-only-teacher-topk32-opd-baseline/` 保存 Answer-only LoRA 与 Teacher TopK32 OPD。
- `02-structured-three-field-teacher-only-opd/` 保存三字段数据、LoRA 与 Teacher-only OPD。
- `03-structured-three-field-opd-objective-tuning/` 保存 Task Reward、蒸馏系数与 Teacher TopK 优化。
- `04-executable-brief-task-reward-topk16-opd/` 保存本轮公开的 Trace-Enhanced BPA LoRA 数据构造、训练与评测内容。
- 实验 1、实验 2 根目录下的评测脚本由对应 LoRA 和 OPD 共用；实验 3 根目录下的评测脚本与 Prompt 由三轮 OPD 共用；实验 4 根目录下的评测脚本与 Prompt 用于本轮 LoRA 外部评测。

---

## 数据

FinQA 官方数据来自 [czyssrs/FinQA](https://github.com/czyssrs/FinQA)。本仓库保留以下 3 个 split：

| 文件 | 样本数 | 用途 |
|:---|---:|:---|
| `data/train.json` | 6,251 | 训练数据来源 |
| `data/dev.json` | 883 | 验证数据来源 |
| `data/test.json` | 1,147 | 外部测试数据来源 |

实验目录中的 LoRA 数据由上述官方数据构造，并保持实际训练和评测时使用的输入格式；OPD 数据根据 LoRA Teacher 与 Student 的训练集 K=8 结果进一步筛选。
