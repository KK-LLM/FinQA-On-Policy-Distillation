# FinQA On-Policy Distillation

本项目基于 FinQA，研究如何使用 Qwen3-8B 作为 Teacher，通过 On-Policy Distillation（OPD）提升 Qwen3-1.7B 的金融问答能力。实验从 Base Model 直接 OPD 开始，在验证蒸馏有效性的同时，也暴露出 Student 和 Teacher 的 FinQA 基础能力不足，因此后续采用「先 LoRA、再 OPD」的训练方式。

## 前置探索：Base Model 直接 OPD

最初的实验没有对 Qwen3-1.7B 和 Qwen3-8B 进行 FinQA LoRA 训练，而是直接以原始 Qwen3-1.7B 作为 Student、原始 Qwen3-8B 作为 Teacher 进行 OPD。

### 核心配置

- Student：Qwen3-1.7B Base
- Teacher：Qwen3-8B Base
- 输出模式：非思考模式，仅输出最终答案
- 外部评测：FinQA test，每道题采样 8 次，共 1,147 道题

`avg@8` 和 `best@8` 来自外部测试阶段的 8 次采样。

### 外部测试结果

| 模型 | `avg@8` | `best@8` |
|---|---:|---:|
| Qwen3-1.7B Base | 3.48% | 4.97% |
| Qwen3-8B Base | 9.51% | 12.73% |
| Qwen3-1.7B OPD-392 | 4.98% | 9.54% |

相较于原始 Qwen3-1.7B，OPD-392 的 `avg@8` 从 3.48% 提升至 4.98%，绝对提升 1.50 个百分点，相对提升约 43.1%；`best@8` 从 4.97% 提升至 9.54%，绝对提升 4.57 个百分点，相对提升约 92.0%。

这组结果说明，Base Model 直接 OPD 已经能够产生明显的相对增益，但训练后的绝对正确率仍然较低。原始 Qwen3-8B 在相同测试中的 `avg@8` 也只有 9.51%，表明 Teacher 在当前任务和输出模式下的能力同样有限。

FinQA 需要理解长篇金融材料并完成多步计算，而本轮实验采用非思考模式，只蒸馏最终答案。综合这些现象推测，任务本身的推理难度、8B Teacher 的能力上限，以及 Answer-only 输出无法直接传递中间计算过程，共同限制了直接 OPD 的最终效果。这是基于实验结果的分析判断，不能作为对单一因素的严格因果归因。

基于这次探索，后续训练先通过 LoRA 让 Student 和 Teacher 学习 FinQA 的任务形式与答案分布，再进行 OPD。该次探索未保留完整的训练中间文件，因此这里只记录实际保留下来的关键配置和最终评测结果，不提供对应的复现实验目录。

## Answer-only LoRA + OPD：Teacher TopK32 Baseline

在直接 OPD 的基础上，Qwen3-1.7B 和 Qwen3-8B 分别进行了 Answer-only LoRA 训练。本轮 OPD 使用 Qwen3-1.7B LoRA checkpoint-1400 初始化 Student，使用 Qwen3-8B LoRA checkpoint-5750 作为 Teacher。

### 核心配置

- Student：Qwen3-1.7B Answer-only LoRA checkpoint-1400
- Teacher：Qwen3-8B Answer-only LoRA checkpoint-5750
- LoRA 训练框架：LLaMAFactory
- OPD 训练框架：VERL
- 输出格式：Answer-only
- 蒸馏目标：Teacher-token TopK32 Forward KL
- 任务奖励参与损失：否
- Policy Gradient：否
- 外部评测：FinQA test，每道题采样 8 次，共 1,147 道题

### 外部测试结果

| 模型 | `avg@8` | `best@8` |
|---|---:|---:|
| LoRA checkpoint-1400 | 21.17% | 34.44% |
| LoRA checkpoint-800 | 20.73% | 35.92% |
| OPD-910 | 23.61% | **36.79%** |
| OPD-1001 | 23.56% | 36.62% |
| OPD-1092 | **23.89%** | 36.44% |

三个 OPD checkpoint 的 `avg@8` 均在 23.56%–23.89% 之间，`best@8` 均在 36.44%–36.79% 之间。其中，OPD-1092 取得最高 `avg@8`，OPD-910 取得最高 `best@8`。

### 结果分析

LoRA checkpoint-1400 在 OPD 前已经达到 21.17% `avg@8` 和 34.44% `best@8`，相较于原始 Qwen3-1.7B 的 3.48% 和 4.97%，为后续蒸馏提供了明显更强的任务起点。

相较于实际用于初始化 Student 的 LoRA checkpoint-1400，OPD-1092 的 `avg@8` 提升 2.72 个百分点，`best@8` 提升 2.00 个百分点，说明本轮 OPD 训练能够继续带来增益。

LoRA checkpoint-800 在同口径测试中达到 20.73% `avg@8` 和 35.92% `best@8`。与 checkpoint-1400 相比，它的 `avg@8` 低 0.44 个百分点，`best@8` 高 1.48 个百分点，因此不能将其简单视为两个指标都更强的 LoRA checkpoint。

与 checkpoint-800 相比，OPD-1092 的 `avg@8` 提升 3.16 个百分点，`best@8` 提升 0.52 个百分点。若分别取本轮最高值，`avg@8` 和 `best@8` 的提升为 3.16 和 0.87 个百分点。

因此，本轮实验被保留为「有效但收益有限的失败基线」：训练流程已经产生可测量的提升，但相较于 `best@8` 更高的 LoRA checkpoint-800，OPD 在 `best@8` 上的额外收益仍然有限。

上述结果只能说明这套完整训练配置的最终表现，不能将收益或局限严格归因于某一个超参数。LoRA checkpoint-800 也不是本轮 OPD 的实际初始化模型，仅作为同口径外部测试下的性能参照。

## 目录结构

```text
FinQA-On-Policy-Distillation/
├── README.md
├── data/
│   ├── train.json
│   ├── dev.json
│   └── test.json
└── 01-answer-only-teacher-topk32-opd-baseline/
    ├── lora/
    │   ├── Qwen3-1.7B/
    │   ├── Qwen3-8B/
    │   ├── data/
    │   ├── finqa_lora_eval.py
    │   └── README.md
    └── opd/
        ├── data/
        ├── scripts/
        └── README.md
```

- `data/` 保存 FinQA 官方数据。
- 编号实验目录直接保存每轮实验的 LoRA、OPD 配置与结果。
- `lora/data/` 保存 Answer-only LoRA 实际使用的训练、验证和测试数据。
- `lora/Qwen3-1.7B/` 和 `lora/Qwen3-8B/` 分别保存 Student、Teacher 的 LoRA 训练配置、启动脚本和合并配置。
- `lora/finqa_lora_eval.py` 为 LoRA 外部评测脚本。
- `opd/data/` 保存 VERL 实际读取的 OPD 训练集和验证集。
- `opd/scripts/` 保存 OPD 训练入口、VERL 配置和 FinQA Reward。

LoRA 阶段的具体配置见 [Answer-only LoRA 目录](./01-answer-only-teacher-topk32-opd-baseline/lora/)。

OPD 阶段的数据筛选、训练配置和实验结果见 [Teacher TopK32 OPD 目录](./01-answer-only-teacher-topk32-opd-baseline/opd/)。

## 数据

FinQA 官方数据来自 [czyssrs/FinQA](https://github.com/czyssrs/FinQA)。本仓库保留以下 3 个 split：

| 文件 | 样本数 | 用途 |
|---|---:|---|
| `data/train.json` | 6,251 | 训练数据来源 |
| `data/dev.json` | 883 | 验证数据来源 |
| `data/test.json` | 1,147 | 外部测试数据来源 |

实验目录中的 LoRA 数据由上述官方数据构造，并保持实际训练和评测时使用的输入格式；OPD 数据根据 LoRA Teacher 与 Student 的训练集 K=8 结果进一步筛选。
