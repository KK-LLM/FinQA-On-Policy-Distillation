# FinQA Trace-Enhanced BPA LoRA

最初引入 Brief–Program–Answer（下文简称 BPA），是希望突破 Answer-only 输出过短的限制。增加与任务相关的输出内容后，Teacher 可以在更多 token 上向 Student 提供分布监督。使用这套数据训练后，Qwen3-1.7B 非思考模型的 Answer `avg@8` 达到 41.26%，说明增加结构化输出内容确实能够为模型提供更多可学习的信息。

但后续 [Teacher-only TopK32 OPD](../../02-structured-three-field-teacher-only-opd/opd/README.md) 完整训练后，`avg@8` 只进一步提高到 43.70%，相对 LoRA Student 增加 2.44 个百分点。继续加入 Task Reward、提高蒸馏系数并将 Teacher TopK32 调整为 TopK16 后，代表性 `avg@8` 提高到 45.31%，后续增量仍然逐渐收窄。测试中也仍然存在 Brief（计算描述）和 Program 基本正确、最终 Answer 却错误的情况。

这些结果让我进一步意识到，增加输出内容本身还不够，新增内容是否包含足够有效的计算信息同样重要。原有 Brief 已经能够说明计算目标、来源数值和核心操作关系，但通常不会逐步写出实际中间结果，也没有完整建立这些结果到最终 Answer 的数值映射。更长的输出虽然增加了蒸馏信号的数量，如果缺少清晰的过程约束，Student 学到的仍可能主要是字段形式、相关数字和局部计算模式，Program 到 Answer 的转换也就难以保持稳定。

基于这一判断，后续对 Brief 的优化不再停留在增加输出内容，而是进一步明确每一步计算及其结果：先在 Brief 中展开可核验的计算轨迹，再由 Program 形式化同一过程，最后由 Program 的执行结果确定 Answer。通过建立 `Brief → Program → Answer` 的三阶段映射，为 Student 提供信息密度更高、字段关系更完整的蒸馏信号，并在此基础上重新训练 Student 和 Teacher。

完整的目标函数与蒸馏参数优化过程见[三字段 OPD 目标函数与蒸馏参数优化](../../03-structured-three-field-opd-objective-tuning/README.md)。

## 显式计算轨迹

每条监督数据都要求 Brief 写出实际参与计算的数值、运算顺序、中间结果和最终结果，再用同一计算生成 Program，并由 Program 的执行结果确定 Answer。

```text
Context and Question
        ↓
explicit calculation in Brief
        ↓
the same calculation formalized as Program
        ↓
executed Program result
        ↓
Answer
```

这次优化的重点不再是继续增加输出长度，而是把每一步的实际计算结果写清楚，并完整建立它们经由 Program 到最终 Answer 的对应关系。推理时仍使用 `qwen3_nothink`，不生成开放式 `<think>` 内容。

## 数据重构方法

### 补全多步计算中的结果传递

数据重构首先解决多步计算中的结果传递问题。原有 Brief 会说明计算目标、来源数值和“先减、再除”等核心操作关系，但通常不会写出每一步实际得到的数值，也不会明确展示该结果如何进入下一步。新的 Brief 将所有影响最终 Answer 的操作和中间结果依次展开，使同一条计算链能够继续映射到 Program 和 Answer。

每条 Brief 必须满足以下要求：

- 说明 Question 要求的金融目标；
- 标明实际参与计算的数值、金融含义和单位；
- 按 Program 顺序写出全部必要运算；
- 多步计算保留后续步骤真正依赖的中间结果；
- 最终值与 Gold Program 的归一化执行结果一致，并通过 FinQA 标准答案复核；
- 不添加 Program 中不存在的步骤；
- 不引用 Context 无法支持的数值或关系。

例如：

```text
Brief: Total 2018 operating expenses were 41932.20339 million: 9896 million ÷ 23.6% = 41932.20339 million.
Program: divide(9896, 23.6%)
Answer: 41932.20339
```

Brief 中的除法、操作数和结果与 Program 完全对应。Answer 由最终确认的 Gold Program 执行并归一化得到，再与 FinQA 标准答案进行交叉复核。三个字段不再只是按照固定格式排列，而是把同一计算分别表示为自然语言执行轨迹、形式化程序和最终结果。

### 按 Program 复杂度分配 Brief 长度

这次没有把所有 Brief 统一拉长。1–2 步样本继续限制在 64 tokens 内，只有多步计算确实需要展开中间结果时，才根据 Program 复杂度增加可用长度：

| Program 运算步数 | Brief 上限 |
|:---:|:---:|
| 1 – 2 | 64 Qwen3 tokens |
| 3 – 4 | 128 Qwen3 tokens |
| 5 – 6 | 192 Qwen3 tokens |

分级上限让输出长度服务于计算过程的完整表达，而不是单纯增加 token 数量。新增空间只用于保留必要的中间结果和步骤依赖，不用于补充与求解无关的背景。最终 Brief 长度平均为 50 tokens，P95 为 103 tokens，最大为 191 tokens，全部满足对应层级的限制。

### 建立 Brief、Program 与 Answer 的一致映射

数据构造以最终确认并通过执行校验的 Gold Program 作为计算依据。Program 从左到右执行后，将每一步实际使用的操作数、中间结果和最终结果写入 Brief；Program 保留同一运算顺序，Answer 则由 Gold Program 的最终执行结果归一化得到，再与 FinQA 标准答案进行交叉复核。只有执行结果与标准答案一致的记录才进入最终训练集。

数值答案最多保留 5 位小数，并删除无意义尾零；判断结果统一为小写 `yes` 或 `no`。这样可以保证 Brief 展开的计算过程、Program 的形式化表达和 Answer 的最终结果来自同一条经过双重复核的计算链。

## LLaMAFactory 数据格式

最终数据转换为 LLaMAFactory 使用的 Alpaca 四字段格式：

| 字段 | 内容 |
|---|---|
| `system` | 三字段输出协议、显式计算轨迹约束和 FinQA DSL 规则 |
| `instruction` | `Solve the Question using the provided Context.` |
| `input` | FinQA Context、表格和 Question |
| `output` | 严格三行的 `Brief`、`Program` 和 `Answer` |

[`dataset_info.json`](./data/dataset_info.json) 将四个字段分别注册为 LLaMAFactory 的 `system`、`prompt`、`query` 和 `response`。数据只包含 LoRA train，训练期间不使用单独的 LoRA valid；模型选择在训练完成后通过 FinQA test K=8 外部评测完成。

## 数据结果

最终构造并用于 LoRA 训练的数据为 [`finqa_brief_program_train_C_0826.json`](./data/finqa_brief_program_train_C_0826.json)，共 5,951 条。所有记录均通过 Gold Program 解析与执行、Answer 一致性、三字段结构、Brief 分级长度和 Alpaca 字段完整性检查。

| 检查项 | 结果 |
|---|---:|
| 最终训练记录 | 5,951 |
| Alpaca 四字段完整 | 5,951 / 5,951 |
| 三行 target 完整 | 5,951 / 5,951 |
| 源 Gold 校验通过 | 5,951 / 5,951 |
| Brief 分级长度校验通过 | 5,951 / 5,951 |

数据同时覆盖 1–6 步 Program：

| 操作数 | 记录数 |
|---:|---:|
| 1 | 2,493 |
| 2 | 2,037 |
| 3 | 1,277 |
| 4 | 53 |
| 5 | 73 |
| 6 | 18 |

## LoRA 训练与结果

Qwen3-1.7B 和 Qwen3-8B 使用同一套训练数据、System Prompt 和 `qwen3_nothink` 模板，分别从原始 Qwen3 Base Model 开始训练，没有继承此前的 LoRA 权重。具体参数保留在对应的 [`Qwen3-1.7B/lora_sft.yaml`](./Qwen3-1.7B/lora_sft.yaml) 和 [`Qwen3-8B/lora_sft.yaml`](./Qwen3-8B/lora_sft.yaml) 中。

训练完成后，使用 [`finqa_three_field_eval.py`](../finqa_three_field_eval.py) 在 FinQA test 的 1,147 道题上进行 K=8 外部评测，评测参数设为 `temperature=0.5`、`top_p=1.0`、`enable_thinking=false`。下面同时列出 [Answer-only LoRA](../../01-answer-only-teacher-topk32-opd-baseline/lora/README.md)、此前的 [Brief–Program–Answer LoRA](../../02-structured-three-field-teacher-only-opd/lora/README.md) 与本轮 Trace-Enhanced BPA LoRA 的选用模型结果。`pp` 表示百分点。

### Qwen3-1.7B

| LoRA 监督形式 | Answer `avg@8` | Answer `best@8` | 较上一阶段 `avg@8` |
|---|---:|---:|---:|
| Answer-only | 21.17% | 34.44% | — |
| Brief–Program–Answer | 41.26% | 57.98% | +20.09 pp |
| Trace-Enhanced BPA | **48.812%** | **62.424%** | **+7.552 pp** |

### Qwen3-8B

| LoRA 监督形式 | Answer `avg@8` | Answer `best@8` | 较上一阶段 `avg@8` |
|---|---:|---:|---:|
| Answer-only | 59.20% | 63.73% | — |
| Brief–Program–Answer | 64.71% | 72.71% | +5.51 pp |
| Trace-Enhanced BPA | **70.031%** | **75.937%** | **+5.321 pp** |

从 Answer-only 到 BPA，增加三字段监督已经显著提高了两个模型的 FinQA 能力；本轮进一步将 Brief 从概括性的计算描述重构为包含中间结果及其依赖关系的计算轨迹，并加强 Brief、Program 与 Answer 之间的数值对应后，1.7B 和 8B 的 LoRA 结果都继续提升。

相较上一阶段，Qwen3-1.7B 的 Answer `avg@8` 提高 7.552 个百分点，`best@8` 提高 4.444 个百分点；Qwen3-8B 的 Answer `avg@8` 提高 5.321 个百分点，`best@8` 提高 3.227 个百分点。这说明本轮数据重构带来的收益并不只体现在后续 OPD 阶段。即使尚未进行在线蒸馏，信息密度更高、计算关系更完整的监督目标本身，也更有利于模型学习 FinQA 的解题过程。

本轮所有训练与评测过程仍采用非思考生成方式，1.7B 和 8B 均按照固定的 Brief–Program–Answer 三字段生成结果。在训练数据中引入显式计算轨迹后，两个模型的评测结果均较上一阶段有所提升，进一步支持了通过提高计算信息密度、强化三字段映射来增强非思考模型能力的思路。最终选出的 1.7B 和 8B LoRA 模型随后分别作为 OPD 的 Student 和 Teacher，用于进一步检验 Trace-Enhanced BPA 在在线蒸馏中的实际效果。

## 文件布局

```text
lora/
├── README.md
├── data/
│   ├── dataset_info.json
│   ├── finqa_brief_program_train_C_0826.json
│   └── trace_enhanced_bpa_prompt.txt
├── Qwen3-1.7B/
│   ├── lora_sft.yaml
│   ├── merge_config.yaml
│   ├── nohup_lora_train.sh
│   └── train_mannual.sh
└── Qwen3-8B/
    ├── lora_sft.yaml
    ├── merge_config.yaml
    ├── nohup_lora_train.sh
    └── train_mannual.sh
```

| 文件或目录 | 说明 |
|---|---|
| `data/finqa_brief_program_train_C_0826.json` | 本轮最终使用的 5,951 条 Trace-Enhanced BPA LoRA 训练数据。 |
| `data/trace_enhanced_bpa_prompt.txt` | LoRA 训练与外部评测统一使用的冻结 System Prompt。 |
| `data/dataset_info.json` | LLaMAFactory 数据集注册配置。 |
| `Qwen3-1.7B/`、`Qwen3-8B/` | 两个模型实际使用的 LoRA 训练、权重合并和启动配置。 |
| [`../finqa_three_field_eval.py`](../finqa_three_field_eval.py)、[`../prompt.py`](../prompt.py) | 本轮 LoRA 外部评测使用的脚本和 Prompt 定义。 |

## 与 OPD 的衔接

后续 OPD 直接使用上述 Qwen3-1.7B 和 Qwen3-8B Trace-Enhanced BPA LoRA，分别作为 Student 和 Teacher，继续验证这套 `Brief → Program → Answer` 三阶段映射能否改善 Teacher 到 Student 的蒸馏效果。评估 OPD 收益时，以 Student 的同口径 Answer `avg@8=48.812%` 作为直接基线，不把此前数据和 LoRA 阶段的提升计入 OPD 单独贡献。

本次 GitHub 更新仅公开上述 LoRA 数据、配置与评测结果，后续 OPD 内容不在本次更新范围内。

[返回项目首页](../../README.md)
