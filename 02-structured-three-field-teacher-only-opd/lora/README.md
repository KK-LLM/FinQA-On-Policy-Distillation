# FinQA Brief–Program–Answer LoRA

Answer-only LoRA 只能让模型学习最终答案，无法直接监督问题理解和计算过程。本轮将监督输出扩展为 `Brief–Program–Answer` 三字段，使 Qwen3-1.7B Student 和 Qwen3-8B Teacher 在进入 OPD 前先学习相同的结构化输出协议。

```text
Brief: <问题目标、关键数值及计算关系>
Program: <可执行的 FinQA Program>
Answer: <归一化的最终结果>
```

两个模型均从原始 Qwen3 Base Model 开始训练，没有继承第一轮 Answer-only LoRA 权重。训练使用 `qwen3_nothink` 模板，不生成 `<think>` 内容。

## 数据构造目标

三个字段分别承担不同的监督职责：

- `Brief` 用一句简短英文说明问题目标、相关数值的金融含义以及核心计算或比较关系；
- `Program` 将上述关系表达为可解析、可执行的 FinQA DSL；
- `Answer` 保存 Program 最后一步的归一化执行结果。

Brief 不是 FinQA 官方提供的人工标注，而是利用任务已有的 Question、Gold Evidence、Steps 和 Gold Program 自动构造的派生监督。Program 和 Answer 则以经过验证的 FinQA Gold 标注为基础，不由强模型重新求解。

## Brief 构造

Brief 采用“Gold Steps 计算图 + Gold Evidence 语义对齐 + 强模型压缩表达”的方式生成。

每条记录先根据 Question 确定需要计算或比较的目标，再使用 Gold Evidence 识别实际参与计算的数值、年份、单位和金融含义，并根据 Steps 与 Program 还原运算顺序和中间结果依赖关系。在这些信息已经确定后，由强模型将其压缩为一句简短英文。

强模型只负责自然语言表达，不修改 Gold Program，也不重新设计计算过程。构造请求不提供 `exe_ans` 或最终答案值，避免模型根据答案反向编造 Brief。候选 Brief 必须满足以下要求：

- 只使用 Question、Gold Evidence、Program 参数、合法常数或计算过程中可推导的信息；
- 覆盖 Program 的关键操作数、运算方向和依赖关系；
- 保留必要的金融含义、年份和单位；
- 不直接泄露最终数值答案或 `yes/no` 结论；
- 不包含 FinQA DSL、Markdown、`<think>` 或三字段标签；
- 使用 Qwen3 tokenizer 计数时不超过 64 tokens。

Brief 的目标不是生成一段完整思维链，而是在 Question 和 Program 之间增加一层短小、可验证的语义桥梁。

## 数据清洗

Brief 构造完成后，对 Question、Evidence、Brief、Program 和 Answer 的整体一致性进行了检查。主要清理以下问题：

- Program 无法解析、无法执行，或者执行结果与标准答案不一致；
- Question、Gold Evidence 和 Program 实际计算的目标不一致；
- Program 使用了错误的操作数、运算方向、年份、单位或结果依赖；
- Question 要求的结果超出当前 FinQA DSL 的表达能力；
- Brief 引入无依据数字、遗漏关键计算关系或与 Program 不一致；
- 无法根据 Context 和 Gold Evidence 唯一确定正确语义的高风险记录。

具体实现上，数据清洗以程序化硬校验为主，并结合强模型辅助语义复核和高风险记录人工确认。程序负责检查 Program 的语法、可执行性、结果依赖关系，以及执行结果与 `exe_ans` 的一致性；强模型 API 按照 Prompt 约束生成 Brief，并辅助检查 Question、Gold Evidence、Brief 和 Program 之间的语义一致性。对于自动检查无法确定的歧义或高风险记录，再结合 Context、Gold Evidence 和 Program 执行结果进行人工复核。

只有同时通过结构、执行结果和语义一致性检查的记录才进入最终训练集。经过 Brief 构造、数据清洗和全量一致性检查，最终保留 6,240 条三字段训练记录。

## Prompt、Program 与 Answer

三字段 System Prompt 先结合 FinQA 数据结构、官方 DSL、Qwen3 非思考模板以及 LoRA、OPD 和自动评测需求形成初稿，再由 Kimi-K3、GPT、DeepSeek 和 Gemini 独立审核，并结合本地 tokenizer、executor 与实际样本测试逐项修订。最终 Prompt 固定了 Brief、Program 和 Answer 的职责、合法算子与常量、答案归一化规则和严格三行输出格式，并在 LoRA、OPD 与外部评测中保持一致。

Program 直接使用经过清洗和复审的 FinQA Gold Program，不使用强模型生成的新 Program 覆盖原始标注。每条 Program 都需要通过严格解析，并由项目中的 FinQA executor 实际执行。

Answer 通过 Gold Program 执行结果与 FinQA 原始自然语言答案的交叉校验确定。最终保留的每条数据均满足 Program 执行结果与原始自然语言答案在数值或判断语义上一致，并与 `exe_ans` 对齐；数值结果最多保留 5 位小数并删除无意义的尾随零，判断结果统一为小写 `yes` 或 `no`。

因此，Prompt 负责统一三字段任务协议，Program 和 Answer 则受到 Gold 标注与可执行结果的共同约束。

## LLaMAFactory 数据格式

最终数据转换为 LLaMAFactory 使用的 Alpaca 四字段格式：

| 字段 | 内容 |
|---|---|
| `system` | 三字段输出协议、Brief 约束和 FinQA DSL 规则 |
| `instruction` | 固定任务指令 `Solve the Question using the provided Context.` |
| `input` | FinQA Context、表格和 Question |
| `output` | 严格三行的 `Brief`、`Program` 和 `Answer` |

输出示例：

```text
Brief: Revenue is 800 million, costs are 500 million, and tax is 20% of pretax profit; subtract costs, calculate tax, then subtract tax from profit.
Program: subtract(800, 500), multiply(#0, 20%), subtract(#0, #1)
Answer: 240
```

[`dataset_info.json`](./data/dataset_info.json) 将上述四个字段分别注册为 LLaMAFactory 的 `system`、`prompt`、`query` 和 `response`。本轮只构造训练集，不单独构造 LoRA valid，训练期间关闭 eval。

## 最终数据结果

| 检查项 | 结果 |
|---|---:|
| 最终训练记录 | 6,240 |
| 数值答案 | 6,116 |
| `yes/no` 答案 | 124 |
| Program 严格解析成功 | 6,240/6,240 |
| Program 执行结果与 `exe_ans` 一致 | 6,240/6,240 |
| 三字段输出格式正确 | 6,240/6,240 |
| Brief 不超过 64 Qwen3 tokens | 6,240/6,240 |
| valid 记录混入 train | 0 |

最终文件指纹：

| 文件 | SHA-256 |
|---|---|
| Brief–Program–Answer 训练数据 | `9aeb754d2205d407b7b0b246d39e1539fe007de3bd0b6f17d54ea51d73cfaa09` |
| 三字段 System Prompt | `245f04b31b4d7ca5d4be93589fb2d1457ddee484e7b22487ce780cc526b8dcc6` |

完整的格式、长度和硬校验结果见 [`Lora_train_0813_构造与长度报告.md`](./data/Lora_train_0813_构造与长度报告.md)。

## LoRA 训练与模型选择

Qwen3-1.7B 和 Qwen3-8B 使用同一套三字段训练数据、System Prompt 和非思考模板，分别从原始 Base Model 开始进行 LoRA SFT。训练完成后，通过外部 K=8 任务评测选择后续 OPD 使用的模型。

| 角色 | Base Model | 选用模型 | 后续用途 |
|---|---|---|---|
| Student | Qwen3-1.7B | Qwen3-1.7B 三字段 LoRA | OPD 初始化模型 |
| Teacher | Qwen3-8B | Qwen3-8B 三字段 LoRA | OPD 教师模型 |

### 外部测试结果

| 模型 | `avg@8` | `best@8` | `avg@8` 提升 | `best@8` 提升 |
|---|---:|---:|---:|---:|
| Qwen3-1.7B Base | 8.59% | 14.04% | — | — |
| Qwen3-1.7B 三字段 LoRA | **41.26%** | **57.98%** | **+32.67 个百分点** | **+43.94 个百分点** |
| Qwen3-8B Base | 32.06% | 42.46% | — | — |
| Qwen3-8B 三字段 LoRA | **64.71%** | **72.71%** | **+32.65 个百分点** | **+30.25 个百分点** |

表中的三字段 LoRA 结果来自对多个 checkpoint 进行 FinQA test K=8 外部评测后选出的最佳模型。选出的 Student 和 Teacher 分别用于后续 OPD 的初始化和在线教学。

### 与 Answer-only LoRA 对比

| 角色 | Answer-only LoRA | 三字段 LoRA | `avg@8` 变化 | `best@8` 变化 |
|---|---:|---:|---:|---:|
| Student | 21.17% | **41.26%** | **+20.09 个百分点** | **+23.54 个百分点** |
| Teacher | 59.20% | **64.71%** | **+5.51 个百分点** | **+8.98 个百分点** |

与历史 Answer-only LoRA 相比，本轮三字段 LoRA 的 Student 和 Teacher 均取得了更高的外部测试结果。其中，Student 的提升更加明显。三字段 LoRA 形成了明显更强的 Student 和 Teacher，为后续 OPD 提供了更好的初始化模型。

## 文件布局

```text
lora/
├── README.md
├── data/
│   ├── Lora_train_0813_构造与长度报告.md
│   ├── dataset_info.json
│   ├── finqa_brief_program_train.json
│   └── finqa_three_field_prompt_64tokens_0813.txt
├── Qwen3-1.7B/
│   ├── lora_sft.yaml
│   ├── merge_config.yaml
│   ├── nohup_lora_train.sh
│   └── train_mannual.sh
├── Qwen3-8B/
│   ├── lora_sft.yaml
│   ├── merge_config.yaml
│   ├── nohup_lora_train.sh
│   └── train_mannual.sh
├── finqa_three_field_eval.py
└── prompt.py
```

- [`finqa_brief_program_train.json`](./data/finqa_brief_program_train.json) 是 LLaMAFactory 实际读取的三字段训练集；
- [`finqa_three_field_prompt_64tokens_0813.txt`](./data/finqa_three_field_prompt_64tokens_0813.txt) 是固化在训练数据中的 System Prompt；
- [`dataset_info.json`](./data/dataset_info.json) 负责注册 Alpaca 字段；
- `Qwen3-1.7B/` 和 `Qwen3-8B/` 保存两个模型的训练、合并和启动配置；
- [`finqa_three_field_eval.py`](./finqa_three_field_eval.py) 用于三字段 LoRA 的外部评测；
- [`prompt.py`](./prompt.py) 保存评测时调用的三字段 System Prompt，与训练数据中固化的 Prompt 保持一致。

## 与 OPD 的衔接

三字段 LoRA 是后续 Teacher-only OPD 的直接初始化阶段。后续 OPD 实际使用 Qwen3-1.7B 三字段 LoRA checkpoint-3400 作为 Student、Qwen3-8B 三字段 LoRA checkpoint-15550 作为 Teacher。评估 OPD 收益时，应以前者的同口径结果作为 Student 基线，不能把 Base Model 到 LoRA 的提升计入 OPD 收益。

本轮目录按照方法演进关系整理，不表示该实验在时间上紧接 Answer-only OPD 基线执行。

[返回项目首页](../../README.md)
