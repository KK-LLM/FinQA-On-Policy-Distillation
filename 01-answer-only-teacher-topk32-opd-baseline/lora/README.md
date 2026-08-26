# FinQA Answer-only LoRA

Base Model 直接 OPD 虽然能够提升 Qwen3-1.7B 的 FinQA 表现，但 Student 和 Teacher 的绝对正确率仍然较低。为增强两个模型对 FinQA 任务形式和答案分布的适应能力，Qwen3-1.7B 与 Qwen3-8B 分别进行了 Answer-only LoRA 训练，再作为后续 OPD 的 Student 和 Teacher。

## 训练目标

LoRA 阶段采用监督微调（Supervised Fine-Tuning，SFT），输入为 FinQA 财务上下文与问题，输出为归一化后的最终答案。

- 数值问题只输出数字。
- 判断问题只输出 `yes` 或 `no`。
- 不输出分析过程、Brief、Program 或其他解释。
- Qwen3 使用 `qwen3_nothink` 模板，关闭思考模式。

训练指令如下：

```text
You are a financial question-answering assistant. Answer the user question using only the provided context. Output only the final answer exactly as a number or yes/no, with no explanation or intermediate reasoning.
```

## 数据

| 文件 | 样本数 | 用途 |
|---|---:|---|
| `data/finqa_lora_train.json` | 6,251 | LLaMAFactory 训练集 |
| `data/finqa_lora_valid.json` | 883 | LLaMAFactory 验证集 |
| `data/test.jsonl` | 1,147 | 外部 K=8 测试集 |
| `data/dataset_info.json` | — | LLaMAFactory 数据集注册配置 |

训练集和验证集采用 Alpaca 格式：

| 字段 | 内容 |
|---|---|
| `instruction` | Answer-only 系统指令 |
| `input` | FinQA 财务上下文、表格和问题 |
| `output` | 归一化后的标准答案 |

外部测试数据使用 `query`、`answer`、`id` 和 `text` 字段。FinQA 官方原始数据位于仓库根目录的 [`data/`](../../../data/)，本目录只保留实际交给 LoRA 训练和评测程序的输入格式。

## LoRA 任务适配

Base Model 直接 OPD 能够改善 Qwen3-1.7B 的 FinQA 表现，但 Student 和 Teacher 对任务本身的适应能力仍然有限。因此，在后续 OPD 之前，先对 Qwen3-1.7B 和 Qwen3-8B 进行 Answer-only LoRA 训练，使两个模型学习 FinQA 的输入形式和答案分布。

两组训练采用相同的任务设定：

- 使用相同的 FinQA 训练集和验证集；
- 输入均为财务上下文、表格和问题；
- 输出均为归一化后的数字或 `yes/no`；
- 使用 `qwen3_nothink` 模板，不生成思考过程；
- LoRA rank 为 64，LoRA alpha 为 128；
- 最大序列长度为 3,072；
- LoRA 合并后使用相同的 K=8 评测方法进行 checkpoint 对比。

Qwen3-1.7B 和 Qwen3-8B 分别面向 Student 初始化和 Teacher 构建进行训练。两组训练独立选择优化参数和 checkpoint，其结果用于确定后续 OPD 的初始模型。

完整训练配置见：

- [Qwen3-1.7B LoRA 配置](./Qwen3-1.7B/lora_sft.yaml)
- [Qwen3-8B LoRA 配置](./Qwen3-8B/lora_sft.yaml)

## 文件布局

```text
lora/
├── README.md
├── data/
│   ├── dataset_info.json
│   ├── finqa_lora_train.json
│   ├── finqa_lora_valid.json
│   └── test.jsonl
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

训练配置中的 `dataset_dir` 固定为 `/root/lora_config`。在原始容器布局中，相关文件按以下方式放置：

```text
/root/lora_config/
├── dataset_info.json
├── finqa_lora_train.json
├── finqa_lora_valid.json
├── Qwen3-1.7B/
└── Qwen3-8B/
```

仓库中的 `data/` 对应上述目录中的 3 个数据配置文件。若使用其他路径，需要同步修改 YAML、启动脚本、合并配置和评测脚本中的绝对路径。

## 训练与合并

启动 Qwen3-1.7B LoRA：

```bash
bash Qwen3-1.7B/train_mannual.sh
```

启动 Qwen3-8B LoRA：

```bash
bash Qwen3-8B/train_mannual.sh
```

训练完成后，使用对应的 `merge_config.yaml` 合并 LoRA adapter：

```bash
llamafactory-cli export Qwen3-1.7B/merge_config.yaml
llamafactory-cli export Qwen3-8B/merge_config.yaml
```

实际用于后续 OPD 的模型为：

| 角色 | LoRA checkpoint | 合并输出目录 |
|---|---:|---|
| Student | Qwen3-1.7B checkpoint-1400 | `/root/model/Qwen3-1.7B-Lora` |
| Teacher | Qwen3-8B checkpoint-5750 | `/root/model/Qwen3-8B-Lora` |

checkpoint-800 的 `avg@8` 低于 checkpoint-1400，`best@8` 高于 checkpoint-1400，但它不是本轮 OPD 实际使用的 Student 初始化模型。

## 外部评测

实验根目录下的 [`finqa_answer_only_eval.py`](../finqa_answer_only_eval.py) 由本轮 LoRA 与后续 OPD 共用，通过 OpenAI-compatible Chat Completions 接口评测合并后的模型。两阶段使用相同的 Answer-only System Prompt、`temperature=0.5` 和 K=8 测试口径。脚本默认连接 `http://127.0.0.1:6006/v1/chat/completions`，运行前需要先启动对应模型的推理服务，并根据待测模型调整脚本顶部的 `MODEL_ID`、`TEST_FILE` 和 `OUTPUT_DIR`。

主要评测参数如下：

| 参数 | 值 |
|---|---:|
| 测试题数 | 1,147 |
| 每题采样次数 | 8 |
| Temperature | 0.5 |
| Top-p | 1.0 |
| Max tokens | 128 |
| 并发数 | 150 |
| 单次请求超时 | 120 秒 |
| 失败重试 | 2 次 |
| Thinking | 关闭 |

运行评测：

```bash
python3 ../finqa_answer_only_eval.py
```

评测同时输出两项指标：

- `avg@8`：全部 9,176 次生成中的正确比例。
- `best@8`：1,147 道题中至少有 1 次生成正确的题目比例。

## Student 测试结果与选择

Qwen3-1.7B Base、LoRA checkpoint-800 和 LoRA checkpoint-1400 均在 FinQA test 上进行 K=8 评测：

| 模型 | `avg@8` | `best@8` | 与 OPD 的关系 |
|---|---:|---:|---|
| Qwen3-1.7B Base | 3.48% | 4.97% | LoRA 前的 Base Model 参考点 |
| Qwen3-1.7B LoRA checkpoint-800 | 20.73% | **35.92%** | LoRA checkpoint 对比 |
| Qwen3-1.7B LoRA checkpoint-1400 | **21.17%** | 34.44% | 后续 OPD 实际使用的 Student 初始化模型 |

两个 LoRA checkpoint 都明显优于 Qwen3-1.7B Base，说明 Answer-only LoRA 已经显著增强了 Student 对 FinQA 任务的适应能力。checkpoint-1400 相比 checkpoint-800 的 `avg@8` 高 0.44 个百分点，但 `best@8` 低 1.48 个百分点，两个 checkpoint 在不同指标上各有优势。

后续 OPD 实际使用 checkpoint-1400 初始化 Student，因此评估 OPD 的直接增益时以 checkpoint-1400 为基线。checkpoint-800 没有参与后续 OPD 训练，只用于比较 OPD 模型与其他 LoRA checkpoint 的表现。

## Teacher 测试结果与选择

Qwen3-8B Base、LoRA checkpoint-5750 和 LoRA checkpoint-6256 均在 FinQA test 上进行 K=8 评测：

| 模型 | 评测数据 | `avg@8` | `best@8` | 说明 |
|---|---|---:|---:|---|
| Qwen3-8B Base | FinQA test | 9.51% | 12.73% | LoRA 前的 Base Model 参考点 |
| Qwen3-8B LoRA checkpoint-5750 | FinQA test | **59.20%** | **63.73%** | 后续 OPD 实际使用的 Teacher |
| Qwen3-8B LoRA checkpoint-6256 | FinQA test | 58.46% | 63.56% | LoRA checkpoint 对比 |

两个 LoRA checkpoint 的结果接近，并且都明显优于 Qwen3-8B Base。checkpoint-5750 相比 checkpoint-6256 的 `avg@8` 高 0.74 个百分点，`best@8` 高 0.17 个百分点。

后续 OPD 实际使用 checkpoint-5750 作为 Teacher，因此相关实验分析以 checkpoint-5750 的测试结果作为 Teacher 基线。checkpoint-6256 只用于比较 LoRA 训练过程中不同 checkpoint 的表现，没有参与后续 OPD 训练。

## 与 OPD 的衔接

LoRA 阶段本身是监督微调，不涉及 Policy Gradient。合并后的 Qwen3-1.7B checkpoint-1400 和 Qwen3-8B checkpoint-5750 用于 Answer-only Teacher TopK32 OPD 基线；该 OPD 配置直接反向传播 TopK32 Forward KL，不使用 Policy Gradient，FinQA 任务奖励也不参与参数更新。

[返回项目首页](../../../README.md)
