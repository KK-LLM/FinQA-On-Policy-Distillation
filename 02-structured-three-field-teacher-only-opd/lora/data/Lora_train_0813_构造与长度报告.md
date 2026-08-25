# FinQA 三字段 LoRA train 0813 构造与长度报告

构造日期：2026-08-13

## 构造范围

- 源数据：`data/train_clean_final_v4_with_brief.json`
- 输出数据：`Lora/Lora_data_0813/finqa_brief_program_train.json`
- 训练记录：6,240 条
- 数值答案：6,116 条
- `yes/no` 答案：124 条
- valid：按本轮决定不构造、不注册，也不参与 LoRA 训练过程中的验证

每条数据均为 Alpaca 四字段 `system/instruction/input/output`；`output` 严格由 `Brief`、`Program`、`Answer` 三行组成。Answer 来自重新执行 Gold Program 后的标准化结果，并再次与 `exe_ans` 比较。

为适配 `cutoff_len: 3000`，仅对原始完整训练序列超过 3,000 tokens 的 83 条记录压缩 `input` 中的 Context；其余 6,157 条记录与基础转换结果逐字段不变。压缩 Context 保留原 `model_input` 检索证据、全部 Gold Evidence 和涉及的精确表格行，Question 与三字段 target 不变。

## 硬校验结果

| 检查项 | 结果 |
|---|---:|
| Program 严格解析成功 | 6,240/6,240 |
| `execute(program) == exe_ans` | 6,240/6,240 |
| 三字段输出格式正确 | 6,240/6,240 |
| Brief 不超过 64 Qwen3 tokens | 6,240/6,240 |
| 原始超过 3,000 tokens、已压缩 Context | 83 |
| 压缩后仍超过 3,000 tokens | 0 |
| valid 记录混入 train | 0 |

## Qwen3-1.7B tokenizer 长度

长度通过 `qwen3_nothink` 对应的 Qwen3 chat template 统计，完整序列包含 system、user 和 assistant target。

| 范围 | P50 | P90 | P95 | P99 | 最大值 |
|---|---:|---:|---:|---:|---:|
| Brief | 45 | 56 | 59 | 63 | 64 |
| 完整 response | 74 | 97 | 103 | 122 | 151 |
| 完整训练序列 | 1967 | 2389 | 2494 | 2797 | 2994 |

- 压缩前超过 3,000 tokens：83 条
- 压缩后超过 3,000 tokens：0 条
- 压缩前最大长度：4248 tokens
- 压缩后最大长度：2994 tokens
- 当前数据已适配 `cutoff_len: 3000`。

## 文件指纹

- 源数据 SHA256：`7a830af8a306f78efe4d8001d116c2b905ddaf06749137b7523e530ded0752d3`
- 固化 Prompt SHA256：`245f04b31b4d7ca5d4be93589fb2d1457ddee484e7b22487ce780cc526b8dcc6`
- 输出 train SHA256：`9aeb754d2205d407b7b0b246d39e1539fe007de3bd0b6f17d54ea51d73cfaa09`

源数据和原始 48-token Prompt 均未被覆盖；本轮在输出目录内固化了一份仅将 Brief 上限改为 64 tokens 的 Prompt。
