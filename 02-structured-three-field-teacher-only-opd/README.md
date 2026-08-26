# Brief–Program–Answer LoRA + Teacher-only TopK32 OPD 实验

本目录保存 Brief–Program–Answer 三字段 LoRA 与 Teacher-only TopK32 OPD 实验的相关数据、配置和评测内容。

## 目录导航

- [LoRA](./lora/)：Student 与 Teacher 的三字段 LoRA 数据构造、训练配置和评测说明。
- [OPD](./opd/)：Teacher-only TopK32 OPD 的训练数据、运行脚本、实验结果和结果分析。
- [`finqa_three_field_eval.py`](./finqa_three_field_eval.py)：三字段 LoRA 与 OPD 共用的 FinQA K=8 外部评测脚本。
- [`prompt.py`](./prompt.py)：LoRA 与 OPD 外部评测共用的三字段 System Prompt。

项目背景与整体实验结果见 [项目主页](../README.md)。
