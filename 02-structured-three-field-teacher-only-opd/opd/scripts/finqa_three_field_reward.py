"""VERL custom reward entry point for FinQA Brief + Program + Answer."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from finqa_three_field_core import evaluate_reward


_TOKENIZER: Any | None = None


def _brief_token_count(text: str) -> int:
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        tokenizer_path = os.environ.get(
            "FINQA_TOKENIZER_PATH",
            "/root/model/Qwen3-1.7B-LoRA-BPA-0813-ckp-3400",
        )
        _TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )
    return len(_TOKENIZER.encode(text, add_special_tokens=False))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Return the weighted task reward and numeric component diagnostics."""
    del data_source, extra_info, kwargs
    return evaluate_reward(
        solution_str,
        ground_truth,
        brief_token_counter=_brief_token_count,
    )
