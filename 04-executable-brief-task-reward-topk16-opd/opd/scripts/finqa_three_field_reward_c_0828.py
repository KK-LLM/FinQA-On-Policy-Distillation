"""VERL custom reward entry point for FinQA experiment C."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from finqa_three_field_core_c_0828 import evaluate_reward


_TOKENIZER: Any | None = None
_TOKENIZER_LOGGED = False


def preload_tokenizer() -> Any:
    """Load and validate the Qwen3 tokenizer once in each reward worker."""
    global _TOKENIZER, _TOKENIZER_LOGGED
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        tokenizer_path = os.environ.get(
            "FINQA_TOKENIZER_PATH",
            "/root/model/Qwen3-1.7B-LoRA-BPA-C-0826-checkpoint-7920",
        )
        if not Path(tokenizer_path).exists():
            raise FileNotFoundError(
                f"FINQA_TOKENIZER_PATH does not exist: {tokenizer_path}"
            )
        _TOKENIZER = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=True,
            trust_remote_code=False,
        )
    if not _TOKENIZER_LOGGED:
        probe = "1 divided by 2 is 0.5."
        probe_tokens = len(_TOKENIZER.encode(probe, add_special_tokens=False))
        tokenizer_path = os.environ.get(
            "FINQA_TOKENIZER_PATH",
            "/root/model/Qwen3-1.7B-LoRA-BPA-C-0826-checkpoint-7920",
        )
        print(
            "[FinQA Reward] tokenizer="
            f"{tokenizer_path} type={type(_TOKENIZER).__name__} "
            f"probe_tokens={probe_tokens} brief_limits=64/128/192",
            flush=True,
        )
        _TOKENIZER_LOGGED = True
    return _TOKENIZER


def _brief_token_count(text: str) -> int:
    tokenizer = preload_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Return the v2-weighted reward with experiment-C protocol validation."""
    del data_source, extra_info, kwargs
    return evaluate_reward(
        solution_str,
        ground_truth,
        brief_token_counter=_brief_token_count,
    )
