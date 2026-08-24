"""FinQA validation score aligned with the external avg@8 evaluator."""

from __future__ import annotations

import math
import re


_NUMBER_RE = re.compile(
    r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
)
_EVAL_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*%?")
_ABS_TOL = 1e-3
_REL_TOL = 1e-3


def _parse_answer(answer):
    """Return a typed, finite answer token, or None when it is invalid."""
    if not isinstance(answer, str):
        return None

    token = answer.strip()
    lowered = token.lower()
    if lowered == "yes":
        return "boolean", True
    if lowered == "no":
        return "boolean", False
    if not _NUMBER_RE.fullmatch(token):
        return None

    value = float(token.replace(",", ""))
    if not math.isfinite(value):
        return None
    return "number", value


def _clean_answer(text: str) -> str:
    value = str(text).strip()
    value = re.split(r"[。；;]\s*", value)[0].strip()
    value = value.strip("`'\" ")
    value = value.rstrip(".。;；")
    value = re.sub(r"^\$+", "", value).strip()
    return re.sub(r"\s+", " ", value)


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    for pattern in (
        r"Final\s*Answer\s*[:：]\s*(.+)",
        r"Answer\s*[:：]\s*(.+)",
        r"答案\s*[:：]\s*(.+)",
    ):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return _clean_answer(matches[-1])
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return _clean_answer(lines[-1]) if lines else ""


def _normalize_text_answer(text: str) -> str:
    value = _clean_answer(text).lower()
    value = value.replace("％", "%").replace(",", "").replace("$", "")
    return re.sub(r"\s+", " ", value).strip()


def _numeric_candidates(text: str) -> set[float]:
    candidates: set[float] = set()
    for raw in _EVAL_NUMBER_RE.findall(_normalize_text_answer(text)):
        token = raw.strip()
        is_percent = token.endswith("%")
        token = token.rstrip("%")
        try:
            number = float(token)
        except ValueError:
            continue
        candidates.add(number)
        if is_percent:
            candidates.add(number / 100.0)
    return candidates


def _compare_like_external_evaluator(prediction: str, gold: str) -> bool:
    pred_norm = _normalize_text_answer(prediction)
    gold_norm = _normalize_text_answer(gold)
    if not pred_norm:
        return False
    if pred_norm == gold_norm:
        return True
    yes_no = {"yes", "no"}
    if pred_norm in yes_no or gold_norm in yes_no:
        return pred_norm == gold_norm
    return any(
        math.isclose(pred_value, gold_value, rel_tol=_REL_TOL, abs_tol=_ABS_TOL)
        for pred_value in _numeric_candidates(prediction)
        for gold_value in _numeric_candidates(gold)
    )


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Return external-eval correctness plus a separate strict-format diagnostic."""
    raw_solution = str(solution_str or "")
    prediction = _extract_final_answer(raw_solution)
    answer_correct = float(
        _compare_like_external_evaluator(prediction, str(ground_truth or ""))
    )
    format_ok = float(_parse_answer(raw_solution) is not None)
    return {
        "score": answer_correct,
        "format_ok": format_ok,
        "answer_correct": answer_correct,
    }
