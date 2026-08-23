from __future__ import annotations

import asyncio
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import requests


HEADERS = {"Content-Type": "application/json"}
MODEL_URL = "http://127.0.0.1:6006/v1/chat/completions"
# MODEL_ID = "qwen3-1p7B"
# MODEL_ID = "qwen3-8B-LoRA"
MODEL_ID = "qwen3-1.7B-LoRA-1400"

# TEST_FILE = Path("/root/flare_finqa/train.jsonl")
TEST_FILE = Path("/root/flare_finqa/test.jsonl")
# OUTPUT_DIR = Path(f"/root/finqa_single_model_train_optimized_0713/qwen3_8b_correct_{MODEL_ID}")
OUTPUT_DIR = Path(f"/root/finqa_single_model_train_optimized_0806/{MODEL_ID}")

CONCURRENCY = 150
CALLS_PER_SAMPLE = 8
REQUEST_TIMEOUT = 120
MAX_RETRIES = 2
MAX_TOKENS = 128
TEMPERATURE = 0.5
TOP_P = 1.0
TEST_LIMIT: int | None = None

PREDICTIONS_FILE = OUTPUT_DIR / "finqa_predictions.jsonl"
SUMMARY_FILE = OUTPUT_DIR / "finqa_summary.json"

SYSTEM_PROMPT = (
    "You are a financial question-answering assistant. "
    "Answer the user question using only the provided context. "
    "Output only the final answer exactly as a number or yes/no, "
    "with no explanation or intermediate reasoning."
)
# SYSTEM_PROMPT = (
#     "You are a careful financial question-answering assistant. "
#     "Answer the question using the given context. "
#     "Keep the reasoning concise and end with exactly one line: Final Answer: <answer>."
# )

ABS_TOL = 1e-3
REL_TOL = 1e-3


def load_test_data(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows[:TEST_LIMIT] if TEST_LIMIT is not None else rows


def build_request_body(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [
            {
                "role": "user",
                "content": f"{SYSTEM_PROMPT}\n{sample['query']}",
            },
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }


def call_model_once(sample: dict[str, Any]) -> tuple[str, float, str | None]:
    body = build_request_body(sample)
    start = time.perf_counter()
    last_error: str | None = None

    for attempt in range(1, MAX_RETRIES + 2):
        try:
            response = requests.post(
                MODEL_URL,
                headers=HEADERS,
                json=body,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
            return str(content), time.perf_counter() - start, None
        except Exception as exc:
            last_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            if attempt <= MAX_RETRIES:
                time.sleep(min(2**attempt, 8))

    return "", time.perf_counter() - start, last_error


async def call_model_attempt(
    sample: dict[str, Any],
    semaphore: asyncio.Semaphore,
    call_index: int,
) -> dict[str, Any]:
    async with semaphore:
        output, latency_sec, error = await asyncio.to_thread(call_model_once, sample)

    prediction = extract_final_answer(output)
    correct, match_type = compare_answer(prediction, str(sample["answer"]))
    if error:
        correct = False
        match_type = "request_error"
    return {
        "call_index": call_index,
        "prediction": prediction,
        "valid_prediction": is_valid_prediction(prediction),
        "correct": correct,
        "match_type": match_type,
        "latency_sec": round(latency_sec, 4),
        "error": error,
        "output": output,
    }


async def call_model(sample: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    start = time.perf_counter()
    attempts = await asyncio.gather(
        *[
            call_model_attempt(sample, semaphore, call_index)
            for call_index in range(1, CALLS_PER_SAMPLE + 1)
        ]
    )

    correct_attempts = [attempt for attempt in attempts if attempt["correct"]]
    valid_attempts = [
        attempt
        for attempt in attempts
        if attempt["error"] is None and attempt["valid_prediction"]
    ]
    successful_attempts = [attempt for attempt in attempts if attempt["error"] is None]

    if correct_attempts:
        selected = correct_attempts[0]
        selection_method = "first_correct_best_at_k"
    elif valid_attempts:
        selected = valid_attempts[0]
        selection_method = "first_valid_no_correct"
    elif successful_attempts:
        selected = successful_attempts[0]
        selection_method = "first_successful_no_correct"
    else:
        selected = attempts[0]
        selection_method = "all_calls_failed"

    correct_count = len(correct_attempts)
    avg_at_k = correct_count / len(attempts)
    best_at_k = int(correct_count > 0)
    successful_calls = len(successful_attempts)
    call_errors = len(attempts) - successful_calls
    error = "all_calls_failed" if successful_calls == 0 else None

    return {
        "id": str(sample.get("id", "")),
        "question": str(sample.get("text", "")),
        "gold": str(sample.get("answer", "")),
        "prediction": str(selected["prediction"]),
        "correct": bool(best_at_k),
        "correct_count": correct_count,
        "avg_at_k": round(avg_at_k, 6),
        "best_at_k": best_at_k,
        "k": len(attempts),
        "match_type": str(selected["match_type"]),
        "latency_sec": round(time.perf_counter() - start, 4),
        "call_count": len(attempts),
        "successful_calls": successful_calls,
        "call_errors": call_errors,
        "valid_prediction_calls": len(valid_attempts),
        "selection_method": selection_method,
        "selected_call_index": selected["call_index"],
        "error": error,
        "output": selected["output"],
        "calls": attempts,
    }


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"Final\s*Answer\s*[:：]\s*(.+)",
        r"Answer\s*[:：]\s*(.+)",
        r"答案\s*[:：]\s*(.+)",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return clean_answer(matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return clean_answer(lines[-1]) if lines else ""


def clean_answer(text: str) -> str:
    value = str(text).strip()
    value = re.split(r"[。；;]\s*", value)[0].strip()
    value = value.strip("`'\" ")
    value = value.rstrip(".。;；")
    value = re.sub(r"^\$+", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def normalize_text_answer(text: str) -> str:
    value = clean_answer(text).lower()
    value = value.replace("％", "%")
    value = value.replace(",", "")
    value = value.replace("$", "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def numeric_candidates(text: str) -> set[float]:
    value = normalize_text_answer(text)
    candidates: set[float] = set()

    for raw in re.findall(r"[-+]?\d+(?:\.\d+)?\s*%?", value):
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


def compare_answer(prediction: str, gold: str) -> tuple[bool, str]:
    pred_norm = normalize_text_answer(prediction)
    gold_norm = normalize_text_answer(gold)
    if not pred_norm:
        return False, "empty_prediction"

    if pred_norm == gold_norm:
        return True, "exact"

    yes_no = {"yes", "no"}
    if pred_norm in yes_no or gold_norm in yes_no:
        return (pred_norm == gold_norm), "yes_no" if pred_norm == gold_norm else "failed"

    pred_numbers = numeric_candidates(prediction)
    gold_numbers = numeric_candidates(gold)
    for pred_value in pred_numbers:
        for gold_value in gold_numbers:
            if math.isclose(pred_value, gold_value, rel_tol=REL_TOL, abs_tol=ABS_TOL):
                return True, "numeric"

    return False, "failed"


def is_valid_prediction(prediction: str) -> bool:
    normalized = normalize_text_answer(prediction)
    return normalized in {"yes", "no"} or bool(numeric_candidates(prediction))


def update_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["total"] += 1
    summary["correct"] += int(record["best_at_k"])
    summary["correct_calls"] += int(record["correct_count"])
    summary["errors"] += int(bool(record["error"]))
    summary["request_calls"] += int(record["call_count"])
    summary["request_errors"] += int(record["call_errors"])
    summary["latency_sum"] += float(record["latency_sec"])
    summary["match_types"][record["match_type"]] = summary["match_types"].get(record["match_type"], 0) + 1


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    total = max(int(summary["total"]), 1)
    total_calls = total * CALLS_PER_SAMPLE
    summary["k"] = CALLS_PER_SAMPLE
    summary["avg_at_k"] = round(summary["correct_calls"] / total_calls, 6)
    summary["best_at_k"] = round(summary["correct"] / total, 6)
    summary["accuracy"] = summary["best_at_k"]
    summary["accuracy_definition"] = "best_at_k"
    summary["avg_latency_sec"] = round(summary["latency_sum"] / total, 4)
    del summary["latency_sum"]
    return summary


def print_record(record: dict[str, Any], completed: int, total: int) -> None:
    status = "✓" if record["correct"] else "✗"
    print(f"\n[{completed}/{total}] id={record['id']} {status} (K={record['k']})")
    print(f"  gold: {record['gold']!r}")
    print(f"  correct_count: {record['correct_count']}/{record['k']}")
    print(f"  avg@k: {record['avg_at_k']:.4f} | best@k: {record['best_at_k']}")
    if record["error"]:
        print(f"  error: {record['error']}")


async def main() -> None:
    if CALLS_PER_SAMPLE < 1:
        raise ValueError("CALLS_PER_SAMPLE 必须大于等于 1")
    if CONCURRENCY < 1:
        raise ValueError("CONCURRENCY 必须大于等于 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_test_data(TEST_FILE)
    semaphore = asyncio.Semaphore(CONCURRENCY)
    summary: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_url": MODEL_URL,
        "test_file": str(TEST_FILE),
        "total": 0,
        "correct": 0,
        "correct_calls": 0,
        "errors": 0,
        "request_calls": 0,
        "request_errors": 0,
        "latency_sum": 0.0,
        "match_types": {},
    }

    print(f"Loaded {len(samples)} FinQA test samples from {TEST_FILE}")
    print(f"Model: {MODEL_ID}")
    print(f"Concurrency: {CONCURRENCY}")
    print(f"Calls per sample: {CALLS_PER_SAMPLE}")

    tasks = [asyncio.create_task(call_model(sample, semaphore)) for sample in samples]
    with PREDICTIONS_FILE.open("w", encoding="utf-8") as out:
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            record = await task
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            update_summary(summary, record)
            print_record(record, completed, len(samples))

    summary = finalize_summary(summary)
    SUMMARY_FILE.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n===== Final Results (best_at_k, K={CALLS_PER_SAMPLE}) =====")
    print(f"model: {MODEL_ID}")
    print(f"avg@{CALLS_PER_SAMPLE}: {summary['avg_at_k']:.4f}")
    print(f"best@{CALLS_PER_SAMPLE}: {summary['best_at_k']:.4f}")
    print(f"total_questions: {summary['total']}")
    print(f"errors: {summary['errors']}")
    print(f"request_errors: {summary['request_errors']}/{summary['request_calls']}")
    print(f"avg_latency_sec: {summary['avg_latency_sec']}")
    print(f"predictions: {PREDICTIONS_FILE}")
    print(f"summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    asyncio.run(main())


