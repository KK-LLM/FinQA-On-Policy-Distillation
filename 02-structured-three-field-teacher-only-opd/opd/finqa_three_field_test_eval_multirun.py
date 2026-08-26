from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests


PROMPT_ROOT = Path("/root")
if str(PROMPT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROMPT_ROOT))

from prompt import prompt


HEADERS = {"Content-Type": "application/json"}
MODEL_URL = "http://127.0.0.1:6006/v1/chat/completions"
# MODEL_ID = "Qwen3-8B-LoRA-BPA-0813-ckp-15550"
# MODEL_ID = "Qwen3-1.7B-LoRA-BPA-0813"
MODEL_ID = "qwen3-1.7B-opd-coef_1-108"

PROMPT_SOURCE = "/root/prompt.py"
TEST_FILE = Path("/root/flare_finqa/test.json")
OUTPUT_DIR = Path(f"/root/finqa_opd_eval_0816/{MODEL_ID}")
TEST_RUNS = 3
RUNS_DIR = OUTPUT_DIR / "runs"
RESULTS_FILE = OUTPUT_DIR / "test_runs_summary.jsonl"

CONCURRENCY = 130
CALLS_PER_SAMPLE = 8
REQUEST_TIMEOUT = 300
MAX_RETRIES = 2
MAX_TOKENS = 512
TEMPERATURE = 0.5
TOP_P = 1.0
TEST_LIMIT: int | None = None
WRITE_BATCH_SIZE = 500

INSTRUCTION = "Solve the Question using the provided Context."

ABS_TOL = 1e-3
REL_TOL = 1e-3

OPS = {
    "add",
    "subtract",
    "multiply",
    "divide",
    "exp",
    "greater",
    "table_max",
    "table_min",
    "table_sum",
    "table_average",
}
ALLOWED_CONSTANTS = {
    "const_m1",
    "const_1",
    "const_2",
    "const_3",
    "const_4",
    "const_5",
    "const_6",
    "const_7",
    "const_8",
    "const_9",
    "const_10",
    "const_100",
    "const_1000",
    "const_10000",
    "const_100000",
    "const_1000000",
    "const_1000000000",
}
NUMBER_RE = re.compile(r"^-?(?:\d+(?:\.\d*)?|\.\d+)%?$")
REF_RE = re.compile(r"^#(\d+)$")
SCALAR_OUTPUT_RE = re.compile(
    r"^\s*\$?\s*[-+]?(?:\d[\d,]*(?:\.\d*)?|\.\d+)\s*%?\s*$"
)
PROGRAM_DSL_RE = re.compile(
    r"\b(?:add|subtract|multiply|divide|exp|greater|"
    r"table_max|table_min|table_sum|table_average)\s*\("
)
NORMALIZED_ANSWER_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d{1,5})?$")


def normalize_system_prompt(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError("prompt imported from /root/prompt.py must be a string")
    normalized = value.strip()
    if "Return exactly these three non-empty lines" not in normalized:
        raise ValueError("imported prompt is not the three-field FinQA prompt")
    if "In at most 64 tokens" not in normalized:
        raise ValueError("imported prompt does not contain the 64-token Brief rule")
    return normalized


SYSTEM_PROMPT = normalize_system_prompt(prompt)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def render_table(table: list[list[Any]]) -> str:
    return "\n".join(
        "|" + "|".join(str(cell).replace("|", "\\|") for cell in row) + "|"
        for row in table
    )


def render_context(record: dict[str, Any]) -> str:
    parts = [
        " ".join(record.get("pre_text", [])).strip(),
        render_table(record.get("table", [])),
        " ".join(record.get("post_text", [])).strip(),
    ]
    return "\n".join(part for part in parts if part)


def build_user_content(sample: dict[str, Any]) -> str:
    question = str(sample["qa"]["question"]).strip()
    return (
        f"{INSTRUCTION}\n\n"
        f"Context:\n{render_context(sample)}\n"
        f"Question: {question}"
    )


def load_test_data(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON array in {path}")

    required_qa_fields = {"question", "program", "exe_ans"}
    for index, sample in enumerate(data):
        if not isinstance(sample, dict) or not isinstance(sample.get("qa"), dict):
            raise ValueError(f"record {index} is missing the qa object")
        missing = required_qa_fields - set(sample["qa"])
        if missing:
            raise ValueError(f"record {index} is missing qa fields: {sorted(missing)}")
        if not isinstance(sample.get("table"), list):
            raise ValueError(f"record {index} is missing the structured table")

    return data[:TEST_LIMIT] if TEST_LIMIT is not None else data


def build_request_body(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_content(sample)},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }


def clean_answer(text: str) -> str:
    value = str(text).strip()
    value = re.split(r"[。；;]\s*", value)[0].strip()
    value = value.strip("`'\" ")
    value = value.rstrip(".。;；")
    value = re.sub(r"^\$+", "", value).strip()
    value = re.sub(r"\s+", " ", value)
    return value


def extract_final_answer(text: str) -> str:
    if not text:
        return ""

    patterns = [
        r"^Final\s*Answer\s*[:：]\s*(.+?)\s*$",
        r"^Answer\s*[:：]\s*(.+?)\s*$",
        r"^答案\s*[:：]\s*(.+?)\s*$",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if matches:
            return clean_answer(matches[-1])

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    last_line = lines[-1]
    if last_line.lower() in {"yes", "no"} or SCALAR_OUTPUT_RE.fullmatch(last_line):
        return clean_answer(last_line)
    return ""


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
            if math.isclose(
                pred_value,
                gold_value,
                rel_tol=REL_TOL,
                abs_tol=ABS_TOL,
            ):
                return True, "numeric"

    return False, "failed"


def is_valid_prediction(prediction: str) -> bool:
    normalized = normalize_text_answer(prediction)
    return normalized in {"yes", "no"} or bool(numeric_candidates(prediction))


def _extract_labeled_field(text: str, label: str) -> str:
    matches = re.findall(
        rf"^{re.escape(label)}:[ \t]*(.*?)\s*$",
        text,
        flags=re.MULTILINE,
    )
    return matches[-1].strip() if matches else ""


def parse_three_field_output(text: str) -> tuple[dict[str, str], list[str]]:
    fields = {
        "brief": _extract_labeled_field(text, "Brief"),
        "program": _extract_labeled_field(text, "Program"),
        "answer": _extract_labeled_field(text, "Answer"),
    }
    errors: list[str] = []
    lines = text.strip().splitlines() if text.strip() else []
    if len(lines) != 3:
        errors.append("line_count_not_three")

    expected = (("Brief: ", "brief"), ("Program: ", "program"), ("Answer: ", "answer"))
    for index, (prefix, field_name) in enumerate(expected):
        if index >= len(lines) or not lines[index].startswith(prefix):
            errors.append(f"invalid_{field_name}_line")
        elif not lines[index][len(prefix) :].strip():
            errors.append(f"empty_{field_name}")

    return fields, errors


def validate_brief_format(brief: str) -> list[str]:
    errors: list[str] = []
    if not brief:
        return ["missing_brief"]
    if "<think>" in brief.lower() or "</think>" in brief.lower():
        errors.append("contains_think_tag")
    if re.search(r"\b(?:Brief|Program|Answer):", brief, flags=re.IGNORECASE):
        errors.append("contains_field_label")
    if PROGRAM_DSL_RE.search(brief):
        errors.append("contains_program_dsl")
    return errors


def validate_answer_format(answer: str) -> list[str]:
    if not answer:
        return ["missing_answer"]
    if answer in {"yes", "no"}:
        return []
    if not NORMALIZED_ANSWER_RE.fullmatch(answer):
        return ["invalid_normalized_answer"]
    if answer == "-0" or ("." in answer and answer.endswith("0")):
        return ["invalid_normalized_answer"]
    return []


def _split_operations(program: str) -> list[str]:
    operations: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(program):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError(f"unbalanced Program parentheses: {program}")
        elif char == "," and depth == 0:
            operations.append(program[start:index].strip())
            start = index + 1
    if depth != 0:
        raise ValueError(f"unbalanced Program parentheses: {program}")
    operations.append(program[start:].strip())
    if any(not operation for operation in operations):
        raise ValueError(f"empty Program operation: {program}")
    return operations


def parse_program(program: str, *, strict_table: bool = False) -> list[tuple[str, str, str]]:
    parsed: list[tuple[str, str, str]] = []
    for operation in _split_operations(program):
        open_index = operation.find("(")
        if open_index <= 0 or not operation.endswith(")"):
            raise ValueError(f"invalid Program operation: {operation}")
        op = operation[:open_index].strip()
        if op not in OPS:
            raise ValueError(f"unsupported FinQA operator: {op}")
        arguments = operation[open_index + 1 : -1]
        if op.startswith("table_"):
            if "," not in arguments:
                raise ValueError(f"table operator must have two arguments: {operation}")
            arg1, arg2 = [part.strip() for part in arguments.rsplit(",", 1)]
            if strict_table and arg2 != "none":
                raise ValueError(f"table operator must end with ', none': {operation}")
        else:
            parts = [part.strip() for part in arguments.split(",")]
            if len(parts) != 2:
                raise ValueError(f"operator must have two arguments: {operation}")
            arg1, arg2 = parts
        parsed.append((op, arg1.strip(), arg2.strip()))
    return parsed


def validate_program_syntax(
    program: str,
    *,
    strict_table: bool = False,
) -> list[tuple[str, str, str]]:
    parsed = parse_program(program, strict_table=strict_table)
    for index, (op, arg1, arg2) in enumerate(parsed):
        if op.startswith("table_"):
            if not arg1:
                raise ValueError("table operator row label must be non-empty")
            continue
        for argument in (arg1, arg2):
            match = REF_RE.fullmatch(argument)
            if match:
                if int(match.group(1)) >= index:
                    raise ValueError(
                        f"forward or missing Program reference: {argument}"
                    )
            else:
                _number(argument)
    return parsed


def _number(argument: str) -> float:
    if argument in ALLOWED_CONSTANTS:
        value = argument.removeprefix("const_")
        return -1.0 if value == "m1" else float(value)
    if argument.startswith("const_"):
        raise ValueError(f"non-whitelisted FinQA constant: {argument}")
    if not NUMBER_RE.fullmatch(argument):
        raise ValueError(f"invalid numeric argument: {argument}")
    if argument.endswith("%"):
        return float(argument[:-1]) / 100.0
    return float(argument)


def _table_numbers(table: list[list[Any]], row_label: str) -> list[float]:
    rows = {str(row[0]): row[1:] for row in table if row}
    if row_label not in rows:
        raise ValueError(f"table row label not found: {row_label}")
    values: list[float] = []
    for raw in rows[row_label]:
        value = str(raw).replace("$", "").replace(",", "").strip()
        value = value.split("(")[0].strip()
        values.append(_number(value))
    if not values:
        raise ValueError(f"table row has no numeric values: {row_label}")
    return values


def evaluate_program(
    program: str,
    table: list[list[Any]],
    *,
    strict_table: bool = False,
) -> float | str:
    results: list[float | str] = []
    for index, (op, arg1, arg2) in enumerate(
        parse_program(program, strict_table=strict_table)
    ):
        if op.startswith("table_"):
            values = _table_numbers(table, arg1)
            if op == "table_max":
                result: float | str = max(values)
            elif op == "table_min":
                result = min(values)
            elif op == "table_sum":
                result = sum(values)
            else:
                result = sum(values) / len(values)
        else:
            operands: list[float | str] = []
            for argument in (arg1, arg2):
                match = REF_RE.fullmatch(argument)
                if match:
                    reference = int(match.group(1))
                    if reference >= index:
                        raise ValueError(f"forward or missing Program reference: {argument}")
                    operands.append(results[reference])
                else:
                    operands.append(_number(argument))
            left, right = operands
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                raise ValueError(f"non-numeric input to {op}: {left}, {right}")
            if op == "add":
                result = left + right
            elif op == "subtract":
                result = left - right
            elif op == "multiply":
                result = left * right
            elif op == "divide":
                if right == 0:
                    raise ValueError("division by zero")
                result = left / right
            elif op == "exp":
                result = left**right
            else:
                result = "yes" if left > right else "no"
        if isinstance(result, float) and not math.isfinite(result):
            raise ValueError(f"non-finite Program result at operation {index}")
        results.append(result)
    final = results[-1]
    return round(final, 5) if isinstance(final, float) else final


def normalize_program_result(result: float | str) -> str:
    if isinstance(result, str):
        return result.strip().lower()
    normalized = format(float(result), ".5f").rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def evaluate_output(output: str, sample: dict[str, Any]) -> dict[str, Any]:
    fields, format_errors = parse_three_field_output(output)
    prediction = extract_final_answer(output)
    gold = normalize_program_result(sample["qa"]["exe_ans"])
    answer_correct, answer_match_type = compare_answer(prediction, gold)

    brief_errors = validate_brief_format(fields["brief"])
    answer_format_errors = validate_answer_format(fields["answer"])
    format_errors.extend(
        error for error in answer_format_errors if error not in format_errors
    )
    program = fields["program"]
    program_parse_valid = False
    program_executable = False
    program_result: str | None = None
    program_result_correct: bool | None = None
    program_answer_consistent: bool | None = None
    program_error: str | None = None

    if not program:
        program_error = "missing_program"
    else:
        try:
            validate_program_syntax(program, strict_table=True)
            program_parse_valid = True
            executed = evaluate_program(program, sample["table"], strict_table=True)
            program_executable = True
            program_result = normalize_program_result(executed)
            program_result_correct, _ = compare_answer(program_result, gold)
            if is_valid_prediction(prediction):
                program_answer_consistent, _ = compare_answer(
                    program_result,
                    prediction,
                )
        except Exception as exc:
            program_error = f"{type(exc).__name__}: {exc}"

    return {
        "prediction": prediction,
        "valid_prediction": is_valid_prediction(prediction),
        "answer_correct": answer_correct,
        "answer_match_type": answer_match_type,
        "answer_format_valid": not answer_format_errors,
        "answer_format_errors": answer_format_errors,
        "three_field_format_valid": not format_errors,
        "three_field_format_errors": format_errors,
        "brief": fields["brief"],
        "brief_format_valid": not brief_errors,
        "brief_format_errors": brief_errors,
        "program": program,
        "program_parse_valid": program_parse_valid,
        "program_executable": program_executable,
        "program_result": program_result,
        "program_result_correct": program_result_correct,
        "program_answer_consistent": program_answer_consistent,
        "program_error": program_error,
    }


def call_model_once(
    sample: dict[str, Any],
) -> tuple[str, float, str | None, str | None, int | None, int]:
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
            choice = payload["choices"][0]
            content = str(choice["message"]["content"])
            finish_reason = choice.get("finish_reason")
            completion_tokens = payload.get("usage", {}).get("completion_tokens")
            return (
                content,
                time.perf_counter() - start,
                None,
                str(finish_reason) if finish_reason is not None else None,
                int(completion_tokens) if completion_tokens is not None else None,
                attempt,
            )
        except Exception as exc:
            last_error = f"attempt {attempt}: {type(exc).__name__}: {exc}"
            if attempt <= MAX_RETRIES:
                time.sleep(min(2**attempt, 8))

    return "", time.perf_counter() - start, last_error, None, None, MAX_RETRIES + 1


async def call_model_attempt(
    sample: dict[str, Any],
    semaphore: asyncio.Semaphore,
    call_index: int,
) -> dict[str, Any]:
    async with semaphore:
        (
            output,
            latency_sec,
            error,
            finish_reason,
            completion_tokens,
            http_attempts,
        ) = await asyncio.to_thread(call_model_once, sample)

    evaluation = evaluate_output(output, sample)
    if error:
        evaluation["answer_correct"] = False
        evaluation["answer_match_type"] = "request_error"

    return {
        "call_index": call_index,
        **evaluation,
        "latency_sec": round(latency_sec, 4),
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "http_attempts": http_attempts,
        "error": error,
        "output": output,
    }


def _step_bucket(sample: dict[str, Any]) -> str:
    step_count = len(parse_program(str(sample["qa"]["program"]), strict_table=True))
    if step_count == 1:
        return "1_step"
    if step_count == 2:
        return "2_step"
    return "3_plus_steps"


async def call_model(
    sample: dict[str, Any],
    semaphore: asyncio.Semaphore,
    dataset_index: int,
) -> dict[str, Any]:
    start = time.perf_counter()
    attempts = await asyncio.gather(
        *[
            call_model_attempt(sample, semaphore, call_index)
            for call_index in range(1, CALLS_PER_SAMPLE + 1)
        ]
    )

    successful_attempts = [attempt for attempt in attempts if attempt["error"] is None]
    answer_correct_count = sum(bool(attempt["answer_correct"]) for attempt in attempts)
    first_attempt = attempts[0]
    successful_calls = len(successful_attempts)

    return {
        "dataset_index": dataset_index,
        "id": str(sample.get("id", "")),
        "question": str(sample["qa"]["question"]),
        "gold": normalize_program_result(sample["qa"]["exe_ans"]),
        "gold_program": str(sample["qa"]["program"]),
        "gold_step_bucket": _step_bucket(sample),
        "prediction": str(first_attempt["prediction"]),
        "first_call_answer_correct": bool(first_attempt["answer_correct"]),
        "answer_correct_count": answer_correct_count,
        "answer_avg_at_k": round(answer_correct_count / len(attempts), 6),
        "answer_best_at_k": int(answer_correct_count > 0),
        "k": len(attempts),
        "latency_sec": round(time.perf_counter() - start, 4),
        "call_count": len(attempts),
        "successful_calls": successful_calls,
        "call_errors": len(attempts) - successful_calls,
        "selection_method": "first_call_no_gold_selection",
        "error": "all_calls_failed" if successful_calls == 0 else None,
        "output": first_attempt["output"],
        "calls": attempts,
    }


def new_summary() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_url": MODEL_URL,
        "test_file": str(TEST_FILE),
        "test_file_sha256": sha256(TEST_FILE),
        "prompt_source": PROMPT_SOURCE,
        "prompt_sha256": text_sha256(SYSTEM_PROMPT),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
        "total_questions": 0,
        "answer_best_correct_questions": 0,
        "answer_correct_calls": 0,
        "logical_calls": 0,
        "successful_calls": 0,
        "request_errors": 0,
        "http_attempts": 0,
        "all_calls_failed_questions": 0,
        "three_field_format_valid_calls": 0,
        "answer_format_valid_calls": 0,
        "brief_format_valid_calls": 0,
        "program_parse_valid_calls": 0,
        "program_executable_calls": 0,
        "program_result_correct_calls": 0,
        "program_answer_consistent_calls": 0,
        "program_answer_consistency_eligible_calls": 0,
        "truncated_calls": 0,
        "completion_tokens": 0,
        "completion_token_calls": 0,
        "question_latency_sum": 0.0,
        "call_latency_sum": 0.0,
        "answer_match_types": {},
        "step_buckets": {
            "1_step": {"questions": 0, "answer_correct_calls": 0, "answer_best_correct_questions": 0},
            "2_step": {"questions": 0, "answer_correct_calls": 0, "answer_best_correct_questions": 0},
            "3_plus_steps": {"questions": 0, "answer_correct_calls": 0, "answer_best_correct_questions": 0},
        },
    }


def update_summary(summary: dict[str, Any], record: dict[str, Any]) -> None:
    summary["total_questions"] += 1
    summary["answer_best_correct_questions"] += int(record["answer_best_at_k"])
    summary["answer_correct_calls"] += int(record["answer_correct_count"])
    summary["logical_calls"] += int(record["call_count"])
    summary["successful_calls"] += int(record["successful_calls"])
    summary["request_errors"] += int(record["call_errors"])
    summary["all_calls_failed_questions"] += int(bool(record["error"]))
    summary["question_latency_sum"] += float(record["latency_sec"])

    bucket = summary["step_buckets"][record["gold_step_bucket"]]
    bucket["questions"] += 1
    bucket["answer_correct_calls"] += int(record["answer_correct_count"])
    bucket["answer_best_correct_questions"] += int(record["answer_best_at_k"])

    for attempt in record["calls"]:
        summary["http_attempts"] += int(attempt["http_attempts"])
        summary["call_latency_sum"] += float(attempt["latency_sec"])
        summary["three_field_format_valid_calls"] += int(
            bool(attempt["three_field_format_valid"])
        )
        summary["answer_format_valid_calls"] += int(
            bool(attempt["answer_format_valid"])
        )
        summary["brief_format_valid_calls"] += int(bool(attempt["brief_format_valid"]))
        summary["program_parse_valid_calls"] += int(bool(attempt["program_parse_valid"]))
        summary["program_executable_calls"] += int(bool(attempt["program_executable"]))
        summary["program_result_correct_calls"] += int(
            attempt["program_result_correct"] is True
        )
        if attempt["program_answer_consistent"] is not None:
            summary["program_answer_consistency_eligible_calls"] += 1
            summary["program_answer_consistent_calls"] += int(
                bool(attempt["program_answer_consistent"])
            )
        summary["truncated_calls"] += int(attempt["finish_reason"] == "length")
        if attempt["completion_tokens"] is not None:
            summary["completion_tokens"] += int(attempt["completion_tokens"])
            summary["completion_token_calls"] += 1
        match_type = str(attempt["answer_match_type"])
        summary["answer_match_types"][match_type] = (
            summary["answer_match_types"].get(match_type, 0) + 1
        )


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    total_questions = int(summary["total_questions"])
    logical_calls = int(summary["logical_calls"])
    successful_calls = int(summary["successful_calls"])
    consistency_eligible = int(summary["program_answer_consistency_eligible_calls"])

    summary["k"] = CALLS_PER_SAMPLE
    summary["answer_avg_at_k"] = _rate(
        int(summary["answer_correct_calls"]), logical_calls
    )
    summary["answer_best_at_k"] = _rate(
        int(summary["answer_best_correct_questions"]), total_questions
    )
    summary["accuracy"] = summary["answer_avg_at_k"]
    summary["accuracy_definition"] = f"answer_avg_at_{CALLS_PER_SAMPLE}"
    summary["three_field_format_valid_rate"] = _rate(
        int(summary["three_field_format_valid_calls"]), successful_calls
    )
    summary["answer_format_valid_rate"] = _rate(
        int(summary["answer_format_valid_calls"]), successful_calls
    )
    summary["brief_format_valid_rate"] = _rate(
        int(summary["brief_format_valid_calls"]), successful_calls
    )
    summary["program_parse_valid_rate"] = _rate(
        int(summary["program_parse_valid_calls"]), successful_calls
    )
    summary["program_executable_rate"] = _rate(
        int(summary["program_executable_calls"]), successful_calls
    )
    summary["program_result_correct_rate"] = _rate(
        int(summary["program_result_correct_calls"]), successful_calls
    )
    summary["program_answer_consistent_rate"] = _rate(
        int(summary["program_answer_consistent_calls"]), consistency_eligible
    )
    summary["avg_question_latency_sec"] = round(
        float(summary["question_latency_sum"]) / max(total_questions, 1), 4
    )
    summary["avg_call_latency_sec"] = round(
        float(summary["call_latency_sum"]) / max(logical_calls, 1), 4
    )
    summary["avg_completion_tokens"] = round(
        int(summary["completion_tokens"])
        / max(int(summary["completion_token_calls"]), 1),
        2,
    )

    for bucket in summary["step_buckets"].values():
        questions = int(bucket["questions"])
        calls = questions * CALLS_PER_SAMPLE
        bucket["answer_avg_at_k"] = _rate(int(bucket["answer_correct_calls"]), calls)
        bucket["answer_best_at_k"] = _rate(
            int(bucket["answer_best_correct_questions"]), questions
        )

    del summary["question_latency_sum"]
    del summary["call_latency_sum"]
    return summary


def print_record(record: dict[str, Any], completed: int, total: int) -> None:
    status = "✓" if record["answer_best_at_k"] else "✗"
    first_call_status = "✓" if record["first_call_answer_correct"] else "✗"
    print(f"\n[{completed}/{total}] id={record['id']} Answer best@{record['k']} {status}")
    print(f"  gold: {record['gold']!r}")
    print(f"  first_call: {record['prediction']!r} {first_call_status}")
    print(f"  answer_correct_count: {record['answer_correct_count']}/{record['k']}")
    print(
        f"  answer avg@{record['k']}: {record['answer_avg_at_k']:.4f} | "
        f"best@{record['k']}: {record['answer_best_at_k']}"
    )
    if record["error"]:
        print(f"  error: {record['error']}")


def create_run_dir(run_index: int) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{run_index:03d}_{timestamp}"
    candidate = RUNS_DIR / base_name
    duplicate_index = 1
    while candidate.exists():
        candidate = RUNS_DIR / f"{base_name}_{duplicate_index:02d}"
        duplicate_index += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def append_run_result(record: dict[str, Any]) -> None:
    with RESULTS_FILE.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()


async def run_test_once(
    samples: list[dict[str, Any]],
    run_index: int,
    run_dir: Path,
) -> dict[str, Any]:
    predictions_file = run_dir / "finqa_predictions.jsonl"
    summary_file = run_dir / "finqa_summary.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    start = time.perf_counter()
    semaphore = asyncio.Semaphore(CONCURRENCY)
    summary = new_summary()

    print(f"\n===== Test run {run_index}/{TEST_RUNS}: {run_dir.name} =====")

    tasks = [
        asyncio.create_task(call_model(sample, semaphore, dataset_index))
        for dataset_index, sample in enumerate(samples)
    ]
    pending_lines: list[str] = []
    with predictions_file.open("w", encoding="utf-8") as out:
        for completed, task in enumerate(asyncio.as_completed(tasks), start=1):
            record = await task
            pending_lines.append(json.dumps(record, ensure_ascii=False) + "\n")
            if len(pending_lines) >= WRITE_BATCH_SIZE:
                out.writelines(pending_lines)
                out.flush()
                pending_lines.clear()
            update_summary(summary, record)
            print_record(record, completed, len(samples))
        if pending_lines:
            out.writelines(pending_lines)
            out.flush()
            pending_lines.clear()

    summary = finalize_summary(summary)
    completed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    duration_sec = round(time.perf_counter() - start, 4)
    summary.update(
        {
            "run_index": run_index,
            "run_id": run_dir.name,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_sec": duration_sec,
        }
    )
    summary_file.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n===== Final Answer Results: run {run_index} (K={CALLS_PER_SAMPLE}) =====")
    print(f"model: {MODEL_ID}")
    print(f"Answer avg@{CALLS_PER_SAMPLE}: {summary['answer_avg_at_k']:.4f}")
    print(f"Answer best@{CALLS_PER_SAMPLE}: {summary['answer_best_at_k']:.4f}")
    print(f"total_questions: {summary['total_questions']}")
    print(f"request_errors: {summary['request_errors']}/{summary['logical_calls']}")
    print(f"three_field_format_valid_rate: {summary['three_field_format_valid_rate']:.4f}")
    print(f"program_parse_valid_rate: {summary['program_parse_valid_rate']:.4f}")
    print(f"program_executable_rate: {summary['program_executable_rate']:.4f}")
    print(f"program_result_correct_rate: {summary['program_result_correct_rate']:.4f}")
    print(
        "program_answer_consistent_rate: "
        f"{summary['program_answer_consistent_rate']:.4f}"
    )
    print(f"truncated_calls: {summary['truncated_calls']}")
    print(f"avg_question_latency_sec: {summary['avg_question_latency_sec']}")
    print(f"avg_call_latency_sec: {summary['avg_call_latency_sec']}")
    print(f"predictions: {predictions_file}")
    print(f"summary: {summary_file}")

    return {
        "run_index": run_index,
        "run_id": run_dir.name,
        "started_at": started_at,
        "completed_at": completed_at,
        "duration_sec": duration_sec,
        "run_dir": str(run_dir),
        "predictions_file": str(predictions_file),
        "summary_file": str(summary_file),
        "summary": summary,
    }


async def main() -> None:
    if TEST_RUNS < 1:
        raise ValueError("TEST_RUNS 必须大于等于 1")
    if CALLS_PER_SAMPLE < 1:
        raise ValueError("CALLS_PER_SAMPLE 必须大于等于 1")
    if CONCURRENCY < 1:
        raise ValueError("CONCURRENCY 必须大于等于 1")
    if WRITE_BATCH_SIZE < 1:
        raise ValueError("WRITE_BATCH_SIZE 必须大于等于 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    samples = load_test_data(TEST_FILE)

    print(f"Loaded {len(samples)} FinQA test samples from {TEST_FILE}")
    print(f"Model: {MODEL_ID}")
    print(f"Prompt: {PROMPT_SOURCE}")
    print(f"Test runs: {TEST_RUNS}")
    print(f"Concurrency per run: {CONCURRENCY}")
    print(f"Calls per sample: {CALLS_PER_SAMPLE}")
    print(f"Run outputs: {RUNS_DIR}")
    print(f"Aggregate results: {RESULTS_FILE}")
    print("Primary metric: Answer avg@K; Brief/Program checks are diagnostics only.")

    for run_index in range(1, TEST_RUNS + 1):
        run_dir = create_run_dir(run_index)
        result = await run_test_once(samples, run_index, run_dir)
        append_run_result(result)
        print(f"Recorded run {run_index} in {RESULTS_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
