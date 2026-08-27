"""Shared FinQA three-field parsing, execution, and rule-reward logic."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Callable


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
EVAL_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?\s*%?")
PROGRAM_DSL_RE = re.compile(
    r"\b(?:add|subtract|multiply|divide|exp|greater|"
    r"table_max|table_min|table_sum|table_average)\s*\("
)
FIELD_LABEL_RE = re.compile(r"\b(?:Brief|Program|Answer):", re.IGNORECASE)
NORMALIZED_ANSWER_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d{1,5})?$")

ABS_TOL = 1e-3
REL_TOL = 1e-3

REWARD_OUTPUT_KEYS = {
    "score",
    "answer_format_valid",
    "answer_correct",
    "program_parse_valid",
    "program_executable",
    "program_result_correct",
    "program_answer_consistent",
    "brief_compliant",
    "strict_three_field_format",
}


def _split_operations(program: str) -> list[str]:
    if not program.strip():
        raise ValueError("empty Program")
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


def parse_program(program: str, *, strict_table: bool = True) -> list[tuple[str, str, str]]:
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
        parsed.append((op, arg1, arg2))
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


def validate_program(program: str) -> list[tuple[str, str, str]]:
    parsed = parse_program(program, strict_table=True)
    for index, (op, arg1, arg2) in enumerate(parsed):
        if op.startswith("table_"):
            if not arg1:
                raise ValueError("table operator row label must be non-empty")
            continue
        for argument in (arg1, arg2):
            match = REF_RE.fullmatch(argument)
            if match:
                if int(match.group(1)) >= index:
                    raise ValueError(f"forward or missing Program reference: {argument}")
            else:
                _number(argument)
    return parsed


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


def evaluate_program(program: str, table: list[list[Any]]) -> float | str:
    parsed = validate_program(program)
    results: list[float | str] = []
    for index, (op, arg1, arg2) in enumerate(parsed):
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
                operands.append(results[int(match.group(1))] if match else _number(argument))
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
        if isinstance(result, (int, float)) and not math.isfinite(float(result)):
            raise ValueError(f"non-finite Program result at operation {index}")
        results.append(result)
    return results[-1]


def normalize_program_result(result: Any) -> str:
    if isinstance(result, str):
        lowered = result.strip().lower()
        if lowered not in {"yes", "no"}:
            raise ValueError(f"invalid non-numeric Program result: {result}")
        return lowered
    value = float(result)
    if not math.isfinite(value):
        raise ValueError(f"non-finite Program result: {result}")
    normalized = format(round(value, 5), ".5f").rstrip("0").rstrip(".")
    return "0" if normalized in {"", "-0"} else normalized


def _clean_answer(text: Any) -> str:
    value = ("" if text is None else str(text)).strip().replace("％", "%")
    value = value.strip("`'\" ").rstrip(".。;；")
    return re.sub(r"\s+", " ", value)


def _normalized_answer(text: Any) -> str:
    return _clean_answer(text).lower().replace(",", "").replace("$", "")


def _numeric_candidates(text: Any) -> set[float]:
    candidates: set[float] = set()
    for raw in EVAL_NUMBER_RE.findall(_normalized_answer(text)):
        token = raw.strip()
        is_percent = token.endswith("%")
        try:
            value = float(token.rstrip("%"))
        except ValueError:
            continue
        if math.isfinite(value):
            candidates.add(value)
            if is_percent:
                candidates.add(value / 100.0)
    return candidates


def answers_equal(prediction: Any, expected: Any) -> bool:
    prediction_normalized = _normalized_answer(prediction)
    expected_normalized = _normalized_answer(expected)
    if not prediction_normalized:
        return False
    if prediction_normalized == expected_normalized:
        return True
    if prediction_normalized in {"yes", "no"} or expected_normalized in {"yes", "no"}:
        return False
    return any(
        math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)
        for left in _numeric_candidates(prediction)
        for right in _numeric_candidates(expected)
    )


def answer_format_is_valid(answer: str) -> bool:
    return answer in {"yes", "no"} or bool(NORMALIZED_ANSWER_RE.fullmatch(answer))


def parse_generated_output(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    all_lines = raw.splitlines() if raw else []
    nonempty_lines = [line.strip() for line in all_lines if line.strip()]

    strict = len(all_lines) == 3 and all(line.strip() for line in all_lines)
    strict_prefixes = ("Brief: ", "Program: ", "Answer: ")
    strict = strict and all(
        all_lines[index].startswith(prefix)
        and bool(all_lines[index][len(prefix) :].strip())
        for index, prefix in enumerate(strict_prefixes)
    )

    def labeled(label: str) -> str:
        prefix = f"{label}:"
        matches = [line[len(prefix) :].strip() for line in nonempty_lines if line.startswith(prefix)]
        return matches[-1] if matches else ""

    brief = labeled("Brief")
    program = labeled("Program")
    answer = labeled("Answer")
    if (
        not brief
        and len(nonempty_lines) == 3
        and not FIELD_LABEL_RE.search(nonempty_lines[0])
        and nonempty_lines[1].startswith("Program:")
        and nonempty_lines[2].startswith("Answer:")
    ):
        brief = nonempty_lines[0]

    return {
        "brief": brief,
        "program": program,
        "answer": answer,
        "strict_three_field_format": bool(strict),
    }


def brief_is_compliant(brief: str, token_counter: Callable[[str], int]) -> bool:
    if not brief or "\n" in brief or "\r" in brief:
        return False
    lowered = brief.lower()
    if "<think>" in lowered or "</think>" in lowered:
        return False
    if FIELD_LABEL_RE.search(brief) or PROGRAM_DSL_RE.search(brief):
        return False
    try:
        return token_counter(brief) <= 64
    except Exception:
        return False


def parse_ground_truth(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    value = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not isinstance(value, dict):
        raise ValueError("ground_truth must decode to an object")
    required = ("id", "answer", "program", "brief", "table")
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"ground_truth missing fields: {', '.join(missing)}")
    if not isinstance(value["table"], list):
        raise ValueError("ground_truth table must be a list")
    return value


def evaluate_reward(
    solution: Any,
    ground_truth: str | dict[str, Any],
    *,
    brief_token_counter: Callable[[str], int],
) -> dict[str, float]:
    gold = parse_ground_truth(ground_truth)
    parsed = parse_generated_output(solution)

    answer_format_valid = float(answer_format_is_valid(parsed["answer"]))
    answer_correct = float(
        bool(answer_format_valid) and answers_equal(parsed["answer"], gold["answer"])
    )
    program_parse_valid = 0.0
    program_executable = 0.0
    program_result_correct = 0.0
    program_answer_consistent = 0.0
    if parsed["program"]:
        try:
            validate_program(parsed["program"])
            program_parse_valid = 1.0
            result = normalize_program_result(
                evaluate_program(parsed["program"], gold["table"])
            )
            program_executable = 1.0
            program_result_correct = float(answers_equal(result, gold["answer"]))
            program_answer_consistent = float(
                bool(answer_format_valid) and answers_equal(result, parsed["answer"])
            )
        except (ArithmeticError, OverflowError, TypeError, ValueError):
            pass

    brief_compliant = float(
        brief_is_compliant(parsed["brief"], brief_token_counter)
    )
    strict_format = float(parsed["strict_three_field_format"])
    score = (
        0.50 * answer_correct
        + 0.45 * (program_result_correct * program_answer_consistent)
        + 0.025 * brief_compliant
        + 0.025 * strict_format
    )
    return {
        "score": float(round(score, 10)),
        "answer_format_valid": answer_format_valid,
        "answer_correct": answer_correct,
        "program_parse_valid": program_parse_valid,
        "program_executable": program_executable,
        "program_result_correct": program_result_correct,
        "program_answer_consistent": program_answer_consistent,
        "brief_compliant": brief_compliant,
        "strict_three_field_format": strict_format,
    }
