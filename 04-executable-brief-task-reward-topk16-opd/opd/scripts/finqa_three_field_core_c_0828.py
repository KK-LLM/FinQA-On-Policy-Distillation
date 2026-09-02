"""FinQA experiment-C three-field parsing, execution, and task reward."""

from __future__ import annotations

import json
import re
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
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
PROGRAM_DSL_RE = re.compile(
    r"\b(?:add|subtract|multiply|divide|exp|greater|"
    r"table_max|table_min|table_sum|table_average)\s*\("
)
FIELD_LABEL_RE = re.compile(r"\b(?:Brief|Program|Answer):", re.IGNORECASE)
ANSWER_NUMBER_RE = re.compile(r"^-?(?:0|[1-9]\d*)(?:\.\d{1,5})?$")
BRIEF_NUMBER_RE = re.compile(
    r"(?P<approx>≈\s*)?"
    r"(?<![\w#])"
    r"(?P<number>[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+))"
    r"(?P<percent>\s*%)?"
)
PERCENTAGE_RESULT_PATTERNS = (
    re.compile(r"\bwhat\s+percent(?:age|ual)?\b", flags=re.IGNORECASE),
    re.compile(
        r"\bwhat\s+(?:was|is|were|are|would\s+be)\s+"
        r"(?:the\s+)?(?:[\w&'-]+\s+){0,5}"
        r"(?:percent(?:age|ual)?|margin)\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bby\s+what\s+percent(?:age|ual)?\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:represented|represents|are|is|was|were)\s+"
        r"what\s+percent(?:age|ual)?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bas\s+(?:a\s+)?percent(?:age|ual)?\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:gross|net\s+income|operating|profit)\s+margin\b",
        flags=re.IGNORECASE,
    ),
    re.compile(r"\bby\s+how\s+many\s+basis\s+points?\b", flags=re.IGNORECASE),
    re.compile(r"\bchange\s+in\s+basis\s+points?\b", flags=re.IGNORECASE),
    re.compile(r"\bhow\s+much\s+percent(?:age|ual)?\b", flags=re.IGNORECASE),
    re.compile(r"\bin\s+percent(?:age|ual)?\b", flags=re.IGNORECASE),
    re.compile(
        r"\b(?:difference|growth)\b[^?]*\bpercent(?:age|ual)?\b",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"\bwhat(?:'s|s)\s+is\s+(?:the\s+)?percent(?:age|ual)?\b",
        flags=re.IGNORECASE,
    ),
)

REWARD_OUTPUT_KEYS = {
    "score",
    "answer_format_valid",
    "answer_correct",
    "program_parse_valid",
    "program_executable",
    "program_step_count_valid",
    "program_graph_valid",
    "program_operands_supported",
    "program_result_correct",
    "program_answer_consistent",
    "program_chain_correct",
    "brief_compliant",
    "brief_trace_consistent",
    "final_value_exact",
    "strict_three_field_format",
    "chain_valid",
}

ExactNumber = Fraction | Decimal
ProgramOperation = tuple[str, str, str]


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


def parse_program(program: str, *, strict_table: bool = True) -> list[ProgramOperation]:
    parsed: list[ProgramOperation] = []
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


def _fraction_from_literal(argument: str) -> Fraction:
    if argument in ALLOWED_CONSTANTS:
        value = argument.removeprefix("const_")
        return Fraction(-1 if value == "m1" else int(value), 1)
    if argument.startswith("const_"):
        raise ValueError(f"non-whitelisted FinQA constant: {argument}")
    if not NUMBER_RE.fullmatch(argument):
        raise ValueError(f"invalid numeric argument: {argument}")
    if argument.endswith("%"):
        return Fraction(argument[:-1]) / 100
    return Fraction(argument)


def validate_program(program: str) -> list[ProgramOperation]:
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
                _fraction_from_literal(argument)
    return parsed


def program_graph_is_valid(parsed: list[ProgramOperation]) -> bool:
    if not parsed:
        return False
    required = {len(parsed) - 1}
    pending = [len(parsed) - 1]
    while pending:
        index = pending.pop()
        _, arg1, arg2 = parsed[index]
        for argument in (arg1, arg2):
            match = REF_RE.fullmatch(argument)
            if match:
                dependency = int(match.group(1))
                if dependency not in required:
                    required.add(dependency)
                    pending.append(dependency)
    return required == set(range(len(parsed)))


def _table_numbers(table: list[list[Any]], row_label: str) -> list[Fraction]:
    matching_rows = [row[1:] for row in table if row and str(row[0]) == row_label]
    if not matching_rows:
        raise ValueError(f"table row label not found: {row_label}")
    if len(matching_rows) != 1:
        raise ValueError(f"table row label is not unique: {row_label}")
    values: list[Fraction] = []
    for raw in matching_rows[0]:
        value = str(raw).replace("−", "-").replace("$", "").replace(",", "").strip()
        value = value.split("(", 1)[0].strip()
        value = re.sub(r"\s+", "", value)
        if value in {"", "-", "–", "—", "n/a", "N/A"}:
            continue
        values.append(_fraction_from_literal(value))
    if not values:
        raise ValueError(f"table row has no numeric values: {row_label}")
    return values


def _as_decimal(value: ExactNumber, precision: int) -> Decimal:
    with localcontext() as context:
        context.prec = precision
        if isinstance(value, Decimal):
            return +value
        return Decimal(value.numerator) / Decimal(value.denominator)


def _decimal_binary(
    left: ExactNumber,
    right: ExactNumber,
    operation: Callable[[Decimal, Decimal], Decimal],
) -> Decimal:
    with localcontext() as context:
        context.prec = 120
        return +operation(_as_decimal(left, 120), _as_decimal(right, 120))


def _decimal_power(left: ExactNumber, right: ExactNumber) -> Decimal:
    def calculate(precision: int) -> Decimal:
        with localcontext() as context:
            context.prec = precision
            base = _as_decimal(left, precision)
            exponent = _as_decimal(right, precision)
            if base <= 0:
                raise ValueError("non-integer exponent requires a positive base")
            try:
                return +context.power(base, exponent)
            except InvalidOperation as error:
                raise ValueError("invalid non-integer exponent") from error

    lower_precision = calculate(80)
    higher_precision = calculate(120)
    if _normalize_numeric(lower_precision, 5) != _normalize_numeric(
        higher_precision, 5
    ):
        raise ValueError("non-integer exponent is unstable at five decimals")
    return higher_precision


def _compare_numbers(left: ExactNumber, right: ExactNumber) -> int:
    if isinstance(left, Fraction) and isinstance(right, Fraction):
        return (left > right) - (left < right)
    left_decimal = _as_decimal(left, 120)
    right_decimal = _as_decimal(right, 120)
    return (left_decimal > right_decimal) - (left_decimal < right_decimal)


def evaluate_program_trace(
    program: str,
    table: list[list[Any]],
) -> list[ExactNumber | str]:
    parsed = validate_program(program)
    results: list[ExactNumber | str] = []
    for index, (op, arg1, arg2) in enumerate(parsed):
        if op.startswith("table_"):
            values = _table_numbers(table, arg1)
            if op == "table_max":
                result: ExactNumber | str = max(values)
            elif op == "table_min":
                result = min(values)
            elif op == "table_sum":
                result = sum(values, Fraction(0, 1))
            else:
                result = sum(values, Fraction(0, 1)) / len(values)
        else:
            operands: list[ExactNumber | str] = []
            for argument in (arg1, arg2):
                match = REF_RE.fullmatch(argument)
                operands.append(
                    results[int(match.group(1))]
                    if match
                    else _fraction_from_literal(argument)
                )
            left, right = operands
            if isinstance(left, str) or isinstance(right, str):
                raise ValueError(f"non-numeric input to {op}: {left}, {right}")
            if op == "add":
                result = (
                    left + right
                    if isinstance(left, Fraction) and isinstance(right, Fraction)
                    else _decimal_binary(left, right, lambda a, b: a + b)
                )
            elif op == "subtract":
                result = (
                    left - right
                    if isinstance(left, Fraction) and isinstance(right, Fraction)
                    else _decimal_binary(left, right, lambda a, b: a - b)
                )
            elif op == "multiply":
                result = (
                    left * right
                    if isinstance(left, Fraction) and isinstance(right, Fraction)
                    else _decimal_binary(left, right, lambda a, b: a * b)
                )
            elif op == "divide":
                if _compare_numbers(right, Fraction(0, 1)) == 0:
                    raise ValueError("division by zero")
                result = (
                    left / right
                    if isinstance(left, Fraction) and isinstance(right, Fraction)
                    else _decimal_binary(left, right, lambda a, b: a / b)
                )
            elif op == "exp":
                if isinstance(right, Fraction) and right.denominator == 1:
                    result = left ** right.numerator
                else:
                    result = _decimal_power(left, right)
            else:
                result = "yes" if _compare_numbers(left, right) > 0 else "no"
        if isinstance(result, Decimal) and not result.is_finite():
            raise ValueError(f"non-finite Program result at operation {index}")
        results.append(result)
    return results


def evaluate_program(program: str, table: list[list[Any]]) -> ExactNumber | str:
    return evaluate_program_trace(program, table)[-1]


def _format_scaled_integer(value: int, digits: int) -> str:
    sign = "-" if value < 0 else ""
    absolute = abs(value)
    if digits == 0:
        return f"{sign}{absolute}"
    scale = 10**digits
    integer, fraction = divmod(absolute, scale)
    rendered = f"{sign}{integer}.{fraction:0{digits}d}".rstrip("0").rstrip(".")
    return "0" if rendered in {"", "-0"} else rendered


def _normalize_fraction(value: Fraction, digits: int) -> str:
    scaled_numerator = abs(value.numerator) * (10**digits)
    quotient, remainder = divmod(scaled_numerator, value.denominator)
    if 2 * remainder >= value.denominator:
        quotient += 1
    if value.numerator < 0:
        quotient = -quotient
    return _format_scaled_integer(quotient, digits)


def _normalize_decimal(value: Decimal, digits: int) -> str:
    if not value.is_finite():
        raise ValueError(f"non-finite Program result: {value}")
    sign = -1 if value < 0 else 1
    absolute = value.copy_abs()
    scale = Decimal(10) ** digits
    with localcontext() as context:
        context.prec = max(120, len(absolute.as_tuple().digits) + abs(absolute.adjusted()) + 20)
        scaled = absolute * scale
        quotient = int(scaled)
        remainder = scaled - Decimal(quotient)
        if remainder >= Decimal("0.5"):
            quotient += 1
    return _format_scaled_integer(sign * quotient, digits)


def _normalize_numeric(value: ExactNumber, digits: int) -> str:
    if not 0 <= digits <= 5:
        raise ValueError("normalization digits must be between 0 and 5")
    if isinstance(value, Fraction):
        return _normalize_fraction(value, digits)
    return _normalize_decimal(value, digits)


def normalize_program_result(result: Any) -> str:
    if isinstance(result, str):
        lowered = result.strip().lower()
        if lowered not in {"yes", "no"}:
            raise ValueError(f"invalid non-numeric Program result: {result}")
        return lowered
    if isinstance(result, Fraction):
        return _normalize_fraction(result, 5)
    if isinstance(result, Decimal):
        return _normalize_decimal(result, 5)
    if isinstance(result, int) and not isinstance(result, bool):
        return str(result)
    raise TypeError(f"unsupported Program result type: {type(result).__name__}")


def _clean_answer(text: Any) -> str:
    value = "" if text is None else str(text)
    value = value.strip().replace("％", "%")
    return value.lower()


def answers_equal(
    prediction: Any,
    expected: Any,
    *,
    allow_percentage_scale: bool = False,
) -> bool:
    del allow_percentage_scale
    return bool(_clean_answer(prediction)) and _clean_answer(prediction) == _clean_answer(
        expected
    )


def answer_format_is_valid(answer: str) -> bool:
    if answer in {"yes", "no"}:
        return True
    if not ANSWER_NUMBER_RE.fullmatch(answer):
        return False
    if "." in answer and answer.endswith("0"):
        return False
    if answer.startswith("-") and Decimal(answer) == 0:
        return False
    return True


def question_requests_percentage_result(question: str) -> bool:
    return any(pattern.search(question) for pattern in PERCENTAGE_RESULT_PATTERNS)


def parse_generated_output(text: Any) -> dict[str, Any]:
    raw = str(text or "")
    strict_lines = raw.split("\n") if raw else []
    strict_prefixes = ("Brief: ", "Program: ", "Answer: ")
    strict = (
        "\r" not in raw
        and not raw.startswith("\n")
        and not raw.endswith("\n")
        and len(strict_lines) == 3
        and all(
            line.startswith(prefix)
            and line == line.rstrip()
            and bool(line[len(prefix) :])
            and line[len(prefix) :] == line[len(prefix) :].strip()
            for line, prefix in zip(strict_lines, strict_prefixes, strict=True)
        )
    )

    nonempty_lines = [line.strip() for line in raw.splitlines() if line.strip()]

    def labeled(label: str) -> tuple[str, bool]:
        prefix = f"{label}:"
        matches = [
            line[len(prefix) :].strip()
            for line in nonempty_lines
            if line.startswith(prefix)
        ]
        return (matches[-1] if matches else "", len(matches) == 1)

    brief, brief_labeled = labeled("Brief")
    program, program_labeled = labeled("Program")
    answer, answer_labeled = labeled("Answer")
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
        "brief_labeled": brief_labeled,
        "program_labeled": program_labeled,
        "answer_labeled": answer_labeled,
        "strict_three_field_format": bool(strict),
    }


def brief_token_limit(operation_count: int | None) -> int | None:
    if operation_count is None:
        return None
    if 1 <= operation_count <= 2:
        return 64
    if 3 <= operation_count <= 4:
        return 128
    if 5 <= operation_count <= 6:
        return 192
    return None


def brief_is_compliant(
    brief: str,
    token_counter: Callable[[str], int],
    *,
    operation_count: int | None,
) -> bool:
    token_limit = brief_token_limit(operation_count)
    if token_limit is None or not brief or "\n" in brief or "\r" in brief:
        return False
    lowered = brief.lower()
    if "<think>" in lowered or "</think>" in lowered:
        return False
    if FIELD_LABEL_RE.search(brief) or PROGRAM_DSL_RE.search(brief):
        return False
    return token_counter(brief) <= token_limit


def _numeric_key(argument: str) -> tuple[int, int]:
    value = _fraction_from_literal(argument)
    return value.numerator, value.denominator


def _approved_keys(values: list[str], *, percentage: bool) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for value in values:
        if not isinstance(value, str):
            raise ValueError("approved operand values must be strings")
        if percentage != value.endswith("%"):
            kind = "percentage" if percentage else "numeric"
            raise ValueError(f"invalid approved {kind} operand: {value}")
        keys.add(_numeric_key(value))
    return keys


def program_operands_are_supported(
    parsed: list[ProgramOperation],
    ground_truth: dict[str, Any],
) -> bool:
    allowed_numeric = _approved_keys(
        ground_truth["allowed_source_operands"], percentage=False
    )
    allowed_percentages = _approved_keys(
        ground_truth["allowed_percentage_literals"], percentage=True
    )
    allowed_rows = set(ground_truth["allowed_table_row_labels"])
    allowed_constants = set(ground_truth["allowed_constants"])
    for op, arg1, arg2 in parsed:
        if op.startswith("table_"):
            if arg1 not in allowed_rows:
                return False
            continue
        for argument in (arg1, arg2):
            if REF_RE.fullmatch(argument):
                continue
            if argument.startswith("const_"):
                if argument not in allowed_constants:
                    return False
            elif argument.endswith("%"):
                if _numeric_key(argument) not in allowed_percentages:
                    return False
            elif _numeric_key(argument) not in allowed_numeric:
                return False
    return True


def _finite_decimal_places(value: Fraction) -> int | None:
    denominator = value.denominator
    twos = 0
    fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    return max(twos, fives) if denominator == 1 else None


def _result_requires_approximation(value: ExactNumber) -> bool:
    if isinstance(value, Decimal):
        return True
    places = _finite_decimal_places(value)
    return places is None or places > 5


def _brief_mentions(brief: str) -> list[dict[str, Any]]:
    normalized = brief.replace("−", "-")
    mentions: list[dict[str, Any]] = []
    for match in BRIEF_NUMBER_RE.finditer(normalized):
        number = match.group("number").replace(",", "")
        try:
            value = Fraction(number)
        except ValueError:
            continue
        decimals = len(number.split(".", 1)[1]) if "." in number else 0
        mentions.append(
            {
                "start": match.start(),
                "end": match.end(),
                "value": value,
                "decimals": decimals,
                "approx": bool(match.group("approx")),
                "percent": bool(match.group("percent")),
            }
        )
    return mentions


def _mention_matches_result(
    mention: dict[str, Any],
    result: ExactNumber,
    *,
    final_answer: str | None,
) -> bool:
    if final_answer is not None:
        if mention["approx"]:
            return False
        return _normalize_fraction(mention["value"], mention["decimals"]) == final_answer
    requires_approximation = _result_requires_approximation(result)
    if requires_approximation:
        if not mention["approx"] or not 1 <= mention["decimals"] <= 5:
            return False
        return _normalize_numeric(result, mention["decimals"]) == _normalize_fraction(
            mention["value"], mention["decimals"]
        )
    if mention["approx"]:
        return False
    if isinstance(result, Decimal):
        return False
    return mention["value"] == result


def _find_result_mention(
    mentions: list[dict[str, Any]],
    result: ExactNumber,
    *,
    after: int,
    final_answer: str | None,
) -> dict[str, Any] | None:
    for mention in mentions:
        if mention["start"] < after:
            continue
        if _mention_matches_result(mention, result, final_answer=final_answer):
            return mention
    return None


def _source_appears_before(
    mentions: list[dict[str, Any]],
    argument: str,
    *,
    before: int,
) -> bool:
    expected = _fraction_from_literal(argument)
    requires_percent = argument.endswith("%")
    for mention in mentions:
        if mention["end"] > before or mention["approx"]:
            continue
        if requires_percent and not mention["percent"]:
            continue
        if mention["value"] == (expected * 100 if requires_percent else expected):
            return True
    return False


CONSTANT_WORDS = {
    "const_m1": ("negative one", "minus one"),
    "const_1": ("one",),
    "const_2": ("two",),
    "const_3": ("three",),
    "const_4": ("four",),
    "const_5": ("five",),
    "const_6": ("six",),
    "const_7": ("seven",),
    "const_8": ("eight",),
    "const_9": ("nine",),
    "const_10": ("ten",),
    "const_100": ("one hundred",),
    "const_1000": ("one thousand",),
    "const_10000": ("ten thousand",),
    "const_100000": ("one hundred thousand",),
    "const_1000000": ("one million",),
    "const_1000000000": ("one billion",),
}


def _constant_word_spans(brief: str, argument: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for phrase in CONSTANT_WORDS.get(argument, ()):
        spans.extend(
            (match.start(), match.end())
            for match in re.finditer(
                rf"\b{re.escape(phrase)}\b", brief, flags=re.IGNORECASE
            )
        )
    return spans


def _argument_spans(
    brief: str,
    mentions: list[dict[str, Any]],
    argument: str,
    *,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    if REF_RE.fullmatch(argument):
        return []
    expected = _fraction_from_literal(argument)
    requires_percent = argument.endswith("%")
    spans: list[tuple[int, int]] = []
    for mention in mentions:
        if mention["start"] < start or mention["end"] > end or mention["approx"]:
            continue
        if requires_percent and not mention["percent"]:
            continue
        represented = mention["value"] / 100 if requires_percent else mention["value"]
        if represented == expected:
            spans.append((mention["start"], mention["end"]))
    if argument.startswith("const_"):
        spans.extend(
            span
            for span in _constant_word_spans(brief, argument)
            if start <= span[0] and span[1] <= end
        )
    return sorted(set(spans))


def _normalize_phrase(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


TABLE_TERMS = {
    "table_max": ("max", "maximum", "highest", "largest", "greatest"),
    "table_min": ("min", "minimum", "lowest", "smallest"),
    "table_sum": ("sum", "total", "combined"),
    "table_average": ("average", "mean"),
}


def _operation_inputs_appear_before(
    brief: str,
    mentions: list[dict[str, Any]],
    operation: ProgramOperation,
    *,
    operation_index: int,
    after: int,
    before: int,
) -> bool:
    op, arg1, arg2 = operation
    if op.startswith("table_"):
        return True
    for argument in (arg1, arg2):
        if REF_RE.fullmatch(argument):
            continue
        if _source_appears_before(mentions, argument, before=before):
            continue
        if argument.startswith("const_") and any(
            end <= before for _, end in _constant_word_spans(brief, argument)
        ):
            continue
        if (
            op == "subtract"
            and argument == arg2
            and _uses_immediate_previous_result_as_left_operand(
                brief[after:before],
                op,
                arg1,
                operation_index,
            )
            and _signed_subtraction_rhs_spans(
                mentions,
                argument,
                start=after,
                end=before,
            )
        ):
            continue
        return False
    return True


OPERATION_MARKERS = {
    "add": ("+", " plus ", " add", "sum", "combined", "total"),
    "subtract": (" minus ", "subtract", "difference", "decrease", " less "),
    "multiply": ("\u00d7", "*", " times ", " multiply", "product"),
    "divide": ("\u00f7", "/", " divided by ", " ratio", " per "),
    "exp": ("^", " raised to ", " power"),
    "greater": (
        ">",
        " greater than ",
        " higher than ",
        " more than ",
        " more",
        " versus ",
        "exceed",
        "above",
    ),
}


def _contains_operation_marker(text: str, operator: str) -> bool:
    lowered = text.lower().replace("−", "-")
    return any(marker in lowered for marker in OPERATION_MARKERS[operator])


def _resolved_argument_spans(
    brief: str,
    mentions: list[dict[str, Any]],
    argument: str,
    trace: list[ExactNumber | str],
    *,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    reference = REF_RE.fullmatch(argument)
    if reference is None:
        return _argument_spans(
            brief,
            mentions,
            argument,
            start=start,
            end=end,
        )
    value = trace[int(reference.group(1))]
    ordinal = ("first", "second", "third", "fourth", "fifth", "sixth")[
        int(reference.group(1))
    ]
    ordinal_spans = [
        (match.start(), match.end())
        for match in re.finditer(rf"\b{ordinal}\b", brief, re.IGNORECASE)
        if start <= match.start() and match.end() <= end
    ]
    if isinstance(value, str):
        return ordinal_spans + [
            (match.start(), match.end())
            for match in re.finditer(rf"\b{re.escape(value)}\b", brief, re.IGNORECASE)
            if start <= match.start() and match.end() <= end
        ]
    return ordinal_spans + [
        (mention["start"], mention["end"])
        for mention in mentions
        if start <= mention["start"]
        and mention["end"] <= end
        and _normalize_numeric(value, mention["decimals"])
        == _normalize_fraction(mention["value"], mention["decimals"])
    ]


def _binary_minus_separates_operands(
    brief: str,
    first_spans: list[tuple[int, int]],
    second_spans: list[tuple[int, int]],
) -> bool:
    return any(
        "-" in brief[first_end:second_start]
        for _, first_end in first_spans
        for second_start, _ in second_spans
        if first_end <= second_start
    )


def _signed_subtraction_rhs_spans(
    mentions: list[dict[str, Any]],
    argument: str,
    *,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    if REF_RE.fullmatch(argument):
        return []
    expected = _fraction_from_literal(argument)
    if expected <= 0:
        return []
    requires_percent = argument.endswith("%")
    return [
        (mention["start"], mention["end"])
        for mention in mentions
        if start <= mention["start"]
        and mention["end"] <= end
        and not mention["approx"]
        and (not requires_percent or mention["percent"])
        and (
            mention["value"] / 100 if requires_percent else mention["value"]
        )
        == -expected
    ]


IMPLICIT_LEFT_OPERATOR_PREFIXES = {
    "subtract": r"(?:-|minus\b|subtract\b)",
    "divide": r"(?:÷|/|divided\s+by\b)",
    "exp": r"(?:\^|raised\s+to\b)",
    "greater": r"(?:>|is\s+greater\s+than\b|greater\s+than\b|exceeds?\b)",
}

INFIX_OPERATOR_PATTERNS = {
    "add": r"(?:\+|plus\b)",
    "subtract": r"(?:-|minus\b)",
    "multiply": r"(?:×|\*|times\b)",
    "divide": r"(?:÷|/|divided\s+by\b)",
    "exp": r"(?:\^|raised\s+to\b)",
    "greater": r"(?:>|is\s+greater\s+than\b|greater\s+than\b)",
}


def _repeats_previous_operation_as_left_operand(
    continuation: str,
    previous_operation: ProgramOperation | None,
    current_operator: str,
) -> bool:
    if previous_operation is None:
        return False
    previous_operator, previous_arg1, previous_arg2 = previous_operation
    if not NUMBER_RE.fullmatch(previous_arg1) or not NUMBER_RE.fullmatch(previous_arg2):
        return False
    previous_marker = INFIX_OPERATOR_PATTERNS.get(previous_operator)
    current_marker = INFIX_OPERATOR_PATTERNS.get(current_operator)
    if previous_marker is None or current_marker is None:
        return False
    return (
        re.match(
            rf"^\(\s*{re.escape(previous_arg1)}\s*{previous_marker}\s*"
            rf"{re.escape(previous_arg2)}\s*\)\s*{current_marker}",
            continuation,
        )
        is not None
    )


def _uses_immediate_previous_result_as_left_operand(
    segment: str,
    operator: str,
    argument: str,
    operation_index: int,
    previous_operation: ProgramOperation | None = None,
) -> bool:
    reference = REF_RE.fullmatch(argument)
    if reference is None or int(reference.group(1)) != operation_index - 1:
        return False
    continuation = segment.lower().replace("−", "-").strip()
    if ";" in continuation:
        continuation = continuation.rsplit(";", 1)[1].strip()
    difference_anaphora = re.match(r"^(?:their\s+)?difference\b", continuation)
    if difference_anaphora is not None and operator == "subtract":
        return False
    anaphora = difference_anaphora or re.match(
        r"^(?:(?:that|the)\s+(?:quotient|result|ratio)|"
        r"(?:the\s+)?unrounded(?:\s+(?:quotient|result|ratio))?)\b",
        continuation,
    )
    if anaphora is not None:
        continuation = continuation[anaphora.end() :].lstrip(" ,:")
    prefix = IMPLICIT_LEFT_OPERATOR_PREFIXES.get(operator)
    return bool(
        prefix is not None and re.match(rf"^{prefix}", continuation) is not None
    ) or _repeats_previous_operation_as_left_operand(
        continuation,
        previous_operation,
        operator,
    )


def _operation_semantics_appear_before(
    brief: str,
    mentions: list[dict[str, Any]],
    operation: ProgramOperation,
    trace: list[ExactNumber | str],
    *,
    operation_index: int,
    previous_operation: ProgramOperation | None,
    after: int,
    before: int,
) -> bool:
    op, arg1, _ = operation
    prefix = brief[:before]
    lowered_prefix = prefix.lower()
    if op.startswith("table_"):
        return (
            _normalize_phrase(arg1) in _normalize_phrase(prefix)
            and any(term in lowered_prefix for term in TABLE_TERMS[op])
        )
    segment = brief[after:before]
    arguments = (arg1, operation[2])
    direct = [argument for argument in arguments if not REF_RE.fullmatch(argument)]
    checked_arguments = (
        list(arguments)
        if op in {"subtract", "divide", "exp", "greater"}
        else direct
    )
    spans = {
        argument: _resolved_argument_spans(
            brief,
            mentions,
            argument,
            trace,
            start=after,
            end=before,
        )
        for argument in checked_arguments
    }
    implicit_left = len(checked_arguments) == 2 and (
        _uses_immediate_previous_result_as_left_operand(
            segment,
            op,
            arguments[0],
            operation_index,
            previous_operation,
        )
    )
    if implicit_left:
        spans[arguments[0]] = sorted(set([*spans[arguments[0]], (after, after)]))
        if op == "subtract" and not spans[arguments[1]]:
            spans[arguments[1]] = _signed_subtraction_rhs_spans(
                mentions,
                arguments[1],
                start=after,
                end=before,
            )
    if any(not spans[argument] for argument in checked_arguments):
        return False
    marker_found = _contains_operation_marker(segment, op)
    if op == "subtract" and implicit_left:
        marker_found = True
    if op == "subtract" and not marker_found:
        marker_found = _binary_minus_separates_operands(
            brief,
            spans[arguments[0]],
            spans[arguments[1]],
        )
    if not marker_found:
        return False
    if len(checked_arguments) < 2:
        return True

    first, second = arguments
    if op in {"subtract", "divide", "exp", "greater"}:
        ordered = any(
            first_end <= second_start
            for _, first_end in spans[first]
            for second_start, _ in spans[second]
        )
        if ordered:
            return True
        if op == "subtract" and "subtract" in segment.lower() and " from " in segment.lower():
            return any(
                second_end <= first_start
                for _, second_end in spans[second]
                for first_start, _ in spans[first]
            )
        return False
    return any(
        left_end <= right_start or right_end <= left_start
        for left_start, left_end in spans[first]
        for right_start, right_end in spans[second]
    )


def brief_trace_is_consistent(
    brief: str,
    program: str,
    trace: list[ExactNumber | str],
    answer: str,
) -> bool:
    parsed = validate_program(program)
    if len(parsed) != len(trace) or not trace:
        return False
    if normalize_program_result(trace[-1]) != answer:
        return False
    normalized_brief = brief.replace("−", "-")
    lowered = normalized_brief.lower()
    mentions = _brief_mentions(normalized_brief)
    cursor = 0
    for index, ((op, arg1, arg2), result) in enumerate(zip(parsed, trace, strict=True)):
        operation = (op, arg1, arg2)
        is_final = index == len(parsed) - 1
        if isinstance(result, str):
            result_match = None
            for match in re.finditer(rf"\b{re.escape(result)}\b", lowered[cursor:]):
                candidate_start = cursor + match.start()
                if _operation_inputs_appear_before(
                    normalized_brief,
                    mentions,
                    operation,
                    operation_index=index,
                    after=cursor,
                    before=candidate_start,
                ) and _operation_semantics_appear_before(
                    normalized_brief,
                    mentions,
                    operation,
                    trace,
                    operation_index=index,
                    previous_operation=parsed[index - 1] if index else None,
                    after=cursor,
                    before=candidate_start,
                ):
                    result_match = match
                    break
            if result_match is None:
                return False
            result_start = cursor + result_match.start()
            result_end = cursor + result_match.end()
        else:
            result_mention = None
            for candidate in mentions:
                if candidate["start"] < cursor or not _mention_matches_result(
                    candidate,
                    result,
                    final_answer=answer if is_final else None,
                ):
                    continue
                if not _operation_inputs_appear_before(
                    normalized_brief,
                    mentions,
                    operation,
                    operation_index=index,
                    after=cursor,
                    before=candidate["start"],
                ):
                    continue
                if not _operation_semantics_appear_before(
                    normalized_brief,
                    mentions,
                    operation,
                    trace,
                    operation_index=index,
                    previous_operation=parsed[index - 1] if index else None,
                    after=cursor,
                    before=candidate["start"],
                ):
                    continue
                result_mention = candidate
                break
            if result_mention is None:
                return False
            result_start = result_mention["start"]
            result_end = result_mention["end"]
        cursor = result_end
    return True


def parse_ground_truth(ground_truth: str | dict[str, Any]) -> dict[str, Any]:
    value = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not isinstance(value, dict):
        raise ValueError("ground_truth must decode to an object")
    required = (
        "id",
        "question",
        "answer",
        "program",
        "brief",
        "table",
        "gold_operation_count",
        "allowed_source_operands",
        "allowed_percentage_literals",
        "allowed_table_row_labels",
        "allowed_constants",
    )
    missing = [field for field in required if field not in value]
    if missing:
        raise ValueError(f"ground_truth missing fields: {', '.join(missing)}")
    if not isinstance(value["table"], list):
        raise ValueError("ground_truth table must be a list")
    for field in ("id", "question", "answer", "program", "brief"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"ground_truth {field} must be a non-empty string")
    for field in (
        "allowed_source_operands",
        "allowed_percentage_literals",
        "allowed_table_row_labels",
        "allowed_constants",
    ):
        if not isinstance(value[field], list) or any(
            not isinstance(item, str) for item in value[field]
        ):
            raise ValueError(f"ground_truth {field} must be a list of strings")
        if len(value[field]) != len(set(value[field])):
            raise ValueError(f"ground_truth {field} contains duplicates")
    if not set(value["allowed_constants"]).issubset(ALLOWED_CONSTANTS):
        raise ValueError("ground_truth allowed_constants contains an invalid constant")
    _approved_keys(value["allowed_source_operands"], percentage=False)
    _approved_keys(value["allowed_percentage_literals"], percentage=True)
    if type(value["gold_operation_count"]) is not int:
        raise ValueError("gold_operation_count must be an integer")
    gold_operations = validate_program(value["program"])
    if value["gold_operation_count"] != len(gold_operations):
        raise ValueError("gold_operation_count does not match program")
    if not 1 <= value["gold_operation_count"] <= 6:
        raise ValueError("gold_operation_count must be between 1 and 6")
    gold_constants = {
        argument
        for op, arg1, arg2 in gold_operations
        if not op.startswith("table_")
        for argument in (arg1, arg2)
        if argument.startswith("const_")
    }
    if gold_constants != set(value["allowed_constants"]):
        raise ValueError("allowed_constants must equal Gold Program constants")
    if not program_graph_is_valid(gold_operations):
        raise ValueError("Gold Program contains unused operations")
    if not program_operands_are_supported(gold_operations, value):
        raise ValueError("Gold Program contains unsupported source operands")
    if not answer_format_is_valid(value["answer"]):
        raise ValueError("Gold Answer is not strictly normalized")
    return value


def preflight_ground_truth(
    ground_truth: str | dict[str, Any],
    *,
    brief_token_counter: Callable[[str], int],
) -> dict[str, Any]:
    gold = parse_ground_truth(ground_truth)
    trace = evaluate_program_trace(gold["program"], gold["table"])
    result = normalize_program_result(trace[-1])
    if result != gold["answer"]:
        raise ValueError(
            f"Gold Program result {result} does not match Gold Answer {gold['answer']}"
        )
    if not brief_is_compliant(
        gold["brief"],
        brief_token_counter,
        operation_count=gold["gold_operation_count"],
    ):
        raise ValueError("Gold Brief fails format or token limit")
    if not brief_trace_is_consistent(
        gold["brief"], gold["program"], trace, gold["answer"]
    ):
        raise ValueError("Gold Brief is inconsistent with Gold Program trace")
    return gold


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
        bool(answer_format_valid) and parsed["answer"] == gold["answer"]
    )
    program_parse_valid = 0.0
    program_executable = 0.0
    program_step_count_valid = 0.0
    program_graph_valid = 0.0
    program_operands_supported = 0.0
    program_result_correct = 0.0
    program_answer_consistent = 0.0
    program_chain_correct = 0.0
    final_value_exact = 0.0
    generated_operation_count: int | None = None
    generated_trace: list[ExactNumber | str] | None = None
    if parsed["program"]:
        try:
            generated_operations = validate_program(parsed["program"])
            generated_operation_count = len(generated_operations)
            program_parse_valid = 1.0
            program_step_count_valid = float(1 <= generated_operation_count <= 6)
            program_graph_valid = float(program_graph_is_valid(generated_operations))
            program_operands_supported = float(
                program_operands_are_supported(generated_operations, gold)
            )
            generated_trace = evaluate_program_trace(parsed["program"], gold["table"])
            result = normalize_program_result(generated_trace[-1])
            program_executable = 1.0
            program_result_correct = float(result == gold["answer"])
            program_answer_consistent = float(
                bool(answer_format_valid) and result == parsed["answer"]
            )
            final_value_exact = float(
                bool(program_result_correct) and bool(program_answer_consistent)
            )
            program_chain_correct = float(
                bool(program_step_count_valid)
                and bool(program_graph_valid)
                and bool(program_operands_supported)
                and bool(program_result_correct)
                and bool(program_answer_consistent)
            )
        except (ArithmeticError, InvalidOperation, OverflowError, TypeError, ValueError):
            pass

    brief_trace_consistent = 0.0
    if (
        parsed["brief_labeled"]
        and generated_trace is not None
        and generated_operation_count is not None
        and bool(program_step_count_valid)
        and brief_is_compliant(
            parsed["brief"],
            brief_token_counter,
            operation_count=generated_operation_count,
        )
        and brief_trace_is_consistent(
            parsed["brief"],
            parsed["program"],
            generated_trace,
            parsed["answer"],
        )
    ):
        brief_trace_consistent = 1.0

    strict_format = float(parsed["strict_three_field_format"])
    chain_valid = float(
        bool(strict_format)
        and bool(program_chain_correct)
        and bool(brief_trace_consistent)
    )
    score = (
        0.50 * answer_correct
        + 0.45 * program_chain_correct
        + 0.025 * brief_trace_consistent
        + 0.025 * strict_format
    )
    return {
        "score": float(round(score, 10)),
        "answer_format_valid": answer_format_valid,
        "answer_correct": answer_correct,
        "program_parse_valid": program_parse_valid,
        "program_executable": program_executable,
        "program_step_count_valid": program_step_count_valid,
        "program_graph_valid": program_graph_valid,
        "program_operands_supported": program_operands_supported,
        "program_result_correct": program_result_correct,
        "program_answer_consistent": program_answer_consistent,
        "program_chain_correct": program_chain_correct,
        "brief_compliant": brief_trace_consistent,
        "brief_trace_consistent": brief_trace_consistent,
        "final_value_exact": final_value_exact,
        "strict_three_field_format": strict_format,
        "chain_valid": chain_valid,
    }
