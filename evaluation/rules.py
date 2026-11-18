
"""Rule-based (non-LLM) validation checks (Member 2)."""

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from schemas.dataset_schema import DatasetExample


PLACEHOLDER_PATTERNS = [
    r"lorem ipsum",
    r"\bTODO\b",
    r"\bTBD\b",
    r"\bplaceholder\b",
    r"\bfoo\b",
    r"\bexample\.com\b",
    r"\bXXX+\b",
    r"<[^>]+>",
]

MIN_INPUT_WORDS = 5
MAX_INPUT_WORDS = 400


@dataclass
class RuleViolation:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def _date_matches(value: str, input_text: str) -> bool:
    """Compare dates while supporting multiple date formats."""

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%d.%m.%Y",
        "%d/%m/%y",
        "%m/%d/%y",
        "%d %B %Y",
        "%d %b %Y",
        "%B %d %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%b %d, %Y",
    ]

    def parse_dates(candidate: str) -> set:
        results = set()

        for fmt in formats:
            try:
                results.add(
                    datetime.strptime(candidate.strip(), fmt).date()
                )
            except ValueError:
                continue

        return results

    expected_dates = parse_dates(value)

    if not expected_dates:
        return False

    pattern = (
        r"\b(?:"
        r"\d{4}-\d{1,2}-\d{1,2}|"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"\d{1,2}\.\d{1,2}\.\d{4}|"
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}|"
        r"[A-Za-z]+\s+\d{1,2},?\s+\d{4}"
        r")\b"
    )

    candidates = re.findall(pattern, input_text)

    for candidate in candidates:
        if expected_dates.intersection(parse_dates(candidate)):
            return True

    return False


def _amount_matches(value: str, input_text: str) -> bool:
    """Compare amounts across common international number formats."""

    try:
        expected = Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return False

    candidates = re.findall(
        r"(?<![A-Za-z0-9])[-+]?\s*[₹$£€]?\s*\d[\d.,]*",
        input_text,
    )

    for candidate in candidates:
        cleaned = re.sub(r"[₹$£€\s]", "", candidate).rstrip(".,")

        # European format: 3.210,50 -> 3210.50
        if "," in cleaned and "." in cleaned:
            if cleaned.rfind(",") > cleaned.rfind("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")

        elif "," in cleaned:
            parts = cleaned.split(",")

            if len(parts) == 2 and len(parts[1]) == 3:
                cleaned = cleaned.replace(",", "")
            else:
                cleaned = cleaned.replace(",", ".")

        try:
            if Decimal(cleaned) == expected:
                return True
        except InvalidOperation:
            continue

    return False


def _is_grounded(key: str, value: str, input_text: str) -> bool:
    """Check literal or format-equivalent evidence in the source document."""

    field = key.lower()

    if field in {"date", "due_date"}:
        return _date_matches(value, input_text)

    if field == "amount":
        return _amount_matches(value, input_text)

    return _norm(value) in _norm(input_text)


def check_example(ex: DatasetExample) -> list[RuleViolation]:
    """Run all deterministic checks on a schema-valid example."""

    violations: list[RuleViolation] = []
    words = len(ex.input_text.split())

    # Check input length
    if words < MIN_INPUT_WORDS:
        violations.append(
            RuleViolation(
                "input_too_short",
                f"input has {words} words (min {MIN_INPUT_WORDS})",
            )
        )

    if words > MAX_INPUT_WORDS:
        violations.append(
            RuleViolation(
                "input_too_long",
                f"input has {words} words (max {MAX_INPUT_WORDS})",
            )
        )

    # Check placeholder text
    for field, text in (
        ("instruction", ex.instruction),
        ("input_text", ex.input_text),
    ):
        for pattern in PLACEHOLDER_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(
                    RuleViolation(
                        "placeholder_text",
                        f"{field} contains placeholder pattern '{pattern}'",
                    )
                )
                break

    # Check expected output values against source document
    for key, val in ex.expected_output.items():
        if val is None:
            continue

        val = str(val)

        if not val.strip():
            violations.append(
                RuleViolation(
                    "empty_value",
                    f"'{key}' is an empty string (use null for absent fields)",
                )
            )

        elif not _is_grounded(key, val, ex.input_text):
            violations.append(
                RuleViolation(
                    "ungrounded_value",
                    f"'{key}' value '{val}' does not appear in input_text",
                )
            )

    # Check all-null output
    if all(val is None for val in ex.expected_output.values()):
        violations.append(
            RuleViolation(
                "all_null_output",
                "every expected field is null",
            )
        )

    # Check duplicate required fields
    if len(set(map(str.lower, ex.required_fields))) != len(
        ex.required_fields
    ):
        violations.append(
            RuleViolation(
                "duplicate_fields",
                "required_fields contains duplicates",
            )
        )

    return violations


def validate_example(
    raw: dict[str, Any],
) -> tuple[DatasetExample | None, list[str]]:
    """Validate a raw dict. Returns (model or None, list of error strings)."""

    try:
        example = DatasetExample.model_validate(raw)

    except ValidationError as error:
        errors = [
            f"[schema] {'.'.join(map(str, item['loc'])) or 'root'}: "
            f"{item['msg']}"
            for item in error.errors()
        ]

        return None, errors

    return example, [
        str(violation)
        for violation in check_example(example)
    ]
