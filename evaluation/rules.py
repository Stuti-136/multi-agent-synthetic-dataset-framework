"""Rule-based (non-LLM) validation checks (Member 2)."""
import re
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from schemas.dataset_schema import DatasetExample

PLACEHOLDER_PATTERNS = [r"lorem ipsum", r"\bTODO\b", r"\bTBD\b", r"\bplaceholder\b",
                        r"\bfoo\b", r"\bexample\.com\b", r"\bXXX+\b", r"<[^>]+>"]
MIN_INPUT_WORDS = 5
MAX_INPUT_WORDS = 400


@dataclass
class RuleViolation:
    rule: str
    message: str

    def __str__(self) -> str:
        return f"[{self.rule}] {self.message}"


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def check_example(ex: DatasetExample) -> list[RuleViolation]:
    """Run all deterministic checks on a schema-valid example."""
    v: list[RuleViolation] = []
    words = len(ex.input_text.split())
    if words < MIN_INPUT_WORDS:
        v.append(RuleViolation("input_too_short", f"input has {words} words (min {MIN_INPUT_WORDS})"))
    if words > MAX_INPUT_WORDS:
        v.append(RuleViolation("input_too_long", f"input has {words} words (max {MAX_INPUT_WORDS})"))

    for field, text in (("instruction", ex.instruction), ("input_text", ex.input_text)):
        for pat in PLACEHOLDER_PATTERNS:
            if re.search(pat, text, re.IGNORECASE):
                v.append(RuleViolation("placeholder_text", f"{field} contains placeholder pattern '{pat}'"))
                break

    haystack = _norm(ex.input_text)
    for key, val in ex.expected_output.items():
        if val is None:
            continue
        if not val.strip():
            v.append(RuleViolation("empty_value", f"'{key}' is an empty string (use null for absent fields)"))
        elif _norm(val) not in haystack:
            v.append(RuleViolation("ungrounded_value", f"'{key}' value '{val}' does not appear in input_text"))

    if all(val is None for val in ex.expected_output.values()):
        v.append(RuleViolation("all_null_output", "every expected field is null"))

    if len(set(map(str.lower, ex.required_fields))) != len(ex.required_fields):
        v.append(RuleViolation("duplicate_fields", "required_fields contains duplicates"))
    return v


def validate_example(raw: dict[str, Any]) -> tuple[DatasetExample | None, list[str]]:
    """Validate a raw dict. Returns (model or None, list of error strings)."""
    try:
        ex = DatasetExample.model_validate(raw)
    except ValidationError as e:
        errs = [f"[schema] {'.'.join(map(str, err['loc'])) or 'root'}: {err['msg']}" for err in e.errors()]
        return None, errs
    return ex, [str(x) for x in check_example(ex)]
