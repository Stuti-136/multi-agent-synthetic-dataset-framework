"""Critic agent: combines rule checks, extractor agreement and an LLM rubric (Member 2)."""
import json
from pathlib import Path
from typing import Any, Callable

from evaluation.rules import check_example, _norm
from schemas.dataset_schema import CriticScores, CritiqueResult, DatasetExample

try:  # thresholds live in Member 3's config.py; fall back to defaults if absent
    import config as _cfg
except ImportError:  # pragma: no cover
    _cfg = None

ACCEPT_MEAN = getattr(_cfg, "CRITIC_ACCEPT_MEAN", 4.0)      # mean >= this (and floor ok) -> accept
ACCEPT_FLOOR = getattr(_cfg, "CRITIC_ACCEPT_FLOOR", 3)       # no single score below this
REJECT_BELOW = getattr(_cfg, "CRITIC_REJECT_BELOW", 2.0)     # mean < this -> reject (unfixable)
MIN_AGREEMENT_F1 = getattr(_cfg, "CRITIC_MIN_AGREEMENT_F1", 0.5)

PROMPT_PATH = Path(__file__).resolve().parent.parent / "prompts" / "critic_prompt.txt"
LLMFn = Callable[[str], dict]  # takes prompt, returns parsed JSON dict


def compare_outputs(expected: dict[str, Any], predicted: dict[str, Any] | None) -> tuple[bool, float]:
    """Return (exact_match, field-level F1) between gold and extractor output."""
    if predicted is None:
        return False, 0.0
    def n(v): return None if v is None else _norm(str(v))
    exp = {k: n(v) for k, v in expected.items()}
    pred = {k: n(v) for k, v in predicted.items()}
    exact = exp == pred
    exp_nn = {k: v for k, v in exp.items() if v is not None}
    pred_nn = {k: v for k, v in pred.items() if v is not None}
    correct = sum(1 for k, v in pred_nn.items() if exp_nn.get(k) == v)
    if not exp_nn and not pred_nn:
        return exact, 1.0
    p = correct / len(pred_nn) if pred_nn else 0.0
    r = correct / len(exp_nn) if exp_nn else 0.0
    return exact, (2 * p * r / (p + r) if p + r else 0.0)


def build_prompt(example: DatasetExample, predicted: dict[str, Any] | None) -> str:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    repl = {
        "<<DIFFICULTY>>": example.recipe.difficulty,
        "<<INSTRUCTION>>": example.instruction,
        "<<INPUT_TEXT>>": example.input_text,
        "<<EXPECTED_OUTPUT>>": json.dumps(example.expected_output, ensure_ascii=False),
        "<<PREDICTED_OUTPUT>>": json.dumps(predicted, ensure_ascii=False) if predicted is not None else "none",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def _default_llm(prompt: str) -> dict:  # pragma: no cover
    from llm import call_llm_json  # Member 3's wrapper: prompt -> parsed dict
    return call_llm_json(prompt)


def decide(scores: CriticScores, exact_match: bool | None, f1: float | None) -> tuple[str, str]:
    """Threshold logic. Returns (decision, extra_feedback)."""
    if scores.mean < REJECT_BELOW:
        return "reject", "Mean rubric score too low to salvage."
    if scores.mean >= ACCEPT_MEAN and scores.minimum >= ACCEPT_FLOOR:
        if f1 is not None and f1 < MIN_AGREEMENT_F1:
            return "revise", (f"Extractor agreement is low (F1={f1:.2f}); make the text and gold "
                              "answer less ambiguous or easier to extract.")
        return "accept", ""
    return "revise", ""


def critique(example: DatasetExample, predicted: dict[str, Any] | None = None,
             llm_fn: LLMFn | None = None) -> CritiqueResult:
    """Judge one example. Rule failures short-circuit before any LLM call."""
    violations = [str(v) for v in check_example(example)]
    exact, f1 = compare_outputs(example.expected_output, predicted) if predicted is not None else (None, None)
    if violations:
        return CritiqueResult(decision="revise", exact_match=exact, field_f1=f1,
                              rule_violations=violations,
                              feedback="Fix these issues: " + "; ".join(violations))
    raw = (llm_fn or _default_llm)(build_prompt(example, predicted))
    scores = CriticScores.model_validate(raw)
    decision, extra = decide(scores, exact, f1)
    feedback = " ".join(x for x in (scores.feedback, extra) if x)
    return CritiqueResult(decision=decision, scores=scores, mean_score=round(scores.mean, 2),
                          exact_match=exact, field_f1=f1, feedback=feedback)
