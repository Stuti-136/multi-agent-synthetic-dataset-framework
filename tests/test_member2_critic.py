import json
from pathlib import Path

import pytest

from agents.critic import compare_outputs, critique, decide
from evaluation.rules import validate_example
from schemas.dataset_schema import CriticScores

DATA = json.loads((Path(__file__).parent / "flawed_examples.json").read_text(encoding="utf-8"))


def llm_scores(c=5, k=5, u=5, d=5, fb=""):
    return lambda prompt: {"clarity": c, "correctness": k, "unambiguity": u, "difficulty_match": d, "feedback": fb}


@pytest.mark.parametrize("raw", DATA["good"], ids=lambda r: r["id"])
def test_good_examples_pass_rules(raw):
    ex, errs = validate_example(raw)
    assert ex is not None and errs == []


@pytest.mark.parametrize("raw", [r for r in DATA["flawed"] if r["caught_by"] == "rules"], ids=lambda r: r["id"])
def test_rule_flawed_examples_are_flagged(raw):
    ex, errs = validate_example(raw)
    assert errs, "rule-detectable flaw was not caught"


def test_critique_accepts_good_example():
    ex, _ = validate_example(DATA["good"][0])
    res = critique(ex, predicted=ex.expected_output, llm_fn=llm_scores())
    assert res.decision == "accept" and res.exact_match and res.field_f1 == 1.0


def test_rule_violation_short_circuits_llm():
    ex, _ = validate_example(DATA["flawed"][0])
    def boom(prompt): raise AssertionError("LLM should not be called")
    res = critique(ex, llm_fn=boom)
    assert res.decision == "revise" and res.rule_violations


def test_low_scores_reject_and_mid_scores_revise():
    ex, _ = validate_example(DATA["good"][1])
    assert critique(ex, llm_fn=llm_scores(1, 2, 1, 2)).decision == "reject"
    assert critique(ex, llm_fn=llm_scores(4, 3, 3, 3)).decision == "revise"


def test_low_extractor_agreement_forces_revise():
    ex, _ = validate_example(DATA["good"][0])
    bad_pred = {"vendor": "Nope", "invoice_date": "x", "total_amount": "y"}
    assert critique(ex, predicted=bad_pred, llm_fn=llm_scores()).decision == "revise"


def test_decide_floor():
    assert decide(CriticScores(clarity=5, correctness=5, unambiguity=2, difficulty_match=5), None, None)[0] == "revise"


def test_compare_outputs_normalises_and_handles_none():
    assert compare_outputs({"a": "X  Y", "b": None}, {"a": "x y", "b": None}) == (True, 1.0)
    assert compare_outputs({"a": "x"}, None) == (False, 0.0)


def test_schema_rejects_extra_and_missing_fields():
    raw = dict(DATA["good"][0]); raw["expected_output"] = {"vendor": "Acme Corp"}
    ex, errs = validate_example(raw)
    assert ex is None and errs
