"""Tests for Member 1's part. No API key and no model download needed.

Run from the repo root:  python -m pytest tests/test_member1_generator.py
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from agents import generator as gen
from evaluation import diversity as div
import recipes as rc

SAMPLES = json.loads((Path(__file__).parent / "sample_generated.json").read_text(encoding="utf-8"))


def good_reply(**overrides):
    reply = {k: SAMPLES[0][k] for k in gen.REQUIRED_KEYS}
    reply.update(overrides)
    return reply


@pytest.fixture(autouse=True)
def _clean_recent():
    gen.reset_recent()
    yield
    gen.reset_recent()


# ----------------------------- recipes ------------------------------------- #
def test_build_recipes_50_is_balanced():
    recipes = rc.build_recipes(50, seed=42)
    assert [r.index for r in recipes] == list(range(1, 51))
    ok, report = rc.check_coverage(recipes)
    assert ok, report["problems"]
    assert set(Counter(r.edge_case for r in recipes).values()) <= {6, 7}
    assert set(Counter(r.style for r in recipes).values()) == {10}
    assert set(Counter(r.locale for r in recipes).values()) <= {12, 13}
    assert dict(Counter(r.difficulty for r in recipes)) == {"easy": 15, "medium": 20, "hard": 15}


def test_every_edge_case_has_every_difficulty():
    for seed in (1, 2, 3):
        _, report = rc.check_coverage(rc.build_recipes(50, seed=seed))
        for row in report["edge_by_difficulty"].values():
            assert all(c > 0 for c in row.values())


def test_build_recipes_is_deterministic_per_seed():
    assert rc.build_recipes(20, seed=7) == rc.build_recipes(20, seed=7)
    assert rc.build_recipes(20, seed=7) != rc.build_recipes(20, seed=8)


def test_small_runs_pass_coverage():
    for n in (1, 8, 10, 30):
        ok, report = rc.check_coverage(rc.build_recipes(n))
        assert ok, (n, report["problems"])


def test_coverage_fails_when_unbalanced():
    recipes = [rc.Recipe(i, "easy", "missing_field", "email", "usa") for i in range(1, 17)]
    ok, report = rc.check_coverage(recipes)
    assert not ok
    assert any("edge_case" in p for p in report["problems"])


def test_coverage_flags_unknown_values():
    ok, report = rc.check_coverage([rc.Recipe(1, "easy", "not_an_edge_case", "email", "usa")])
    assert not ok and any("unknown edge_case" in p for p in report["problems"])


def test_coverage_table_lists_all_edge_cases():
    table = rc.format_coverage_table(rc.build_recipes(50))
    for name in rc.EDGE_CASES:
        assert name in table


def test_there_are_exactly_8_edge_cases():
    assert len(rc.EDGE_CASES) == 8


# ----------------------------- generator ----------------------------------- #
def test_generate_example_returns_plan_fields():
    recipe = rc.Recipe(1, "easy", "date_format_variation", "paragraph", "india")
    ex = gen.generate_example(recipe, llm_fn=lambda prompt: good_reply())
    assert ex["id"] == "INV_001"
    assert (ex["category"], ex["difficulty"], ex["edge_case"], ex["style"], ex["locale"]) == \
           ("invoice", "easy", "date_format_variation", "paragraph", "india")
    assert set(gen.REQUIRED_KEYS) <= set(ex)
    assert ex["ground_truth"]["date"] == "2026-03-15"


def test_recipe_can_be_a_dict():
    recipe = {"index": 12, "difficulty": "hard", "edge_case": "credit_note", "style": "table", "locale": "uk"}
    ex = gen.generate_example(recipe, llm_fn=lambda prompt: good_reply())
    assert ex["id"] == "INV_012" and ex["edge_case"] == "credit_note"


def test_prompt_contains_the_recipe():
    seen = []
    recipe = rc.Recipe(3, "medium", "multiple_vendors", "bullet_list", "europe")
    gen.generate_example(recipe, llm_fn=lambda p: seen.append(p) or good_reply())
    prompt = seen[0]
    assert "multiple_vendors" in prompt and "bullet_list" in prompt and "Europe" in prompt
    assert "<<" not in prompt  # every placeholder was filled


def test_revision_prompt_uses_previous_example_and_feedback():
    seen = []
    recipe = rc.Recipe(1, "easy", "date_format_variation", "paragraph", "india")
    previous = dict(SAMPLES[0])
    gen.generate_example(recipe, feedback="The amount is not supported by the document.",
                         previous_example=previous, llm_fn=lambda p: seen.append(p) or good_reply())
    prompt = seen[0]
    assert "revising" in prompt
    assert "The amount is not supported by the document." in prompt
    assert previous["source_document"] in prompt
    assert "Never change ground_truth just to match" in prompt
    assert "<<" not in prompt


def test_feedback_without_previous_example_goes_into_fresh_prompt():
    seen = []
    recipe = rc.Recipe(1, "easy", "missing_field", "email", "usa")
    gen.generate_example(recipe, feedback="make it less ambiguous", llm_fn=lambda p: seen.append(p) or good_reply())
    assert "make it less ambiguous" in seen[0] and "revising" not in seen[0]


def test_retries_on_invalid_reply_then_succeeds():
    prompts, replies = [], iter([{"oops": 1}, good_reply()])
    recipe = rc.Recipe(1, "easy", "missing_field", "email", "usa")
    ex = gen.generate_example(recipe, llm_fn=lambda p: prompts.append(p) or next(replies))
    assert ex["id"] == "INV_001"
    assert len(prompts) == 2 and "was rejected because" in prompts[1]


def test_json_decode_error_is_retried():
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 1:
            raise json.JSONDecodeError("bad", "", 0)
        return good_reply()

    gen.generate_example(rc.Recipe(1, "easy", "missing_field", "email", "usa"), llm_fn=flaky)
    assert calls["n"] == 2


@pytest.mark.parametrize("bad", [
    good_reply(ground_truth={"invoice_number": "INV-2041"}),                       # keys differ from schema
    good_reply(json_schema={"invoice_number": "string (any)"}, ground_truth={"invoice_number": "x"}),
    good_reply(json_schema={}, ground_truth={}),
    good_reply(source_document="too short"),
    good_reply(instruction="short"),
    "not a dict",
])
def test_bad_shapes_raise_generation_error_after_retries(bad):
    calls = {"n": 0}

    def always_bad(prompt):
        calls["n"] += 1
        return bad

    with pytest.raises(gen.GenerationError):
        gen.generate_example(rc.Recipe(1, "easy", "missing_field", "email", "usa"), llm_fn=always_bad)
    assert calls["n"] == gen.MAX_SHAPE_RETRIES + 1


def test_recent_examples_are_passed_to_the_next_prompt():
    prompts = []
    llm = lambda p: prompts.append(p) or good_reply()
    gen.generate_example(rc.Recipe(1, "easy", "date_format_variation", "paragraph", "india"), llm_fn=llm)
    gen.generate_example(rc.Recipe(2, "easy", "missing_field", "email", "usa"), llm_fn=llm)
    assert "(none yet)" in prompts[0]
    assert "Sharma Traders issued invoice" in prompts[1]


def test_explicit_recent_overrides_memory():
    prompts = []
    gen.generate_example(rc.Recipe(1, "easy", "missing_field", "email", "usa"),
                         recent=[SAMPLES[1]], llm_fn=lambda p: prompts.append(p) or good_reply())
    assert "BrightPath" in prompts[0]


def test_json_schema_types_are_normalized_to_lowercase():
    reply = good_reply(json_schema={"invoice_number": "String", "vendor": "STRING", "date": "Date", "amount": "Number"})
    ex = gen.generate_example(rc.Recipe(1, "easy", "missing_field", "email", "usa"), llm_fn=lambda p: reply)
    assert ex["json_schema"]["amount"] == "number"


def test_sample_examples_pass_the_shape_check_and_have_unique_ids():
    for sample in SAMPLES:
        gen._check_shape(sample)
        assert sample["edge_case"] in rc.EDGE_CASES and sample["style"] in rc.STYLES
        assert sample["locale"] in rc.LOCALES and sample["difficulty"] in rc.DIFFICULTIES
    assert len({s["id"] for s in SAMPLES}) == len(SAMPLES)


# ----------------------------- diversity ----------------------------------- #
def make_ex(i, text):
    return {"id": f"INV_{i:03d}", "source_document": text}


def fake_embedder(table):
    return lambda texts: [table[t] for t in texts]


def test_normalize_replaces_digits_and_punctuation():
    assert div.normalize_text("Invoice INV-102, due 10 April 2026!") == "invoice inv ### due ## april ####"


def test_exact_duplicate_ignores_numbers_and_case():
    a = make_ex(1, "Invoice INV-102 was issued by ABC Supplies on 10 April 2026. Total: Rs. 15,000.")
    b = make_ex(2, "invoice inv-205 was issued by abc supplies on 11 april 2027. total: rs. 98,500")
    checker = div.DuplicateChecker(embedder=fake_embedder({}))
    is_dup, sim, closest = checker.find_closest(b, [a])
    assert (is_dup, sim, closest) == (True, 1.0, "INV_001")


def test_changing_the_vendor_is_not_an_exact_duplicate():
    a = make_ex(1, "Invoice INV-102 was issued by ABC Supplies on 10 April 2026.")
    b = make_ex(2, "Invoice INV-102 was issued by XYZ Traders on 10 April 2026.")
    table = {a["source_document"]: [1.0, 0.0], b["source_document"]: [0.0, 1.0]}
    is_dup, sim = div.DuplicateChecker(embedder=fake_embedder(table)).check(b, [a])
    assert not is_dup and sim == 0.0


def test_similarity_at_threshold_counts_as_duplicate():
    a, b = make_ex(1, "first document text"), make_ex(2, "second document text")
    table = {a["source_document"]: [1.0, 0.0], b["source_document"]: [0.9, math_sqrt(1 - 0.81)]}
    assert div.DuplicateChecker(threshold=0.90, embedder=fake_embedder(table)).check(b, [a])[0]
    assert not div.DuplicateChecker(threshold=0.95, embedder=fake_embedder(table)).check(b, [a])[0]


def math_sqrt(x):
    return x ** 0.5


def test_no_accepted_examples_means_no_duplicate():
    assert div.DuplicateChecker(embedder=fake_embedder({})).check(make_ex(1, "anything at all"), []) == (False, 0.0)


def test_example_is_not_compared_with_itself():
    a = make_ex(1, "same id example text")
    assert div.DuplicateChecker(embedder=fake_embedder({})).check(a, [a]) == (False, 0.0)


def test_picks_the_closest_example():
    new = make_ex(9, "new document")
    far, near = make_ex(1, "far document"), make_ex(2, "near document")
    table = {"new document": [1.0, 0.0], "far document": [0.0, 1.0], "near document": [0.8, 0.6]}
    is_dup, sim, closest = div.DuplicateChecker(embedder=fake_embedder(table)).find_closest(new, [far, near])
    assert closest == "INV_002" and round(sim, 2) == 0.8 and not is_dup


def test_duplicate_rate_counts_later_copies():
    docs = [make_ex(1, "Invoice A 100 from Acme"), make_ex(2, "Invoice A 200 from Acme"),
            make_ex(3, "Completely different words here")]
    checker = div.DuplicateChecker(embedder=div.bow_embedder)
    assert round(checker.duplicate_rate(docs), 1) == 33.3


def test_bow_fallback_separates_similar_and_unrelated_text():
    v = div.bow_embedder(["invoice issued by acme supplies total due paid",
                          "invoice issued by acme supplies total due unpaid",
                          "completely unrelated sentence about weather today"])
    assert div.cosine(v[0], v[1]) > 0.8 > div.cosine(v[0], v[2])
