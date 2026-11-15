"""Generator agent: creates one synthetic extraction example per call (Member 1).

Contract (shared with the team):
    generate_example(recipe, feedback=None) -> dict

Extra optional arguments (all keyword-friendly, safe to ignore):
    previous_example  the example the Critic asked to revise. When given, the revision
                      prompt is used so the Generator fixes that example.
    recent            examples (or short summaries) to avoid repeating. If omitted, the
                      Generator remembers its own last few outputs automatically.
    llm_fn            function prompt -> parsed dict. Defaults to llm.call_llm_json
                      (Member 3's wrapper). Used by the tests to fake the LLM.

Demo with a real LLM (needs llm.py):
    python -m agents.generator --count 3
"""
from __future__ import annotations

import argparse
import inspect
import json
from collections import deque
from pathlib import Path
from typing import Any, Callable

from recipes import as_recipe, build_recipes, describe_recipe

try:  # settings live in Member 3's config.py; fall back to defaults if absent
    import config as _cfg
except ImportError:  # pragma: no cover
    _cfg = None

GENERATOR_TEMPERATURE = getattr(_cfg, "GENERATOR_TEMPERATURE", 0.9)  # needs variety
MAX_SHAPE_RETRIES = getattr(_cfg, "GENERATOR_MAX_RETRIES", 2)        # regenerations on bad output
RECENT_LIMIT = 5

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
REQUIRED_KEYS = ("source_document", "instruction", "json_schema", "ground_truth")
SCHEMA_TYPES = {"string", "number", "date"}
ID_PREFIX = {"invoice": "INV"}

LLMFn = Callable[[str], dict]
_RECENT: deque = deque(maxlen=RECENT_LIMIT)


class GenerationError(Exception):
    """Raised when the Generator cannot produce a structurally valid example."""


def reset_recent() -> None:
    """Forget the remembered recent examples (call at the start of a new run)."""
    _RECENT.clear()


def summarize_example(example: dict[str, Any], max_words: int = 14) -> str:
    words = str(example.get("source_document", "")).split()
    return f"[{example.get('edge_case', '?')}] " + " ".join(words[:max_words])


def _recent_block(recent: list[Any] | None) -> str:
    items = list(_RECENT) if recent is None else [
        summarize_example(x) if isinstance(x, dict) else str(x) for x in recent[-RECENT_LIMIT:]]
    return "\n".join(f"- {s}" for s in items) if items else "- (none yet)"


def _load_prompt(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def build_prompt(recipe: Any, feedback: str | None = None, previous_example: dict | None = None,
                 recent: list[Any] | None = None, retry_note: str | None = None) -> str:
    """Fill the generator prompt (fresh example) or the revision prompt (fix a rejected one)."""
    r = as_recipe(recipe)
    notes: list[str] = []
    if retry_note:
        notes.append(f"NOTE: your previous reply was rejected because {retry_note}. Fix this.")

    if previous_example is not None:
        text = _load_prompt("revision_prompt.txt")
        replacements = {
            "<<PREVIOUS_EXAMPLE>>": json.dumps(
                {k: previous_example[k] for k in REQUIRED_KEYS if k in previous_example},
                ensure_ascii=False, indent=2),
            "<<FEEDBACK>>": feedback or "The critic gave no specific feedback; improve overall quality.",
        }
    else:
        text = _load_prompt("generator_prompt.txt")
        if feedback:
            notes.append(f"NOTE: avoid these problems found in an earlier attempt: {feedback}")
        replacements = {"<<RECENT_EXAMPLES>>": _recent_block(recent)}

    replacements.update({
        "<<CATEGORY>>": r.category,
        "<<RECIPE_BLOCK>>": describe_recipe(r),
        "<<EXTRA_NOTES>>": ("\n".join(notes) + "\n") if notes else "",
    })
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


def _check_shape(raw: Any) -> None:
    """Light structural check of the LLM reply. Full validation is Member 2's validate_example()."""
    if not isinstance(raw, dict):
        raise ValueError("the reply was not a JSON object")
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"these keys are missing: {missing}")
    if not isinstance(raw["source_document"], str) or len(raw["source_document"].strip()) < 20:
        raise ValueError("source_document must be a text of at least 20 characters")
    if not isinstance(raw["instruction"], str) or len(raw["instruction"].strip()) < 10:
        raise ValueError("instruction must be a sentence of at least 10 characters")
    schema, truth = raw["json_schema"], raw["ground_truth"]
    if not isinstance(schema, dict) or not schema:
        raise ValueError("json_schema must be a non-empty object")
    if not isinstance(truth, dict):
        raise ValueError("ground_truth must be an object")
    bad_types = {k: v for k, v in schema.items() if str(v).strip().lower() not in SCHEMA_TYPES}
    if bad_types:
        raise ValueError(f"json_schema types must be string, number or date (got {bad_types})")
    if set(schema) != set(truth):
        raise ValueError(f"json_schema keys {sorted(schema)} and ground_truth keys {sorted(truth)} differ")


def _default_llm(prompt: str) -> dict:  # pragma: no cover
    from llm import call_llm_json  # Member 3's wrapper: prompt -> parsed dict
    params = inspect.signature(call_llm_json).parameters
    if "temperature" in params or any(p.kind is p.VAR_KEYWORD for p in params.values()):
        return call_llm_json(prompt, temperature=GENERATOR_TEMPERATURE)
    return call_llm_json(prompt)


def generate_example(recipe: Any, feedback: str | None = None, previous_example: dict | None = None,
                     recent: list[Any] | None = None, llm_fn: LLMFn | None = None) -> dict[str, Any]:
    """Generate (or revise) one example for `recipe`. Returns the internal record fields
    the Generator owns: id, category, difficulty, edge_case, style, locale, source_document,
    instruction, json_schema, ground_truth. The pipeline adds attempts, scores, etc."""
    r = as_recipe(recipe)
    llm = llm_fn or _default_llm
    retry_note: str | None = None
    last_error = "unknown error"

    for _ in range(MAX_SHAPE_RETRIES + 1):
        prompt = build_prompt(r, feedback, previous_example, recent, retry_note)
        try:
            raw = llm(prompt)
            _check_shape(raw)
        except (ValueError, TypeError) as err:  # JSONDecodeError is a ValueError
            last_error = str(err)
            retry_note = last_error
            continue
        example = {
            "id": f"{ID_PREFIX.get(r.category, r.category[:3].upper())}_{r.index:03d}",
            "category": r.category,
            "difficulty": r.difficulty,
            "edge_case": r.edge_case,
            "style": r.style,
            "locale": r.locale,
            "source_document": raw["source_document"].strip(),
            "instruction": raw["instruction"].strip(),
            "json_schema": {k: str(v).strip().lower() for k, v in raw["json_schema"].items()},
            "ground_truth": raw["ground_truth"],
        }
        _RECENT.append(summarize_example(example))
        return example

    raise GenerationError(f"could not generate a valid example for recipe {r.index} "
                          f"after {MAX_SHAPE_RETRIES + 1} tries: {last_error}")


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser(description="Generate a few examples with the real LLM.")
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    for recipe in build_recipes(args.count, args.seed):
        print(json.dumps(generate_example(recipe), indent=2, ensure_ascii=False))
