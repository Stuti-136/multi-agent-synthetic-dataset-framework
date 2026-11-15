# Member 1 - Generator & Recipes Lead (branch: feature/generator)

Files
- `recipes.py` - the 8 edge cases, styles, locales, difficulties, `build_recipes()`, `check_coverage()`
- `agents/generator.py` - `generate_example(recipe, feedback=None)`
- `evaluation/diversity.py` - `check_duplicate(example, accepted_list)`
- `prompts/generator_prompt.txt`, `prompts/revision_prompt.txt`
- `docs/prompt_log.md` - shared prompt log (everyone adds their own prompt versions)
- `tests/test_member1_generator.py`, `tests/sample_generated.json` (3 hand-written examples in the exact format)

Run from the repo root (needs `pip install pytest`; `sentence-transformers` is optional for the tests):

    python -m pytest tests/test_member1_generator.py
    python -m recipes --count 50 --seed 42        # coverage table, no LLM needed
    python -m agents.generator --count 3          # real LLM, needs llm.py

## What the Generator returns (follows the team plan v2)

    id, category, difficulty, edge_case, style, locale,
    source_document, instruction, json_schema, ground_truth

- `json_schema` maps field name to `"string"`, `"number"` or `"date"` (a date is a YYYY-MM-DD string).
- `ground_truth` has exactly the same keys as `json_schema`. Dates are YYYY-MM-DD, amounts are plain
  numbers, missing fields are `null`, text values are copied exactly from the document.
- The pipeline adds `predicted_output`, `scores`, `decision`, `feedback`, `attempts`, etc.
- Export mapping: `instruction` -> `instruction`, `source_document` -> `input`, `ground_truth` -> `output`.

## Integration notes for the team

**Member 3 (llm.py, config.py, pipeline)**
- `llm.call_llm_json(prompt, temperature=...) -> dict` is expected. The Generator passes
  `temperature=0.9` if the function accepts it (Extractor and Critic should use 0).
- Optional `config.py` names (defaults in brackets): `GENERATOR_TEMPERATURE` (0.9),
  `GENERATOR_MAX_RETRIES` (2 regenerations when the reply is malformed),
  `SIMILARITY_THRESHOLD` (0.90), `EMBEDDING_MODEL` ("all-MiniLM-L6-v2").
- Pipeline usage: `recipes = build_recipes(50)`, then `check_coverage(recipes)`, call `reset_recent()`
  at the start of a run, then `generate_example(recipe)` per recipe.
- For a REVISE: `generate_example(recipe, feedback=critic_feedback, previous_example=example)`.
  `previous_example` is one extra optional argument beyond the shared contract.
- Catch `agents.generator.GenerationError` and log the example as rejected with that reason.

**Member 2 (schema, rules, critic) - please align with the plan**
- Use the plan's field names: `source_document`, `ground_truth`, `json_schema`; the recipe fields are
  `edge_case`, `style`, `locale`, `difficulty` (and `category`).
- `ground_truth` values are strings, numbers or null. Only `"string"` fields are checked for appearing in
  the document; dates and amounts are normalized before comparing and must not be string-searched.
- Critic rubric from the plan: accuracy, clarity, realism, edge_case_quality, toxicity (safe/unsafe),
  plus the label_wrong / extractor_wrong / ambiguous_document verdict. Python applies the thresholds.
- `tests/sample_generated.json` is a ready-made set of valid examples to test your validator on.

**Member 4 (export, metrics, tests)**
- `evaluation/diversity.py` has `duplicate_rate(examples)` for the duplicate-rate metric.

## Notes
- Duplicate check: exact match after normalizing text (digits become #), then embedding similarity.
  `>= 0.90` counts as a near-duplicate. Without `sentence-transformers` a bag-of-words fallback runs
  and prints a warning; use the real model for the experiments.
- The recipe lists live in `recipes.py` (not `config.py`) so Member 3's file is not edited by two people.
  `config.py` can import them from `recipes.py`.
