# Member 2 - Critic & Validation Lead (branch: feature/critic)

Files: agents/critic.py, evaluation/rules.py, evaluation/calibrate_critic.py,
schemas/dataset_schema.py, prompts/critic_prompt.txt, tests/flawed_examples.json,
tests/test_member2_critic.py

Run from the repo root (needs `pip install pydantic pytest`):
    python -m pytest tests/test_member2_critic.py
    python -m evaluation.calibrate_critic --rules-only
    python -m evaluation.calibrate_critic            # real LLM, needs llm.py

Integration notes
- critic.py calls `llm.call_llm_json(prompt) -> dict` (Member 3). Rename the import in
  `_default_llm` if your wrapper uses a different name.
- Thresholds are read from config.py if present: CRITIC_ACCEPT_MEAN (4.0), CRITIC_ACCEPT_FLOOR (3),
  CRITIC_REJECT_BELOW (2.0), CRITIC_MIN_AGREEMENT_F1 (0.5).
- The schema (DatasetExample) assumes fields: id, recipe{domain,difficulty,style},
  instruction, input_text, required_fields, expected_output. Align with Member 1's generator.
- Log critic_prompt v1/v2 in docs/prompt_log.md.
