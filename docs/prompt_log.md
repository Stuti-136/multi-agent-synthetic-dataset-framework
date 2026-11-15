# Prompt log

One entry per prompt version. For every version write: what changed, what failed in the previous
version (with an example id), and what improved. Keep failed versions; they are part of the grade.

---

## Generator prompt (Member 1) - `prompts/generator_prompt.txt`

### v1 (2026-11-11)
- Based on the plan: recipe block (difficulty, edge case, style, locale), four outputs
  (`source_document`, `instruction`, `json_schema`, `ground_truth`), rules for grounding, nulls,
  YYYY-MM-DD dates, plain-number amounts, fictional names, no repeats of recent examples.
- Temperature 0.9. One example per call.
- Known risks to watch: edge case present but too weak; ground truth not supported by the
  document; documents that look too similar even with different recipes; amounts with symbols.

### v2 (fill in after the first 10 to 50 generated examples)
- Failure seen (example ids):
- What I changed:
- Result after the change:

---

## Revision prompt (Member 1) - `prompts/revision_prompt.txt`

### v1 (2026-11-11)
- Input: previous example, Critic feedback, same recipe.
- Key rule: rewrite `ground_truth` from the final document; never copy the Extractor's prediction.

### v2 (fill in after testing the loop with real Critic feedback)
- Failure seen (example ids):
- What I changed:
- Result after the change:

---

## Extractor prompt (Member 3) - `prompts/extractor_prompt.txt`

### v1
- (fill in)

### v2
- Failure seen:
- What I changed:
- Result after the change:

---

## Critic prompt (Member 2) - `prompts/critic_prompt.txt`

### v1
- (fill in)

### v2 (after calibration on the 6 flawed and 6 good examples)
- Failure seen (calibration example ids):
- What I changed:
- Result after the change:
