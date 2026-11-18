# Member 4 – Evaluation, Export and Testing Report

## 1. Role
Responsible for evaluation, validation, dataset exports, metrics and testing.

## 2. Experiment Setup
- Arm A: Generator-only approach
- Arm B: Full multi-agent approach
- 15 examples per arm
- Both arms used shared recipe specifications.
- 13 fixed evaluation files were maintained separately from live generation.

## 3. Original Experiment Results

| Metric | Arm A | Arm B |
|---|---:|---:|
| Total examples | 15 | 15 |
| Generated | 15 | 0 |
| Accepted | N/A | 1 |
| Needs revision | N/A | 14 |
| Rejected | 0 | 0 |
| Average attempts | 1.00 | 2.87 |
| Exact match rate | N/A | 66.67% |
| Mean field F1 | N/A | 0.9044 |
| Duplicate rate | 0% | 0% |

Note: Arm A's generated status is not equivalent to Arm B's accepted status.

## 4. Offline Deterministic Revalidation
After correcting date and amount matching rules, all 15 saved Arm B examples passed offline deterministic validation.

- Total checked: 15
- Passed: 15
- Failed: 0
- Pass rate: 100%

This is a separate revalidation result. It does not replace the original multi-agent acceptance metrics.

## 5. Automated Testing
- Test framework: pytest
- Tests passed: 26
- Tests failed: 0

## 6. Dataset Exports
The following files were generated from saved experiment results:

- generated.jsonl – 15 Arm A examples
- accepted.jsonl – 1 Arm B accepted example
- needs_revision_rejected.jsonl – 14 Arm B examples needing revision or rejected
- Arm_B_all_statuses.jsonl – all 15 Arm B examples

## 7. Limitations
- Pilot sample size is 15 examples per arm.
- Results are preliminary and should not be generalized.
- Offline deterministic revalidation does not mean the original multi-agent critic accepted every example.
- The original experiment used Gemini for earlier saved work and Groq for the fresh hybrid experiment; provider differences should be considered when interpreting results.

## 8. Conclusion
The evaluation and export pipeline was tested successfully. All 26 automated tests passed, and all 15 saved Arm B examples passed the updated offline deterministic validation. The original experiment metrics are preserved separately.
