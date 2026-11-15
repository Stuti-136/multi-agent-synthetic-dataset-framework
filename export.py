import json
from pathlib import Path


def export_jsonl(results, output_path, include_rejected=False):
    """
    Export generated examples in the project's standard JSONL format:
    instruction, input, output.

    By default, only accepted (Arm B) and generated (Arm A) examples
    are exported. Set include_rejected=True to include other statuses.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_statuses = {"accepted", "generated"}
    if include_rejected:
        allowed_statuses.update({"rejected", "needs_revision"})

    exported = 0

    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            if result.get("status") not in allowed_statuses:
                continue

            example = result.get("example")
            if not isinstance(example, dict):
                continue

            record = {
                "instruction": example.get("instruction", ""),
                "input": example.get("source_document", ""),
                "output": example.get("ground_truth", {}),
            }

            file.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported += 1

    return exported
