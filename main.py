
import argparse
import json
from pathlib import Path

from workflow.pipeline import run_pipeline
from config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Synthetic Invoice Dataset Framework"
    )

    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--category", default="invoice")
    parser.add_argument(
        "--mode",
        choices=["generator_only", "full"],
        default="generator_only",
    )

    args = parser.parse_args()

    results = run_pipeline(
        category=args.category,
        count=args.count,
        mode=args.mode,
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "generated_dataset.jsonl"

    saved = 0

    with output_file.open("w", encoding="utf-8") as file:
        for result in results:
            if result["status"] in ("accepted", "generated"):
                example = result["example"]

                record = {
                    "instruction": example["instruction"],
                    "input": example["source_document"],
                    "output": example["ground_truth"],
                }

                file.write(json.dumps(record, ensure_ascii=False) + "\n")
                saved += 1

    print("\nPIPELINE SUMMARY")
    print("----------------")
    print("Total examples:", len(results))
    print("Saved examples:", saved)

    for result in results:
        print(result["id"], ":", result["status"])

    print("\nOutput saved to:", output_file)


if __name__ == "__main__":
    main()
