import json
from pathlib import Path

from workflow.pipeline import run_pipeline
from export import export_jsonl
from evaluation.metrics import (
    compute_metrics,
    write_metrics_csv,
    plot_metrics_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "experiments"


def save_results(results, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2, default=str)


def run_experiments(count=50):
    """
    Arm A: generator_only.
    Arm B: full Generator -> Extractor -> Critic pipeline.

    Both calls use build_recipes' default deterministic seed, so
    the same recipe sequence is used for both arms.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiment_arms = {
        "Arm_A_Generator_Only": "generator_only",
        "Arm_B_Full_Multi_Agent": "full",
    }

    metrics_by_arm = {}

    for arm_name, mode in experiment_arms.items():
        print(f"\nRunning {arm_name}: mode={mode}, count={count}")

        results = run_pipeline(
            category="invoice",
            count=count,
            mode=mode,
        )

        save_results(
            results,
            OUTPUT_DIR / f"{arm_name}_results.json",
        )

        exported_count = export_jsonl(
            results,
            OUTPUT_DIR / f"{arm_name}_dataset.jsonl",
        )

        metrics = compute_metrics(results, arm_name=arm_name)
        metrics["exported"] = exported_count
        metrics_by_arm[arm_name] = metrics

        print(f"Completed {arm_name}: {len(results)} results")
        print(f"Exported examples: {exported_count}")
        print(f"Metrics: {metrics}")

    write_metrics_csv(
        metrics_by_arm,
        OUTPUT_DIR / "comparison_metrics.csv",
    )

    plot_metrics_comparison(
        metrics_by_arm,
        OUTPUT_DIR / "comparison_chart.png",
    )

    print("\nExperiment completed.")
    print(f"Output folder: {OUTPUT_DIR}")

    return metrics_by_arm


if __name__ == "__main__":
    run_experiments(count=50)
