import csv
from difflib import SequenceMatcher
from pathlib import Path


def _get_critique_value(result, key):
    critique = result.get("critique")
    if isinstance(critique, dict):
        return critique.get(key)
    return result.get(key)


def calculate_duplicate_rate(results, threshold=0.90):
    """
    Percentage of examples whose source document closely matches
    an earlier example. Uses text similarity, not an LLM.
    """
    documents = []

    for result in results:
        example = result.get("example")
        if isinstance(example, dict):
            text = str(example.get("source_document", "")).strip().lower()
            if text:
                documents.append(" ".join(text.split()))

    if len(documents) < 2:
        return 0.0

    duplicate_count = 0
    for index, document in enumerate(documents):
        for previous in documents[:index]:
            similarity = SequenceMatcher(
                None, document, previous
            ).ratio()

            if similarity >= threshold:
                duplicate_count += 1
                break

    return round((duplicate_count / len(documents)) * 100, 2)


def compute_metrics(results, arm_name=""):
    """Calculate comparable metrics for either experiment arm."""
    total = len(results)

    statuses = [
        result.get("status", "unknown")
        for result in results
    ]

    accepted = statuses.count("accepted")
    generated = statuses.count("generated")
    rejected = statuses.count("rejected")
    needs_revision = statuses.count("needs_revision")

    attempts = [
        float(result["attempts"])
        for result in results
        if result.get("attempts") is not None
    ]

    exact_match_values = []
    field_f1_values = []

    for result in results:
        exact_match = _get_critique_value(result, "exact_match")
        field_f1 = _get_critique_value(result, "field_f1")

        if isinstance(exact_match, bool):
            exact_match_values.append(exact_match)

        if isinstance(field_f1, (int, float)):
            field_f1_values.append(float(field_f1))

    success_count = accepted + generated

    def percentage(value):
        return round((value / total) * 100, 2) if total else 0.0

    return {
        "arm": arm_name,
        "total": total,
        "accepted": accepted,
        "generated": generated,
        "rejected": rejected,
        "needs_revision": needs_revision,
        "success_rate": percentage(success_count),
        "accepted_rate": percentage(accepted),
        "generated_rate": percentage(generated),
        "rejected_rate": percentage(rejected),
        "avg_attempts": (
            round(sum(attempts) / len(attempts), 2)
            if attempts else None
        ),
        "exact_match_rate": (
            round(
                sum(exact_match_values) / len(exact_match_values) * 100,
                2,
            )
            if exact_match_values else None
        ),
        "mean_field_f1": (
            round(sum(field_f1_values) / len(field_f1_values), 4)
            if field_f1_values else None
        ),
        "duplicate_rate": calculate_duplicate_rate(results),
    }


def write_metrics_csv(metrics_by_arm, output_path):
    """Write one row per experiment arm."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(metrics_by_arm.values())
    if not rows:
        return 0

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def plot_metrics_comparison(metrics_by_arm, output_path):
    """Create a comparison chart for metrics available in both arms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arms = list(metrics_by_arm.keys())
    metric_names = [
        ("success_rate", "Success rate (%)"),
        ("accepted_rate", "Accepted rate (%)"),
        ("mean_field_f1", "Mean field F1"),
        ("duplicate_rate", "Duplicate rate (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.flatten()

    for axis, (metric, label) in zip(axes, metric_names):
        values = [metrics_by_arm[arm].get(metric) for arm in arms]
        valid = [
            (arm, value)
            for arm, value in zip(arms, values)
            if value is not None
        ]

        if valid:
            axis.bar(
                [item[0] for item in valid],
                [item[1] for item in valid],
            )
            axis.set_title(label)
            axis.set_ylabel(label)
        else:
            axis.set_title(label + " (not available)")

    fig.suptitle("Arm A vs Arm B - Experiment Comparison")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path
