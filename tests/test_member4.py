import json
from types import SimpleNamespace

from export import export_jsonl
from evaluation.metrics import (
    calculate_duplicate_rate,
    compute_metrics,
    write_metrics_csv,
)


def sample_example(text="Invoice INV-001 dated 01 January 2026"):
    return {
        "id": "example-1",
        "instruction": "Extract invoice details as JSON.",
        "source_document": text,
        "ground_truth": {"invoice_number": "INV-001"},
    }


def test_export_jsonl_writes_standard_format(tmp_path):
    results = [
        {
            "status": "accepted",
            "example": sample_example(),
        },
        {
            "status": "rejected",
            "example": sample_example("Rejected example"),
        },
        {
            "status": "generated",
            "example": sample_example("Generated example"),
        },
    ]

    output = tmp_path / "exports" / "dataset.jsonl"
    count = export_jsonl(results, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert count == 2
    assert len(lines) == 2

    record = json.loads(lines[0])
    assert set(record.keys()) == {"instruction", "input", "output"}
    assert record["output"] == {"invoice_number": "INV-001"}


def test_export_can_include_rejected_results(tmp_path):
    results = [
        {"status": "rejected", "example": sample_example()},
        {"status": "needs_revision", "example": sample_example("Another invoice")},
    ]

    count = export_jsonl(
        results,
        tmp_path / "all.jsonl",
        include_rejected=True,
    )

    assert count == 2


def test_metrics_for_generator_only_arm():
    results = [
        {
            "status": "generated",
            "attempts": 1,
            "example": sample_example("Invoice A"),
            "critique": None,
        },
        {
            "status": "generated",
            "attempts": 2,
            "example": sample_example("Invoice B"),
            "critique": None,
        },
    ]

    metrics = compute_metrics(results, "Arm A")

    assert metrics["total"] == 2
    assert metrics["generated"] == 2
    assert metrics["success_rate"] == 100.0
    assert metrics["generated_rate"] == 100.0
    assert metrics["accepted_rate"] == 0.0
    assert metrics["avg_attempts"] == 1.5

    # Arm A has no Critic, so these metrics must be unavailable, not zero.
    assert metrics["exact_match_rate"] is None
    assert metrics["mean_field_f1"] is None


def test_metrics_for_full_arm():
    results = [
        {
            "status": "accepted",
            "attempts": 2,
            "example": sample_example("Invoice A"),
            "critique": {"exact_match": True, "field_f1": 1.0},
        },
        {
            "status": "rejected",
            "attempts": 1,
            "example": sample_example("Invoice B"),
            "critique": {"exact_match": False, "field_f1": 0.5},
        },
    ]

    metrics = compute_metrics(results, "Arm B")

    assert metrics["accepted"] == 1
    assert metrics["rejected"] == 1
    assert metrics["accepted_rate"] == 50.0
    assert metrics["exact_match_rate"] == 50.0
    assert metrics["mean_field_f1"] == 0.75


def test_duplicate_rate_detects_repeated_documents():
    results = [
        {"example": sample_example("Invoice number INV 001 total 500")},
        {"example": sample_example("Invoice number INV 001 total 500")},
        {"example": sample_example("A completely different purchase order")},
    ]

    assert calculate_duplicate_rate(results) > 0


def test_metrics_csv_is_created(tmp_path):
    metrics = {
        "Arm A": compute_metrics(
            [{"status": "generated", "attempts": 1, "example": sample_example()}],
            "Arm A",
        ),
        "Arm B": compute_metrics(
            [{"status": "accepted", "attempts": 2, "example": sample_example(),
              "critique": {"exact_match": True, "field_f1": 1.0}}],
            "Arm B",
        ),
    }

    output = tmp_path / "comparison.csv"
    row_count = write_metrics_csv(metrics, output)

    assert row_count == 2
    assert output.exists()
    assert "success_rate" in output.read_text(encoding="utf-8")


def test_pipeline_generator_only_does_not_call_other_agents(monkeypatch):
    import workflow.pipeline as pipeline

    recipe = SimpleNamespace(category="invoice")
    example = {
        "id": "test-1",
        "category": "invoice",
        "difficulty": "easy",
        "style": "standard",
        "source_document": "Invoice INV-001 dated 01 January 2026",
        "instruction": "Extract invoice details.",
        "json_schema": {"invoice_number": "string"},
        "ground_truth": {"invoice_number": "INV-001"},
    }

    monkeypatch.setattr(pipeline, "reset_recent", lambda: None)
    monkeypatch.setattr(pipeline, "build_recipes", lambda count: [recipe] * count)
    monkeypatch.setattr(
        pipeline,
        "generate_example",
        lambda *args, **kwargs: example,
    )

    def unexpected_call(*args, **kwargs):
        raise AssertionError("Extractor or Critic should not run in generator_only mode")

    monkeypatch.setattr(pipeline, "extract", unexpected_call)
    monkeypatch.setattr(pipeline, "critique", unexpected_call)

    results = pipeline.run_pipeline(count=1, mode="generator_only")

    assert len(results) == 1
    assert results[0]["status"] == "generated"
    assert results[0]["example"]["id"] == "test-1"


def test_pipeline_full_mode_uses_extractor_and_critic(monkeypatch):
    import workflow.pipeline as pipeline

    recipe = SimpleNamespace(category="invoice")
    example = {
        "id": "test-2",
        "category": "invoice",
        "difficulty": "easy",
        "style": "standard",
        "source_document": "Invoice INV-002 dated 02 January 2026",
        "instruction": "Extract invoice details.",
        "json_schema": {"invoice_number": "string"},
        "ground_truth": {"invoice_number": "INV-002"},
    }

    calls = {"extract": 0, "critique": 0}

    monkeypatch.setattr(pipeline, "reset_recent", lambda: None)
    monkeypatch.setattr(pipeline, "build_recipes", lambda count: [recipe] * count)
    monkeypatch.setattr(
        pipeline,
        "generate_example",
        lambda *args, **kwargs: example,
    )
    monkeypatch.setattr(
        pipeline,
        "validate_example",
        lambda raw: (raw, []),
    )

    def fake_extract(value):
        calls["extract"] += 1
        return {"invoice_number": "INV-002"}

    def fake_critique(value, predicted):
        calls["critique"] += 1
        return SimpleNamespace(
            decision="accept",
            model_dump=lambda: {
                "decision": "accept",
                "exact_match": True,
                "field_f1": 1.0,
            },
        )

    monkeypatch.setattr(pipeline, "extract", fake_extract)
    monkeypatch.setattr(pipeline, "critique", fake_critique)

    results = pipeline.run_pipeline(count=1, mode="full")

    assert results[0]["status"] == "accepted"
    assert calls["extract"] == 1
    assert calls["critique"] == 1
