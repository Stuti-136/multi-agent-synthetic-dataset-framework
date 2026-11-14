
from unittest.mock import patch

from workflow import pipeline


def fake_generate_example(recipe, feedback=None, previous_example=None):
    return {
        "id": "test-001",
        "category": "invoice",
        "difficulty": "easy",
        "edge_case": "multiple_vendors",
        "style": "paragraph",
        "locale": "india",
        "source_document": (
            "Invoice No: INV001\n"
            "Vendor: ABC Traders\n"
            "Date: 14-11-2026\n"
            "Total: INR 1500"
        ),
        "instruction": "Extract invoice number, vendor, date and total.",
        "json_schema": {
            "invoice_number": "string",
            "vendor": "string",
            "date": "string",
            "total": "string"
        },
        "ground_truth": {
            "invoice_number": "INV001",
            "vendor": "ABC Traders",
            "date": "14-11-2026",
            "total": "INR 1500"
        }
    }


def fake_extract(example):
    return {
        "invoice_number": "INV001",
        "vendor": "ABC Traders",
        "date": "14-11-2026",
        "total": "INR 1500"
    }


def fake_critique(example, predicted=None):
    class FakeResult:
        decision = "accept"

        def model_dump(self):
            return {"decision": "accept"}

    assert predicted is not None
    return FakeResult()


@patch("workflow.pipeline.critique", side_effect=fake_critique)
@patch("workflow.pipeline.extract", side_effect=fake_extract)
@patch("workflow.pipeline.generate_example", side_effect=fake_generate_example)
def test_full_pipeline_accepts_example(mock_generate, mock_extract, mock_critique):
    results = pipeline.run_pipeline(
        category="invoice",
        count=1,
        mode="full"
    )

    assert len(results) == 1
    assert results[0]["status"] == "accepted"
    assert results[0]["example"]["id"] == "test-001"
    assert results[0]["predicted"]["invoice_number"] == "INV001"

    mock_generate.assert_called_once()
    mock_extract.assert_called_once()
    mock_critique.assert_called_once()

