
import json
from pathlib import Path

from llm import generate_text, parse_json


BASE_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = BASE_DIR / "prompts" / "extractor_prompt.txt"


def extract(example):
    """
    Extract invoice information without accessing ground truth.
    """

    invoice_text = example.get("source_document", "")
    instruction = example.get("instruction", "")
    json_schema = example.get("json_schema", {})

    if not invoice_text:
        raise ValueError("Invoice text is missing.")

    extractor_rules = PROMPT_FILE.read_text(encoding="utf-8")

    prompt = f"""
{extractor_rules}

Extraction instruction:
{instruction}

Required JSON schema:
{json.dumps(json_schema, indent=2)}

Invoice text:
{invoice_text}
"""

    response = generate_text(prompt, temperature=0.0)

    return parse_json(response)
