
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import GEMINI_API_KEY, GEMINI_MODEL, RAW_DATA_DIR


# Connect to Gemini
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(timeout=300000),
)


def is_retryable_error(error):
    """Identify temporary errors that can be retried."""
    status_code = getattr(error, "status_code", None)
    class_name = type(error).__name__

    return (
        status_code in (408, 429, 500, 502, 503, 504)
        or class_name in ("APITimeoutError", "TimeoutError")
    )


@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def generate_text(prompt, temperature=0.0, system_instruction=None):
    """Generate text using Gemini Interactions API."""

    interaction = client.interactions.create(
        model=GEMINI_MODEL,
        input=prompt,
        generation_config={"temperature": temperature},
        system_instruction=system_instruction,
    )

    response_text = interaction.output_text or ""

    # Save raw Gemini response
    raw_dir = Path(RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
        + ".txt"
    )

    (raw_dir / filename).write_text(
        response_text,
        encoding="utf-8",
    )

    return response_text


def parse_json(response_text):
    """Parse JSON from Gemini's response."""

    cleaned = response_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_positions = [
       p for p in (cleaned.find("{"), cleaned.find("["))
       if p != -1
]

    if not start_positions:
        raise ValueError("No JSON object or array found.")

    start = min(start_positions)
    end = max(cleaned.rfind("}"), cleaned.rfind("]"))

    if end < start:
        raise ValueError("Incomplete JSON response.")

    return json.loads(cleaned[start:end + 1])


def count_tokens(text):
    """Count tokens using Gemini."""

    result = client.models.count_tokens(
        model=GEMINI_MODEL,
        contents=text,
    )

    return result.total_tokens or 0
