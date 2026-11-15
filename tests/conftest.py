import os

# Allows offline tests to import config.py and llm.py without a real API key.
# Tests must mock LLM calls; this key must never be used for live experiments.
os.environ.setdefault("GEMINI_API_KEY", "offline-test-key")
