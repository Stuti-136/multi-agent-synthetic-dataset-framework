
# Member 3 – Pipeline, LLM and Extractor

## Responsibilities
- LLM integration using Google Gemini.
- Invoice information extraction.
- Multi-agent pipeline orchestration.
- Main execution file and configuration.

## Files
- `llm.py` – Gemini API integration and JSON parsing.
- `config.py` – Model and pipeline configuration.
- `agents/extractor.py` – Extracts invoice fields from invoice text.
- `workflow/pipeline.py` – Connects Generator, Extractor and Critic.
- `main.py` – Runs the pipeline.
- `requirements.txt` – Python dependencies.
- `.gitignore` – Excludes environment and secret files.

## Pipeline Workflow
1. Generator creates a synthetic invoice.
2. Extractor extracts invoice information.
3. Critic evaluates the generated example and extraction.
4. Pipeline accepts, revises or rejects the example.
5. Accepted examples can be exported as JSONL.

## Execution
Activate the virtual environment and run:

```bash
python main.py --count 3 --category invoice --mode generator_only
```

For the full pipeline:

```bash
python main.py --count 3 --category invoice --mode full
```

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root and add your own Gemini API key:

```text
GEMINI_API_KEY=your_api_key
```

Do not share `.env` or API keys.

## Note
The pipeline has passed Python syntax checks. Full live Gemini integration testing is still required.
