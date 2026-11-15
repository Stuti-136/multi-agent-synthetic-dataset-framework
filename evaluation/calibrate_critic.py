"""Calibration run: 6 good / 6 flawed examples through the Critic (Member 2).

Usage (repo root):
    python -m evaluation.calibrate_critic --rules-only   # no API calls
    python -m evaluation.calibrate_critic                # full critic via llm.py
Pass criterion: all 6 good accepted, 0 of 6 flawed accepted.
"""
import argparse
import json
from pathlib import Path

from agents.critic import critique
from evaluation.rules import validate_example

DATA = Path(__file__).resolve().parent.parent / "tests" / "flawed_examples.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules-only", action="store_true")
    args = ap.parse_args()
    data = json.loads(DATA.read_text(encoding="utf-8"))
    rows, good_ok, flawed_caught = [], 0, 0
    for label in ("good", "flawed"):
        for raw in data[label]:
            if args.rules_only:
                ex, errs = validate_example(raw)
                decision = "accept" if ex and not errs else "revise"
                note = "; ".join(errs)
            else:
                ex, errs = validate_example(raw)
                if ex is None:
                    decision, note = "revise", "; ".join(errs)
                else:
                    res = critique(ex, predicted=ex.expected_output if label == "good" else None)
                    decision, note = res.decision, res.feedback
            ok = (decision == "accept") == (label == "good")
            good_ok += ok and label == "good"
            flawed_caught += ok and label == "flawed"
            rows.append((raw["id"], label, decision, "PASS" if ok else "FAIL", note[:90]))
    for r in rows:
        print(f"{r[0]:10} {r[1]:7} {r[2]:7} {r[3]}  {r[4]}")
    print(f"\nGood accepted: {good_ok}/6   Flawed rejected/revised: {flawed_caught}/6")
    if args.rules_only:
        print("(rules-only mode: flawed-5 and flawed-6 need the LLM critic, so 4/6 is expected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
