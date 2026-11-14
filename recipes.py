"""Recipe builder and coverage check for the invoice extraction domain (Member 1).

A recipe is one combination of difficulty, edge case, document style and locale.
The Generator is told exactly which recipe to follow, so diversity is built in.

Run a quick coverage report (no LLM needed):
    python -m recipes --count 50 --seed 42
"""
from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

CATEGORY = "invoice"

# --------------------------------------------------------------------------- #
# The 8 edge cases (name -> what the document contains / what ground truth does)
# --------------------------------------------------------------------------- #
EDGE_CASES: dict[str, dict[str, str]] = {
    "missing_field": {
        "document": "One requested field is not mentioned anywhere in the document.",
        "ground_truth": "That field is null.",
    },
    "multiple_amounts": {
        "document": "Subtotal, discount, tax and total all appear in the document.",
        "ground_truth": "amount is the final total due.",
    },
    "date_format_variation": {
        "document": "The date is written in a non-ISO format such as 15/03/2024, "
                    "March 15 2024 or 03-15-24 (follow the locale's date order).",
        "ground_truth": "The date is stored as YYYY-MM-DD, read according to the locale's date order.",
    },
    "multiple_vendors": {
        "document": "Two companies appear (the issuer and, for example, a courier or the customer).",
        "ground_truth": "vendor is the company that issued the invoice.",
    },
    "irrelevant_text": {
        "document": "Extra unrelated paragraphs (terms and conditions, marketing text) "
                    "surround the invoice details.",
        "ground_truth": "The irrelevant text is ignored.",
    },
    "credit_note": {
        "document": "The document is a credit note or refund, so the amount is negative.",
        "ground_truth": "amount is copied as stated, as a negative number.",
    },
    "currency_format": {
        "document": "The amount is written in a locale-specific format such as "
                    "\u20b915,000, Rs. 15000, 15,000.00 USD or 1.500,00 EUR.",
        "ground_truth": "amount is a plain number with no symbol or thousands separator "
                        "(dot as the decimal point).",
    },
    "malformed_id": {
        "document": "The invoice number has an unusual format such as INV/2026/0102 or INV 0102-A.",
        "ground_truth": "invoice_number is copied exactly as written.",
    },
}

STYLES: dict[str, str] = {
    "paragraph": "One or two plain prose paragraphs.",
    "email": "A short business email with a subject line, greeting and sign-off.",
    "table": "Details laid out as a simple text table (use | separators), one row per field or line item.",
    "bullet_list": "Details laid out as a bullet list.",
    "scanned_text": "Text that looks like OCR from a scanned invoice: uneven line breaks, "
                    "a few odd spacings, ALL-CAPS headings. Every value must stay readable and unambiguous.",
}

LOCALES: dict[str, dict[str, str]] = {
    "india": {
        "label": "India",
        "currency": "INR (\u20b9)",
        "date_order": "day/month/year, for example 15/03/2026",
        "numbers": "for example 12,500 or 1,25,000.50",
    },
    "usa": {
        "label": "USA",
        "currency": "USD ($)",
        "date_order": "month/day/year, for example 03/15/2026",
        "numbers": "for example 1,234.56",
    },
    "europe": {
        "label": "Europe",
        "currency": "EUR (\u20ac)",
        "date_order": "day.month.year, for example 15.03.2026",
        "numbers": "for example 1.234,56",
    },
    "uk": {
        "label": "UK",
        "currency": "GBP (\u00a3)",
        "date_order": "day/month/year, for example 15/03/2026",
        "numbers": "for example 1,234.56",
    },
}

DIFFICULTIES: dict[str, str] = {
    "easy": "Short document, 3 requested fields, every value stated plainly once.",
    "medium": "Longer document, 4 requested fields, some extra details that are not requested.",
    "hard": "Longer document, 4 to 5 requested fields, several distractor values; "
            "the edge case must really require careful reading.",
}

# 10-step pattern with 3 easy / 4 medium / 3 hard. Any 6 consecutive steps contain all
# three difficulties, so every edge-case group gets a mix (30% / 40% / 30% overall).
_DIFFICULTY_PATTERN = ["easy", "medium", "hard", "medium", "easy",
                       "hard", "medium", "easy", "medium", "hard"]
_DIFFICULTY_SHARE = {"easy": 0.3, "medium": 0.4, "hard": 0.3}


@dataclass
class Recipe:
    index: int
    difficulty: str
    edge_case: str
    style: str
    locale: str
    category: str = CATEGORY

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get(recipe: Any, key: str) -> Any:
    return recipe[key] if isinstance(recipe, dict) else getattr(recipe, key)


def as_recipe(recipe: Any) -> Recipe:
    """Accept a Recipe or a plain dict (for example from JSON) and return a Recipe."""
    if isinstance(recipe, Recipe):
        return recipe
    return Recipe(index=int(recipe.get("index", 1)), difficulty=recipe["difficulty"],
                  edge_case=recipe["edge_case"], style=recipe["style"], locale=recipe["locale"],
                  category=recipe.get("category", CATEGORY))


def describe_recipe(recipe: Any) -> str:
    """Text block describing the recipe, inserted into the Generator prompts."""
    r = as_recipe(recipe)
    ec, loc = EDGE_CASES[r.edge_case], LOCALES[r.locale]
    return "\n".join([
        f"- Difficulty: {r.difficulty} ({DIFFICULTIES[r.difficulty]})",
        f"- Edge case: {r.edge_case}. The document: {ec['document']} The ground truth: {ec['ground_truth']}",
        f"- Document style: {r.style} ({STYLES[r.style]})",
        f"- Locale: {loc['label']}. Currency {loc['currency']}; dates written as {loc['date_order']}; "
        f"numbers written {loc['numbers']}.",
    ])


# --------------------------------------------------------------------------- #
# Recipe builder (round-robin + seeded shuffle, not pure random)
# --------------------------------------------------------------------------- #
def difficulty_targets(n: int) -> dict[str, int]:
    """Target number of easy/medium/hard recipes for n samples (15/20/15 for 50)."""
    raw = {k: v * n for k, v in _DIFFICULTY_SHARE.items()}
    base = {k: int(v) for k, v in raw.items()}
    leftover = n - sum(base.values())
    for k in sorted(raw, key=lambda k: raw[k] - base[k], reverse=True)[:leftover]:
        base[k] += 1
    return base


def build_recipes(count: int, seed: int = 42, start_index: int = 1) -> list[Recipe]:
    """Build `count` recipes with balanced edge cases, difficulties, styles and locales.

    For 50 samples: each edge case 6 or 7 times, each style 10 times, each locale
    12 or 13 times, difficulty 15 / 20 / 15.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    rng = random.Random(seed)

    names = list(EDGE_CASES)
    rng.shuffle(names)  # so the edge cases that get a 7th sample vary with the seed
    base, extra = divmod(count, len(names))
    grouped_edges: list[str] = []
    for k, name in enumerate(names):
        grouped_edges += [name] * (base + (1 if k < extra else 0))

    styles = [list(STYLES)[j % len(STYLES)] for j in range(count)]
    locales = [list(LOCALES)[j % len(LOCALES)] for j in range(count)]
    rng.shuffle(styles)
    rng.shuffle(locales)

    recipes = [
        Recipe(index=0, difficulty=_DIFFICULTY_PATTERN[j % len(_DIFFICULTY_PATTERN)],
               edge_case=grouped_edges[j], style=styles[j], locale=locales[j])
        for j in range(count)
    ]
    rng.shuffle(recipes)  # mix the order so consecutive examples differ
    for pos, r in enumerate(recipes):
        r.index = start_index + pos
    return recipes


# --------------------------------------------------------------------------- #
# Coverage check
# --------------------------------------------------------------------------- #
def check_coverage(recipes: Iterable[Any]) -> tuple[bool, dict[str, Any]]:
    """Return (ok, report). `report["problems"]` lists everything that is under-represented."""
    recipes = list(recipes)
    n = len(recipes)
    problems: list[str] = []

    counts = {
        "edge_case": Counter(_get(r, "edge_case") for r in recipes),
        "style": Counter(_get(r, "style") for r in recipes),
        "locale": Counter(_get(r, "locale") for r in recipes),
        "difficulty": Counter(_get(r, "difficulty") for r in recipes),
    }
    allowed = {"edge_case": EDGE_CASES, "style": STYLES, "locale": LOCALES, "difficulty": DIFFICULTIES}
    minimum = {"edge_case": n // len(EDGE_CASES), "style": n // len(STYLES), "locale": n // len(LOCALES)}

    for dim, counter in counts.items():
        for value in counter:
            if value not in allowed[dim]:
                problems.append(f"unknown {dim} value: {value}")
    for dim, floor in minimum.items():
        for value in allowed[dim]:
            if counts[dim][value] < floor:
                problems.append(f"{dim} '{value}' appears {counts[dim][value]} times (minimum {floor})")

    targets = difficulty_targets(n)
    for value, target in targets.items():
        if abs(counts["difficulty"][value] - target) > 1:
            problems.append(f"difficulty '{value}' appears {counts['difficulty'][value]} times "
                            f"(target {target} +/- 1)")

    edge_by_diff = {e: {d: 0 for d in DIFFICULTIES} for e in EDGE_CASES}
    for r in recipes:
        e, d = _get(r, "edge_case"), _get(r, "difficulty")
        if e in edge_by_diff and d in edge_by_diff[e]:
            edge_by_diff[e][d] += 1
    if n >= 24:
        for e, row in edge_by_diff.items():
            for d, c in row.items():
                if c == 0:
                    problems.append(f"edge case '{e}' has no '{d}' example")

    report = {
        "n": n,
        "counts": {dim: dict(c) for dim, c in counts.items()},
        "edge_by_difficulty": edge_by_diff,
        "problems": problems,
    }
    return (not problems), report


def format_coverage_table(recipes: Iterable[Any]) -> str:
    """Printable coverage report (paste it into the project report as evidence of diversity)."""
    recipes = list(recipes)
    ok, rep = check_coverage(recipes)
    lines = [f"Coverage for {rep['n']} recipes: {'OK' if ok else 'PROBLEMS FOUND'}", "",
             f"{'edge case':<24}{'easy':>6}{'medium':>8}{'hard':>6}{'total':>7}"]
    for e, row in rep["edge_by_difficulty"].items():
        lines.append(f"{e:<24}{row['easy']:>6}{row['medium']:>8}{row['hard']:>6}{sum(row.values()):>7}")
    for dim in ("style", "locale", "difficulty"):
        lines += ["", f"{dim}: " + ", ".join(f"{k}={v}" for k, v in sorted(rep["counts"][dim].items()))]
    if rep["problems"]:
        lines += ["", "Problems:"] + [f"- {p}" for p in rep["problems"]]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Print the recipe coverage table.")
    ap.add_argument("--count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    print(format_coverage_table(build_recipes(args.count, args.seed)))
