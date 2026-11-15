
from agents.generator import generate_example, reset_recent, GenerationError
from agents.extractor import extract
from agents.critic import critique
from evaluation.rules import validate_example
from recipes import build_recipes
from config import MAX_REVISIONS


def convert_for_critic(example):
    """Convert Generator output into the format expected by the Critic."""

    return {
        "id": example["id"],
        "recipe": {
            "domain": example["category"],
            "difficulty": example["difficulty"],
            "style": example["style"],
        },
        "instruction": example["instruction"],
        "input_text": example["source_document"],
        "required_fields": list(example["json_schema"].keys()),
        "expected_output": {
            key: None if value is None else str(value)
            for key, value in example["ground_truth"].items()
        },
    }


def run_pipeline(category="invoice", count=3, mode="full"):
    """
    Run the synthetic invoice generation pipeline.

    Modes:
        generator_only: Generate examples without extraction or criticism.
        full: Generator -> Extractor -> Critic.
    """

    if mode not in ("generator_only", "full"):
        raise ValueError("Mode must be generator_only or full.")

    reset_recent()
    recipes = build_recipes(count)

    recipes = [r for r in recipes if r.category == category]

    if not recipes:
        raise ValueError(f"No recipes found for category: {category}")

    results = []

    for recipe in recipes:
        example = None
        predicted = None
        critic_result = None
        feedback = ""
        previous_example = None
        status = "rejected"
        errors = []

        for attempt in range(MAX_REVISIONS + 1):

            try:
                example = generate_example(
                    recipe,
                    feedback=feedback or None,
                    previous_example=previous_example,
                )
            except GenerationError as error:
                errors.append(str(error))
                break

            if mode == "generator_only":
                status = "generated"
                break

            critic_input = convert_for_critic(example)
            validated, validation_errors = validate_example(critic_input)

            if validated is None:
                feedback = "; ".join(validation_errors)
                errors = validation_errors
                previous_example = example
                continue

            try:
                predicted = extract(example)
            except Exception as error:
                predicted = None
                errors.append(f"Extractor error: {error}")

            critic_result = critique(validated, predicted)

            if critic_result.decision == "accept":
                status = "accepted"
                break

            if critic_result.decision == "reject":
                status = "rejected"
                break

            feedback = critic_result.feedback
            previous_example = example

            if attempt == MAX_REVISIONS:
                status = "needs_revision"

        results.append({
            "id": example["id"] if example else None,
            "status": status,
            "attempts": attempt + 1,
            "example": example,
            "predicted": predicted,
            "critique": (
                critic_result.model_dump()
                if critic_result is not None else None
            ),
            "errors": errors,
        })

    return results
