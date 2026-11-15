"""Pydantic models for the dataset JSON schema and Critic output (Member 2)."""
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Difficulty = Literal["easy", "medium", "hard"]
Decision = Literal["accept", "revise", "reject"]


class Recipe(BaseModel):
    domain: str = Field(min_length=1)
    difficulty: Difficulty
    style: str | None = None


class DatasetExample(BaseModel):
    """One synthetic extraction example produced by the Generator."""

    id: str = Field(min_length=1)
    recipe: Recipe
    instruction: str = Field(min_length=10)
    input_text: str = Field(min_length=10)
    required_fields: list[str] = Field(min_length=1)
    expected_output: dict[str, str | None]

    @model_validator(mode="after")
    def _fields_match_output(self):
        missing = [f for f in self.required_fields if f not in self.expected_output]
        extra = [k for k in self.expected_output if k not in self.required_fields]
        if missing:
            raise ValueError(f"expected_output missing required fields: {missing}")
        if extra:
            raise ValueError(f"expected_output has unexpected fields: {extra}")
        return self


class CriticScores(BaseModel):
    """Rubric scores (1-5) returned by the LLM critic."""

    clarity: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    unambiguity: int = Field(ge=1, le=5)
    difficulty_match: int = Field(ge=1, le=5)
    feedback: str = ""

    @property
    def mean(self) -> float:
        return (self.clarity + self.correctness + self.unambiguity + self.difficulty_match) / 4

    @property
    def minimum(self) -> int:
        return min(self.clarity, self.correctness, self.unambiguity, self.difficulty_match)


class CritiqueResult(BaseModel):
    decision: Decision
    scores: CriticScores | None = None
    mean_score: float | None = None
    exact_match: bool | None = None
    field_f1: float | None = None
    rule_violations: list[str] = Field(default_factory=list)
    feedback: str = ""
