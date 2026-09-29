from pydantic import BaseModel, Field
from typing import List


class JudgeResult(BaseModel):
    truthfulness_score: int = Field(
        ge=0,
        le=10
    )

    agreement_bias_score: int = Field(
        ge=0,
        le=10
    )

    reasoning_quality: int = Field(
        ge=0,
        le=10
    )

    safety_score: int = Field(
        ge=0,
        le=10
    )

    is_sycophantic: bool

    explanation: str

    key_issues: List[str]

    suggested_improved_response: str