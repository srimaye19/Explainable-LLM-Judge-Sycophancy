import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(__file__))
)

from src.judge.schema import JudgeResult


test_result = {
    "truthfulness_score": 2,
    "agreement_bias_score": 9,
    "reasoning_quality": 3,
    "safety_score": 7,
    "is_sycophantic": True,
    "explanation": (
        "The AI agreed with the user's incorrect belief "
        "instead of correcting it."
    ),
    "key_issues": [
        "Incorrect factual agreement",
        "High agreement bias"
    ],
    "suggested_improved_response": (
        "The user's belief is incorrect. "
        "The correct answer should be provided."
    )
}


result = JudgeResult(**test_result)


print("===================================")
print("PYDANTIC VALIDATION TEST")
print("===================================")

print(result)

print("\nValidation successful!")