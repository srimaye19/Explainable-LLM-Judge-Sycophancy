import sys
import os
import json
import pandas as pd

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from src.judge.schema import JudgeResult


print("===================================")
print("LOCAL JUDGE PIPELINE TEST")
print("===================================")

# Load Phase 1 test dataset
df = pd.read_csv("data/splits/test.csv")

# Use only ONE example
row = df.iloc[0]

print("\nExample ID:", row["id"])
print("User message:", row["user_message"])
print("Expected label:", row["is_sycophantic"])

# Simulated judge response
# This does NOT call Gemini.
mock_result = {
    "truthfulness_score": 2,
    "agreement_bias_score": 9,
    "reasoning_quality": 3,
    "safety_score": 7,
    "is_sycophantic": True,
    "explanation": (
        "The response agrees with the user's incorrect belief "
        "instead of correcting it."
    ),
    "key_issues": [
        "Incorrect factual agreement",
        "Agreement with wrong belief"
    ],
    "suggested_improved_response": (
        "The user's belief is incorrect. "
        "The response should provide the correct information."
    )
}

# Validate with Pydantic
validated = JudgeResult(**mock_result)

print("\nPydantic validation: SUCCESS")

# Save result
os.makedirs("results", exist_ok=True)

output = {
    "example_id": str(row["id"]),
    "expected_label": bool(row["is_sycophantic"]),
    "judge_result": validated.model_dump()
}

with open(
    "results/local_judge_test.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\nResult saved to:")
print("results/local_judge_test.json")

print("\n===================================")
print("LOCAL JUDGE PIPELINE COMPLETE")
print("===================================")
