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
print("LOCAL BATCH JUDGE")
print("===================================")


# Load Phase 1 test dataset
df = pd.read_csv("data/splits/test.csv")

results = []


for _, row in df.iterrows():

    # Simulated judge result.
    # This is intentionally local and does NOT call Gemini.
    expected = bool(row["is_sycophantic"])

    mock_result = {
        "truthfulness_score": 8 if not expected else 2,
        "agreement_bias_score": 2 if not expected else 9,
        "reasoning_quality": 8,
        "safety_score": 8,
        "is_sycophantic": expected,
        "explanation": (
            "Local test result used to verify the complete "
            "judge pipeline."
        ),
        "key_issues": [],
        "suggested_improved_response": (
            "Provide a truthful response that does not "
            "reinforce an incorrect belief."
        )
    }

    # Validate using Pydantic
    validated = JudgeResult(**mock_result)

    results.append({
        "id": str(row["id"]),
        "expected_label": expected,
        "judge_label": validated.is_sycophantic,
        "truthfulness_score": validated.truthfulness_score,
        "agreement_bias_score": validated.agreement_bias_score,
        "reasoning_quality": validated.reasoning_quality,
        "safety_score": validated.safety_score,
        "explanation": validated.explanation
    })


# Convert to DataFrame
results_df = pd.DataFrame(results)


# Calculate agreement
results_df["correct"] = (
    results_df["expected_label"]
    == results_df["judge_label"]
)


accuracy = results_df["correct"].mean()


print("\nExamples processed:", len(results_df))

print(
    "Correct predictions:",
    int(results_df["correct"].sum())
)

print(
    "Accuracy:",
    f"{accuracy:.2%}"
)


# Save results
os.makedirs("results", exist_ok=True)

results_df.to_csv(
    "results/local_batch_results.csv",
    index=False
)


with open(
    "results/local_batch_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "examples": len(results_df),
            "correct_predictions": int(
                results_df["correct"].sum()
            ),
            "accuracy": float(accuracy)
        },
        f,
        indent=4
    )


print("\nFiles created:")
print("results/local_batch_results.csv")
print("results/local_batch_summary.json")

print("\n===================================")
print("LOCAL BATCH TEST COMPLETE")
print("===================================")