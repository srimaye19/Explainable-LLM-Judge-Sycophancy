import sys
import os
import json
import pandas as pd

# Add project root to Python path
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
from google import genai

from src.judge.judge_prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt
)

from src.judge.schema import JudgeResult


# ==========================================
# LOAD API KEY
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY was not found in .env"
    )


# ==========================================
# GEMINI CLIENT
# ==========================================

client = genai.Client(
    api_key=api_key
)


# ==========================================
# LOAD TEST DATA
# ==========================================

df = pd.read_csv(
    "data/splits/test.csv"
)

# IMPORTANT:
# Start with only ONE example.
row = df.iloc[0]


print("===================================")
print("REAL GEMINI LLM JUDGE")
print("===================================")

print("\nExample ID:", row["id"])


# ==========================================
# BUILD PROMPT
# ==========================================

judge_prompt = build_judge_prompt(
    user_message=row["user_message"],
    correct_answer=row["correct_answer"],
    wrong_belief=row["wrong_belief"],
    ai_response=row["ai_response"]
)


# ==========================================
# CALL GEMINI
# ==========================================

try:

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            JUDGE_SYSTEM_PROMPT,
            judge_prompt
        ]
    )

    raw_output = response.text

    print("\nGemini response:")
    print(raw_output)


except Exception as e:

    print("\nGemini API error:")
    print(e)

    print(
        "\nThe script stopped without creating "
        "a fake judge result."
    )

    sys.exit(1)


# ==========================================
# CLEAN JSON
# ==========================================

clean_output = raw_output.strip()

if clean_output.startswith("```json"):
    clean_output = clean_output[7:]

if clean_output.startswith("```"):
    clean_output = clean_output[3:]

if clean_output.endswith("```"):
    clean_output = clean_output[:-3]

clean_output = clean_output.strip()


# ==========================================
# PARSE JSON
# ==========================================

try:

    result_json = json.loads(
        clean_output
    )

except json.JSONDecodeError as e:

    print("\nGemini did not return valid JSON.")
    print("JSON error:", e)

    sys.exit(1)


# ==========================================
# PYDANTIC VALIDATION
# ==========================================

try:

    validated = JudgeResult(
        **result_json
    )

except Exception as e:

    print("\nPydantic validation failed:")
    print(e)

    sys.exit(1)


print("\n===================================")
print("JUDGE VALIDATION SUCCESSFUL")
print("===================================")

print("\nSycophantic:")
print(validated.is_sycophantic)

print("\nTruthfulness score:")
print(validated.truthfulness_score)

print("\nAgreement bias score:")
print(validated.agreement_bias_score)

print("\nReasoning quality:")
print(validated.reasoning_quality)

print("\nSafety score:")
print(validated.safety_score)

print("\nExplanation:")
print(validated.explanation)


# ==========================================
# SAVE RESULT
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

output = {
    "id": str(row["id"]),
    "expected_label": bool(
        row["is_sycophantic"]
    ),
    "judge_result": validated.model_dump()
}


with open(
    "results/real_judge_result.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=4,
        ensure_ascii=False
    )


print("\nSaved to:")
print("results/real_judge_result.json")

print("\n===================================")
print("REAL JUDGE TEST COMPLETE")
print("===================================")