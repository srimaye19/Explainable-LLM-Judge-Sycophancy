import sys
import os

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import json
import pandas as pd

from dotenv import load_dotenv
from google import genai

from src.judge.judge_prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt
)

from src.judge.schema import JudgeResult


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in .env")


# Create Gemini client
client = genai.Client(api_key=api_key)


# Load ONE test example
df = pd.read_csv(
    "data/splits/test.csv"
)

row = df.iloc[0]


# Build the judge prompt
user_prompt = build_judge_prompt(
    user_message=row["user_message"],
    correct_answer=row["correct_answer"],
    wrong_belief=row["wrong_belief"],
    ai_response=row["ai_response"]
)


print("===================================")
print("RUNNING LLM JUDGE")
print("===================================")


# Call Gemini
response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=[
        JUDGE_SYSTEM_PROMPT,
        user_prompt
    ]
)


raw_output = response.text

print("\nRAW GEMINI RESPONSE:\n")
print(raw_output)


# Remove markdown fences if Gemini adds them
clean_output = raw_output.strip()

if clean_output.startswith("```json"):
    clean_output = clean_output[7:]

if clean_output.startswith("```"):
    clean_output = clean_output[3:]

if clean_output.endswith("```"):
    clean_output = clean_output[:-3]

clean_output = clean_output.strip()


# Convert JSON
result_json = json.loads(clean_output)


# Validate using Pydantic
result = JudgeResult(**result_json)


print("\n===================================")
print("VALIDATION SUCCESSFUL")
print("===================================")

print("\nJudge result:")
print(result)


# Save result
os.makedirs("results", exist_ok=True)

with open(
    "results/judge_test_result.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        result.model_dump(),
        f,
        indent=4,
        ensure_ascii=False
    )


print("\nSaved to:")
print("results/judge_test_result.json")