import pandas as pd
import os

INPUT_FILE = "data/processed/responses.csv"
OUTPUT_FILE = "data/processed/annotations.csv"

# Check input file
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}. "
        "Run generate_responses.py first."
    )

# Read responses
df = pd.read_csv(INPUT_FILE)

# Create annotation columns
df["truthfulness_score"] = ""
df["sycophancy_score"] = ""
df["is_sycophantic"] = ""
df["notes"] = ""
df["annotator"] = ""

# Keep only the columns needed for annotation
annotation_df = df[
    [
        "id",
        "model",
        "user_message",
        "correct_answer",
        "wrong_belief",
        "ai_response",
        "truthfulness_score",
        "sycophancy_score",
        "is_sycophantic",
        "notes",
        "annotator",
    ]
]

# Save
annotation_df.to_csv(OUTPUT_FILE, index=False)

print("===================================")
print("ANNOTATION FILE CREATED")
print("===================================")
print(f"File: {OUTPUT_FILE}")
print(f"Number of examples: {len(annotation_df)}")