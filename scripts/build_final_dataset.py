import pandas as pd
import os

INPUT_FILE = "data/processed/annotations.csv"
OUTPUT_FILE = "data/processed/final_dataset.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

# Add source information
df["source"] = "Gemini-generated response + human annotation"

# Clean boolean labels
df["is_sycophantic"] = (
    df["is_sycophantic"]
    .astype(str)
    .str.strip()
    .str.upper()
)

# Check required columns
required_columns = [
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

missing = [
    column for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )

# Check for missing annotations
annotation_columns = [
    "truthfulness_score",
    "sycophancy_score",
    "is_sycophantic",
    "notes",
    "annotator",
]

for column in annotation_columns:
    missing_count = df[column].isna().sum()

    if missing_count > 0:
        print(
            f"WARNING: {column} has "
            f"{missing_count} missing values."
        )

# Select final columns
final_columns = [
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
    "source",
]

final_df = df[final_columns]

# Save
final_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("===================================")
print("FINAL DATASET CREATED")
print("===================================")
print(f"File: {OUTPUT_FILE}")
print(f"Examples: {len(final_df)}")
print()

print("Sycophantic labels:")
print(
    final_df["is_sycophantic"].value_counts()
)