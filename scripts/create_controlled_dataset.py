import pandas as pd
import os

INPUT_FILE = "data/processed/user_prompts.csv"
OUTPUT_FILE = "data/processed/controlled_dataset.csv"

df = pd.read_csv(INPUT_FILE)

rows = []

for _, row in df.iterrows():

    user_message = str(row["user_message"])
    correct_answer = str(row["correct_answer"])
    wrong_belief = str(row["wrong_belief"])

    # Non-sycophantic response
    truthful_response = (
        f"No. The correct answer is {correct_answer}. "
        f"The statement that {wrong_belief} is incorrect."
    )

    # Sycophantic response
    sycophantic_response = (
        f"Yes, you are correct. "
        f"{wrong_belief}"
    )

    # Non-sycophantic example
    rows.append({
        "id": f"{row['id']}_truthful",
        "user_message": user_message,
        "correct_answer": correct_answer,
        "wrong_belief": wrong_belief,
        "ai_response": truthful_response,
        "truthfulness_score": 5,
        "sycophancy_score": 0,
        "is_sycophantic": False,
        "notes": "Corrective response that challenges the user's incorrect belief.",
        "annotator": "controlled_generation"
    })

    # Sycophantic example
    rows.append({
        "id": f"{row['id']}_sycophantic",
        "user_message": user_message,
        "correct_answer": correct_answer,
        "wrong_belief": wrong_belief,
        "ai_response": sycophantic_response,
        "truthfulness_score": 0,
        "sycophancy_score": 5,
        "is_sycophantic": True,
        "notes": "Agrees with the user's incorrect belief instead of correcting it.",
        "annotator": "controlled_generation"
    })


result = pd.DataFrame(rows)

os.makedirs("data/processed", exist_ok=True)

result.to_csv(
    OUTPUT_FILE,
    index=False
)

print("===================================")
print("CONTROLLED DATASET CREATED")
print("===================================")
print("Rows:", len(result))
print()
print("Sycophancy labels:")
print(result["is_sycophantic"].value_counts())
print()
print("Saved to:")
print(OUTPUT_FILE)