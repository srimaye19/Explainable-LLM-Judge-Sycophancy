import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/processed/final_dataset.csv"
OUTPUT_DIR = "data/splits"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Could not find {INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("===================================")
print("DATASET SPLITTING")
print("===================================")
print(f"Total examples: {len(df)}")
print()

# Check label distribution
print("Overall label distribution:")
print(df["is_sycophantic"].value_counts())
print()

# First split: 70% train, 30% temporary
train, temp = train_test_split(
    df,
    test_size=0.30,
    random_state=42
)

# Second split: 15% dev, 15% test
dev, test = train_test_split(
    temp,
    test_size=0.50,
    random_state=42
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

train.to_csv(
    f"{OUTPUT_DIR}/train.csv",
    index=False
)

dev.to_csv(
    f"{OUTPUT_DIR}/dev.csv",
    index=False
)

test.to_csv(
    f"{OUTPUT_DIR}/test.csv",
    index=False
)

print("===================================")
print("SPLIT COMPLETE")
print("===================================")
print(f"Train: {len(train)} examples")
print(f"Dev:   {len(dev)} examples")
print(f"Test:  {len(test)} examples")
print()
print("Files saved to:")
print("data/splits/train.csv")
print("data/splits/dev.csv")
print("data/splits/test.csv")