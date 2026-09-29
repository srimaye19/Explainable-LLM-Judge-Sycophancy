import pandas as pd
import os

INPUT_FILE = "data/processed/controlled_dataset.csv"

FINAL_FILE = "data/processed/final_dataset.csv"

TRAIN_FILE = "data/splits/train.csv"
DEV_FILE = "data/splits/dev.csv"
TEST_FILE = "data/splits/test.csv"


# Load dataset
df = pd.read_csv(INPUT_FILE)

print("===================================")
print("CREATING FINAL PHASE 1 DATASET")
print("===================================")

print("Total examples:", len(df))


# Remove accidental duplicates
df = df.drop_duplicates(
    subset=["user_message", "ai_response"]
).reset_index(drop=True)


# Shuffle dataset
df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Save final dataset
os.makedirs("data/processed", exist_ok=True)

df.to_csv(
    FINAL_FILE,
    index=False
)


# Create balanced train/dev/test split
train_size = int(len(df) * 0.70)
dev_size = int(len(df) * 0.15)

train = df.iloc[:train_size]
dev = df.iloc[train_size:train_size + dev_size]
test = df.iloc[train_size + dev_size:]


# Create split directory
os.makedirs("data/splits", exist_ok=True)


train.to_csv(TRAIN_FILE, index=False)
dev.to_csv(DEV_FILE, index=False)
test.to_csv(TEST_FILE, index=False)


print()
print("Final dataset:", len(df))
print("Train:", len(train))
print("Dev:", len(dev))
print("Test:", len(test))

print()
print("Label distribution:")
print(df["is_sycophantic"].value_counts())

print()
print("Files created:")
print(FINAL_FILE)
print(TRAIN_FILE)
print(DEV_FILE)
print(TEST_FILE)

print()
print("PHASE 1 DATASET PREPARATION COMPLETE")