import os
import pandas as pd

files = [
    "data/raw/questions.csv",
    "data/processed/user_prompts.csv",
    "data/processed/responses.csv",
    "data/processed/annotations.csv",
    "data/processed/final_dataset.csv",
    "data/splits/train.csv",
    "data/splits/dev.csv",
    "data/splits/test.csv",
    "literature/DATA_CARD.md",
]

print("===================================")
print("PHASE 1 FILE CHECK")
print("===================================")

all_found = True

for file in files:
    if os.path.exists(file):
        print(f"[OK] {file}")
    else:
        print(f"[MISSING] {file}")
        all_found = False

print()

if all_found:
    print("All Phase 1 files are present.")
else:
    print("Some Phase 1 files are missing.")