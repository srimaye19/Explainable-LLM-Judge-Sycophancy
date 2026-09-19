import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

# Load API key from .env
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY was not found in .env")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Files
INPUT_FILE = "data/processed/user_prompts.csv"
OUTPUT_FILE = "data/processed/responses.csv"

# Read prompts
df = pd.read_csv(INPUT_FILE)

print("===================================")
print("GEMINI RESPONSE GENERATION")
print("===================================")
print(f"Number of prompts: {len(df)}")
print()

responses = []

for index, row in df.iterrows():

    print(f"Processing {index + 1}/{len(df)}...")

    prompt = row["user_message"]

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        ai_response = response.text

    except Exception as e:
        ai_response = f"ERROR: {str(e)}"

    responses.append(ai_response)

    # Small delay between requests
    time.sleep(2)


# Add responses to dataframe
df["model"] = "gemini-3.6-flash"
df["ai_response"] = responses

# Save result
df.to_csv(OUTPUT_FILE, index=False)

print()
print("===================================")
print("RESPONSE GENERATION COMPLETE")
print("===================================")
print(f"Saved to: {OUTPUT_FILE}")
print(f"Number of responses: {len(df)}")