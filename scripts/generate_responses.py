import os
import time
import pandas as pd
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in .env")

client = genai.Client(api_key=API_KEY)

INPUT_FILE = "data/processed/user_prompts.csv"
OUTPUT_FILE = "data/processed/responses.csv"

MODEL = "gemini-3.6-flash"

# Number of retries for temporary errors
MAX_RETRIES = 5

# Wait between successful API requests
REQUEST_DELAY = 2


# --------------------------------------------------
# Load input
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("===================================")
print("GEMINI RESPONSE GENERATION")
print("===================================")
print(f"Total prompts: {len(df)}")
print()


# --------------------------------------------------
# Load existing responses if available
# --------------------------------------------------

if os.path.exists(OUTPUT_FILE):

    existing = pd.read_csv(OUTPUT_FILE)

    if len(existing) == len(df):
        print("Existing response file found.")
        print("Resuming failed/empty responses...")
    else:
        print("Existing file size does not match input.")
        existing = df.copy()

else:
    existing = df.copy()


# Make sure response column exists
if "ai_response" not in existing.columns:
    existing["ai_response"] = ""


# --------------------------------------------------
# Generate responses
# --------------------------------------------------

for index, row in df.iterrows():

    current_response = existing.at[index, "ai_response"]

    # Skip already successful responses
    if (
        pd.notna(current_response)
        and str(current_response).strip() != ""
        and not str(current_response).startswith("ERROR:")
    ):
        print(f"[{index + 1}/{len(df)}] Already completed - skipping")
        continue

    user_message = str(row["user_message"])

    print(f"[{index + 1}/{len(df)}] Generating response...")

    success = False

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=user_message
            )

            text = response.text.strip()

            existing.at[index, "ai_response"] = text

            # Save immediately
            existing.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print(
                f"    Success on attempt {attempt}"
            )

            success = True

            time.sleep(REQUEST_DELAY)

            break

        except Exception as e:

            error_text = str(e)

            print(
                f"    Attempt {attempt}/{MAX_RETRIES} failed:"
            )
            print(
                f"    {error_text[:200]}"
            )

            # Retry temporary server errors
            if "503" in error_text or "UNAVAILABLE" in error_text:

                wait_time = attempt * 10

                print(
                    f"    Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:

                # Non-temporary error
                existing.at[index, "ai_response"] = (
                    f"ERROR: {error_text}"
                )

                existing.to_csv(
                    OUTPUT_FILE,
                    index=False
                )

                break

    if not success:

        # Keep the row available for another run
        if (
            pd.isna(existing.at[index, "ai_response"])
            or str(existing.at[index, "ai_response"]).strip() == ""
        ):
            existing.at[index, "ai_response"] = (
                "ERROR: Gemini temporarily unavailable"
            )

        existing.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            "    Could not generate this response."
        )


print()
print("===================================")
print("GENERATION FINISHED")
print("===================================")

final_df = pd.read_csv(OUTPUT_FILE)

errors = (
    final_df["ai_response"]
    .astype(str)
    .str.startswith("ERROR")
    .sum()
)

successful = len(final_df) - errors

print(f"Total rows: {len(final_df)}")
print(f"Successful responses: {successful}")
print(f"Errors: {errors}")
print(f"Saved to: {OUTPUT_FILE}")