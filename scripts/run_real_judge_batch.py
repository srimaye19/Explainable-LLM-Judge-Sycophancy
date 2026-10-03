"""
scripts/run_real_judge_batch.py

Batch runner for the real Gemini LLM judge on Phase 1 test split.
Evaluates sycophancy using the prompt instructions and Pydantic schema.

Key Features:
- Sequential processing with a configurable LIMIT (default: 5).
- Resumes cleanly without duplicating already-processed IDs.
- Immediately saves each successful result to results/judge_results.csv.
- Safely handles Gemini 429 / quota errors without fabricating predictions.
- Robust string ID handling (e.g., '25_truthful', '25_sycophantic').
- Skips invalid JSON or schema errors safely.
"""

import os
import sys
import json
import time
import argparse
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
from google import genai

from src.judge.judge_prompt import (
    JUDGE_SYSTEM_PROMPT,
    build_judge_prompt,
)
from src.judge.schema import JudgeResult

# =============================================================
# CONFIGURATION
# =============================================================

# Maximum number of NEW examples to evaluate in this run
LIMIT = 5

MODEL_NAME = "gemini-3.5-flash-lite"
TEST_DATA_PATH = "data/splits/test.csv"
RESULTS_FILE_PATH = "results/judge_results.csv"

# Required columns in results/judge_results.csv
RESULT_COLUMNS = [
    "id",
    "expected_label",
    "judge_label",
    "truthfulness_score",
    "agreement_bias_score",
    "reasoning_quality",
    "safety_score",
    "explanation",
    "key_issues",
    "suggested_improved_response",
]


def load_existing_ids(filepath: str) -> set:
    """Loads set of string IDs already recorded in results file to avoid duplicates."""
    if not os.path.exists(filepath):
        return set()
    try:
        df = pd.read_csv(filepath, dtype={"id": str})
        if not df.empty and "id" in df.columns:
            return set(df["id"].dropna().astype(str).str.strip().tolist())
    except Exception:
        pass
    return set()


def append_result_to_csv(record: dict, filepath: str = RESULTS_FILE_PATH):
    """Appends a single verified judge result to the CSV immediately."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    df_row = pd.DataFrame([record])[RESULT_COLUMNS]

    # If file exists and is non-empty, append without repeating header
    file_exists = os.path.exists(filepath) and os.path.getsize(filepath) > 0
    if file_exists:
        df_row.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df_row.to_csv(filepath, mode="w", header=True, index=False)


def clean_json_response(raw_text: str) -> str:
    """Strips markdown code fences from Gemini output."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def is_quota_error(error: Exception) -> bool:
    """Detects whether an exception represents a 429 or quota limit exhaustion."""
    err_str = str(error).upper()
    err_code = getattr(error, "code", None)
    if err_code in (429, "429"):
        return True
    quota_terms = [
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA",
        "RATE LIMIT",
        "RESOURCE HAS BEEN EXHAUSTED",
    ]
    return any(term in err_str for term in quota_terms)


def main():
    parser = argparse.ArgumentParser(
        description="Run real Gemini LLM judge batch on test split."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help=f"Maximum new test examples to process (default: {LIMIT})",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=TEST_DATA_PATH,
        help=f"Path to Phase 1 test split (default: {TEST_DATA_PATH})",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=RESULTS_FILE_PATH,
        help=f"Path to output CSV (default: {RESULTS_FILE_PATH})",
    )

    args = parser.parse_args()

    print("===============================================================")
    print("             REAL LLM JUDGE BATCH EVALUATION                  ")
    print("===============================================================")
    print(f"Model       : {MODEL_NAME}")
    print(f"Batch Limit : {args.limit} new example(s)")
    print(f"Test Split  : {args.test_path}")
    print(f"Results File: {args.results_path}")
    print("---------------------------------------------------------------")

    # 1. Check Gemini API key
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[ERROR] GEMINI_API_KEY not found in .env or environment.")
        print("Please verify your API key is set before running.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # 2. Load test dataset (IDs as strings)
    if not os.path.exists(args.test_path):
        print(f"[ERROR] Test dataset not found at '{args.test_path}'.")
        sys.exit(1)

    test_df = pd.read_csv(args.test_path, dtype={"id": str})
    test_df["id"] = test_df["id"].astype(str).str.strip()

    print(f"[OK] Loaded {len(test_df)} examples from test split.")

    # 3. Load already processed IDs to avoid duplicate calls
    existing_ids = load_existing_ids(args.results_path)
    already_existing_count = 0
    success_count = 0
    skipped_count = 0

    print(f"[OK] Found {len(existing_ids)} existing record(s) in '{args.results_path}'.")
    print("---------------------------------------------------------------")

    # 4. Process examples sequentially up to limit
    for idx, row in test_df.iterrows():
        example_id = str(row["id"]).strip()

        # Skip if already evaluated in previous runs
        if example_id in existing_ids:
            already_existing_count += 1
            continue

        if success_count >= args.limit:
            print(f"\n[INFO] Reached batch limit of {args.limit} new example(s).")
            break

        progress_num = success_count + 1
        print(f"\n[{progress_num}/{args.limit}] Processing ID: {example_id}...")

        # Build prompt
        prompt = build_judge_prompt(
            user_message=row["user_message"],
            correct_answer=row["correct_answer"],
            wrong_belief=row["wrong_belief"],
            ai_response=row["ai_response"],
        )

        # Call Gemini API
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[
                    JUDGE_SYSTEM_PROMPT,
                    prompt,
                ],
            )
            raw_output = response.text

        except Exception as e:
            # Check for quota limit (429 / RESOURCE_EXHAUSTED)
            if is_quota_error(e):
                print("\n===============================================================")
                print("[STOP] GEMINI API 429 / QUOTA EXCEEDED")
                print("===============================================================")
                print(f"Error details: {e}")
                print()
                print("The batch script has stopped safely.")
                print("No fake predictions were created.")
                print(f"All {success_count} successfully completed record(s) are preserved in:")
                print(f"  {args.results_path}")
                print("You can re-run this script when quota resets to continue.")
                print("===============================================================")
                break

            print(f"  [ERROR] Gemini API request failed: {e}")
            print(f"  [SKIP] Skipping ID: {example_id}")
            skipped_count += 1
            continue

        # Parse and validate response
        cleaned_json = clean_json_response(raw_output)

        try:
            result_json = json.loads(cleaned_json)
        except json.JSONDecodeError as err:
            print(f"  [WARN] Gemini did not return valid JSON: {err}")
            print(f"  [SKIP] Skipping ID: {example_id} safely.")
            skipped_count += 1
            continue

        try:
            validated = JudgeResult(**result_json)
        except Exception as val_err:
            print(f"  [WARN] Pydantic validation failed: {val_err}")
            print(f"  [SKIP] Skipping ID: {example_id} safely.")
            skipped_count += 1
            continue

        # Format key issues cleanly
        if isinstance(validated.key_issues, list):
            key_issues_str = "; ".join(str(k).strip() for k in validated.key_issues if str(k).strip())
        else:
            key_issues_str = str(validated.key_issues)

        # Expected ground truth
        expected_bool = (
            str(row["is_sycophantic"]).strip().lower() in ("true", "1", "yes", "t")
        )

        record = {
            "id": example_id,
            "expected_label": expected_bool,
            "judge_label": bool(validated.is_sycophantic),
            "truthfulness_score": int(validated.truthfulness_score),
            "agreement_bias_score": int(validated.agreement_bias_score),
            "reasoning_quality": int(validated.reasoning_quality),
            "safety_score": int(validated.safety_score),
            "explanation": str(validated.explanation),
            "key_issues": key_issues_str,
            "suggested_improved_response": str(validated.suggested_improved_response),
        }

        # Immediately save to disk
        append_result_to_csv(record, args.results_path)
        existing_ids.add(example_id)
        success_count += 1

        print(
            f"  [OK] Validated: is_sycophantic={validated.is_sycophantic} "
            f"(truthfulness={validated.truthfulness_score}, agreement_bias={validated.agreement_bias_score})"
        )
        print(f"  [SAVED] Appended to '{args.results_path}'")

        # Brief pause between calls to respect rate limits
        time.sleep(1.0)

    # 5. Print final summary report
    print("\n===============================================================")
    print("                     BATCH RUN SUMMARY                         ")
    print("===============================================================")
    print(f"- Successfully processed : {success_count}")
    print(f"- Skipped (errors/invalid): {skipped_count}")
    print(f"- Already existing (prior): {already_existing_count}")
    print(f"- Output file path        : {args.results_path}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
