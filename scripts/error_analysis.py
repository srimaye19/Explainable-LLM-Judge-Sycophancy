"""
scripts/error_analysis.py

Error analysis script for Explainable LLM Judge Sycophancy project.
Analyzes incorrect predictions on Phase 1 test split without fabricating data.
"""

import os
import sys
import json
import argparse
import pandas as pd

TEST_PATH = "data/splits/test.csv"
RESULTS_PATH = "results/judge_results.csv"
ERROR_CSV_PATH = "results/error_analysis.csv"
SUMMARY_JSON_PATH = "results/error_analysis_summary.json"

REQUIRED_ERROR_COLUMNS = [
    "id",
    "user_message",
    "wrong_belief",
    "correct_answer",
    "ai_response",
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


def to_bool(val) -> bool:
    """Robust conversion to boolean."""
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "t")


def main():
    parser = argparse.ArgumentParser(
        description="Perform error analysis on LLM judge predictions."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=TEST_PATH,
        help=f"Path to Phase 1 test split (default: {TEST_PATH})",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=RESULTS_PATH,
        help=f"Path to judge results CSV (default: {RESULTS_PATH})",
    )
    parser.add_argument(
        "--error_out",
        type=str,
        default=ERROR_CSV_PATH,
        help=f"Path to output errors CSV (default: {ERROR_CSV_PATH})",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default=SUMMARY_JSON_PATH,
        help=f"Path to output summary JSON (default: {SUMMARY_JSON_PATH})",
    )

    args = parser.parse_args()

    # 1. Check if judge_results.csv exists and has records
    if not os.path.exists(args.results_path):
        print("No real judge results available. Generate real LLM judge results first.")
        sys.exit(0)

    try:
        judge_df = pd.read_csv(args.results_path, dtype={"id": str})
    except Exception:
        print("No real judge results available. Generate real LLM judge results first.")
        sys.exit(0)

    if judge_df.empty or len(judge_df) == 0:
        print("No real judge results available. Generate real LLM judge results first.")
        sys.exit(0)

    # 2. Load Phase 1 test dataset (IDs as strings)
    if not os.path.exists(args.test_path):
        print(f"[ERROR] Test dataset not found at '{args.test_path}'.")
        sys.exit(1)

    test_df = pd.read_csv(args.test_path, dtype={"id": str})
    test_df["id"] = test_df["id"].astype(str).str.strip()
    judge_df["id"] = judge_df["id"].astype(str).str.strip()

    # 3. Match records using string IDs
    context_columns = ["id", "user_message", "wrong_belief", "correct_answer", "ai_response"]
    available_context = [col for col in context_columns if col in test_df.columns]

    merged = pd.merge(
        test_df[available_context],
        judge_df,
        on="id",
        how="inner"
    )

    if len(merged) == 0:
        print("No real judge results available. Generate real LLM judge results first.")
        sys.exit(0)

    # 4. Compare expected_label vs judge_label
    merged["expected_label"] = merged["expected_label"].map(to_bool)
    merged["judge_label"] = merged["judge_label"].map(to_bool)

    # 5. Calculate metrics
    total_evaluated = len(merged)
    correct_mask = merged["expected_label"] == merged["judge_label"]
    correct_predictions = int(correct_mask.sum())
    incorrect_predictions = int((~correct_mask).sum())

    # False Positives: Ground truth is False (truthful), but judge predicted True (sycophantic)
    fp_mask = (~merged["expected_label"]) & (merged["judge_label"])
    false_positives = int(fp_mask.sum())

    # False Negatives: Ground truth is True (sycophantic), but judge predicted False (truthful)
    fn_mask = (merged["expected_label"]) & (~merged["judge_label"])
    false_negatives = int(fn_mask.sum())

    # 6. Extract incorrect examples and save to CSV
    incorrect_df = merged[~correct_mask].copy()

    # Ensure all required columns exist in output dataframe
    for col in REQUIRED_ERROR_COLUMNS:
        if col not in incorrect_df.columns:
            incorrect_df[col] = ""

    incorrect_export = incorrect_df[REQUIRED_ERROR_COLUMNS]

    os.makedirs(os.path.dirname(args.error_out) or ".", exist_ok=True)
    incorrect_export.to_csv(args.error_out, index=False)

    # 7. Save summary JSON
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    summary_data = {
        "total_evaluated": total_evaluated,
        "correct_predictions": correct_predictions,
        "incorrect_predictions": incorrect_predictions,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    # 8. Print clear summary
    print("===============================================================")
    print("                 ERROR ANALYSIS SUMMARY                        ")
    print("===============================================================")
    print(f"Total evaluated       : {total_evaluated}")
    print(f"Correct predictions   : {correct_predictions}")
    print(f"Incorrect predictions : {incorrect_predictions}")
    print(f"False Positives (FP)  : {false_positives}")
    print(f"False Negatives (FN)  : {false_negatives}")
    print("---------------------------------------------------------------")
    print(f"Saved incorrect cases to : {args.error_out}")
    print(f"Saved summary to         : {args.summary_out}")
    print("===============================================================")


if __name__ == "__main__":
    main()
