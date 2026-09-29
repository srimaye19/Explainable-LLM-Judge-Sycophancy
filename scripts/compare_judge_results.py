"""
scripts/compare_judge_results.py

Compares real LLM judge predictions (results/judge_results.csv)
against Phase 1 test labels (data/splits/test.csv).

Calculates:
- Accuracy
- Precision
- Recall
- F1 score
- Confusion Matrix

Outputs:
- Summary JSON: results/judge_evaluation.json
- Detailed CSV: results/judge_comparison.csv
"""

import os
import sys
import json
import argparse
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

DEFAULT_TEST_PATH = "data/splits/test.csv"
DEFAULT_JUDGE_RESULTS_PATH = "results/judge_results.csv"
EVALUATION_JSON_PATH = "results/judge_evaluation.json"
COMPARISON_CSV_PATH = "results/judge_comparison.csv"


def to_bool(val) -> bool:
    """Robust conversion to boolean for strings, bools, or numbers."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("true", "1", "yes", "t")


def to_bool_series(series: pd.Series) -> pd.Series:
    """Robust conversion of a pandas Series to boolean."""
    return series.map(to_bool)


def main():
    parser = argparse.ArgumentParser(
        description="Compare real LLM judge predictions with Phase 1 ground truth test labels."
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=DEFAULT_TEST_PATH,
        help=f"Path to Phase 1 test dataset (default: {DEFAULT_TEST_PATH})",
    )
    parser.add_argument(
        "--results_path",
        "-r",
        type=str,
        default=DEFAULT_JUDGE_RESULTS_PATH,
        help=f"Path to judge results file (default: {DEFAULT_JUDGE_RESULTS_PATH})",
    )
    parser.add_argument(
        "--summary_out",
        type=str,
        default=EVALUATION_JSON_PATH,
        help=f"Path to output evaluation summary JSON (default: {EVALUATION_JSON_PATH})",
    )
    parser.add_argument(
        "--comparison_out",
        type=str,
        default=COMPARISON_CSV_PATH,
        help=f"Path to output detailed comparison CSV (default: {COMPARISON_CSV_PATH})",
    )

    args = parser.parse_args()

    print("===============================================================")
    print("           REAL LLM JUDGE EVALUATION COMPARISON               ")
    print("===============================================================")

    # -------------------------------------------------------------
    # 1. Load data/splits/test.csv
    # -------------------------------------------------------------
    if not os.path.exists(args.test_path):
        print(f"[ERROR] Test dataset not found at '{args.test_path}'.")
        print("Please verify Phase 1 splits exist before running comparison.")
        sys.exit(1)

    # Keep IDs strictly as strings
    test_df = pd.read_csv(args.test_path, dtype={"id": str})
    test_df["id"] = test_df["id"].astype(str).str.strip()

    print(f"[OK] Loaded ground truth test split: '{args.test_path}' ({len(test_df)} examples)")

    # -------------------------------------------------------------
    # 2. Check if results/judge_results.csv exists and has data
    # -------------------------------------------------------------
    if not os.path.exists(args.results_path):
        print("\n---------------------------------------------------------------")
        print("[NOTICE] Real judge results file was not found:")
        print(f"         '{args.results_path}'")
        print()
        print("Real judge results must be generated first before comparison.")
        print("Once Gemini API quota is available, run the real LLM judge")
        print(f"pipeline to populate '{args.results_path}'.")
        print("---------------------------------------------------------------")
        print("===============================================================")
        sys.exit(0)

    # Load judge results, ensuring string IDs
    try:
        judge_df = pd.read_csv(args.results_path, dtype={"id": str})
    except Exception as e:
        print(f"[ERROR] Could not read '{args.results_path}': {e}")
        sys.exit(1)

    if judge_df.empty or len(judge_df) == 0:
        print("\n---------------------------------------------------------------")
        print(f"[NOTICE] '{args.results_path}' exists but contains 0 records")
        print("         (clean schema header template only).")
        print()
        print("Real judge results must be generated first before comparison.")
        print("Once Gemini API quota is available, run the real LLM judge")
        print(f"pipeline to populate '{args.results_path}'.")
        print("---------------------------------------------------------------")
        print("===============================================================")
        sys.exit(0)

    judge_df["id"] = judge_df["id"].astype(str).str.strip()
    print(f"[OK] Loaded judge predictions: '{args.results_path}' ({len(judge_df)} records)")

    # -------------------------------------------------------------
    # 3. Match records using the string ID
    # -------------------------------------------------------------
    # Merge test set and judge results on 'id'
    merged = pd.merge(
        test_df,
        judge_df,
        on="id",
        how="inner",
        suffixes=("_test", "_judge")
    )

    if len(merged) == 0:
        print("\n[ERROR] No matching IDs found between test set and judge results.")
        print(f"Test IDs sample : {test_df['id'].head().tolist()}")
        print(f"Judge IDs sample: {judge_df['id'].head().tolist()}")
        sys.exit(1)

    if len(merged) < len(test_df):
        print(
            f"[WARNING] Evaluated subset: {len(merged)} of {len(test_df)} test records matched."
        )
    else:
        print(f"[OK] All {len(merged)} test records matched successfully.")

    # -------------------------------------------------------------
    # 4. Compare expected_label with judge_label
    # -------------------------------------------------------------
    # Determine ground truth label: prefer 'expected_label' if present, else 'is_sycophantic'
    if "expected_label" in merged.columns:
        y_true = to_bool_series(merged["expected_label"])
    elif "expected_label_judge" in merged.columns:
        y_true = to_bool_series(merged["expected_label_judge"])
    elif "is_sycophantic" in merged.columns:
        y_true = to_bool_series(merged["is_sycophantic"])
    elif "is_sycophantic_test" in merged.columns:
        y_true = to_bool_series(merged["is_sycophantic_test"])
    else:
        raise ValueError("Could not find ground truth sycophancy column in merged dataset.")

    if "judge_label" in merged.columns:
        y_pred = to_bool_series(merged["judge_label"])
    else:
        raise ValueError(f"'{args.results_path}' must contain a 'judge_label' column.")

    merged["expected_label"] = y_true
    merged["judge_label"] = y_pred
    merged["correct"] = (y_true == y_pred)

    # -------------------------------------------------------------
    # 5. Calculate Metrics
    # -------------------------------------------------------------
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()

    # Print Report
    print("\n---------------------------------------------------------------")
    print("                    EVALUATION METRICS                         ")
    print("---------------------------------------------------------------")
    print(f"Matched records: {len(merged)}")
    print(f"Ground truth   : {int(y_true.sum())} Sycophantic (True), {len(y_true) - int(y_true.sum())} Non-Sycophantic (False)")
    print("---------------------------------------------------------------")
    print(f"Accuracy : {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall   : {recall:.2%}")
    print(f"F1 Score : {f1:.2%}")
    print("---------------------------------------------------------------")
    print("Confusion Matrix:")
    print("                         Predicted FALSE    Predicted TRUE")
    print(f"  Actual FALSE (Truthful)   {tn:<16}   {fp:<16}")
    print(f"  Actual TRUE (Sycophantic) {fn:<16}   {tp:<16}")
    print()
    print(f"  - True Negatives  (TN): {tn}")
    print(f"  - False Positives (FP): {fp}")
    print(f"  - False Negatives (FN): {fn}")
    print(f"  - True Positives  (TP): {tp}")
    print("---------------------------------------------------------------")

    # -------------------------------------------------------------
    # 6. Save summary to results/judge_evaluation.json
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    summary_data = {
        "dataset": args.test_path,
        "results_file": args.results_path,
        "total_test_records": len(test_df),
        "evaluated_records": len(merged),
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4)

    print(f"[SAVED] Evaluation summary -> {args.summary_out}")

    # -------------------------------------------------------------
    # 7. Save detailed comparison to results/judge_comparison.csv
    # -------------------------------------------------------------
    os.makedirs(os.path.dirname(args.comparison_out) or ".", exist_ok=True)

    # Order columns cleanly for comparison and inspection
    priority_cols = [
        "id",
        "expected_label",
        "judge_label",
        "correct",
        "truthfulness_score",
        "agreement_bias_score",
        "reasoning_quality",
        "safety_score",
        "explanation",
        "key_issues",
        "suggested_improved_response",
        "user_message",
        "correct_answer",
        "wrong_belief",
        "ai_response",
    ]
    # Add any remaining columns that exist in merged
    final_cols = [col for col in priority_cols if col in merged.columns]
    for col in merged.columns:
        if col not in final_cols and not col.endswith(("_test", "_judge")):
            final_cols.append(col)

    merged[final_cols].to_csv(args.comparison_out, index=False)
    print(f"[SAVED] Detailed comparison -> {args.comparison_out}")
    print("===============================================================\n")


if __name__ == "__main__":
    main()
