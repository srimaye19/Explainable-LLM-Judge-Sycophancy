"""
scripts/evaluate_judge.py

Evaluation framework for the Explainable LLM Judge Sycophancy project.
Evaluates binary classification performance (is_sycophantic) against Phase 1 test labels.

Supported modes:
1. Real predictions: Evaluate predictions from a file via --predictions <file.csv>
2. Local pipeline verification: Fallback mode that verifies the evaluation pipeline
   without claiming actual LLM judge performance.
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

# Required columns in the Phase 1 test dataset
EXPECTED_COLUMNS = [
    "id",
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

DEFAULT_TEST_PATH = "data/splits/test.csv"
RESULTS_DIR = "results/evaluation"


def load_test_dataset(test_path: str = DEFAULT_TEST_PATH) -> pd.DataFrame:
    """
    Loads Phase 1 test dataset and validates required columns.
    Ensures 'id' is strictly preserved as string (e.g. '25_truthful', '25_sycophantic').
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test dataset not found at '{test_path}'. "
            "Please ensure data/splits/test.csv exists."
        )

    # Force 'id' to string so IDs like '25_truthful' or numeric IDs are never converted to integers
    df = pd.read_csv(test_path, dtype={"id": str})
    df["id"] = df["id"].astype(str)

    # Verify all expected columns are present
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Test dataset is missing required columns: {missing_cols}"
        )

    # Ensure is_sycophantic is boolean
    df["is_sycophantic"] = df["is_sycophantic"].astype(str).str.lower().isin(["true", "1"])

    return df


def parse_boolean_series(series: pd.Series) -> pd.Series:
    """Safely converts boolean, string, or numeric representations into bool."""
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "t"])


def load_predictions(predictions_path: str, test_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """
    Loads predictions from CSV or JSON and joins them with test_df on 'id'.
    Returns (merged_df, prediction_column_name).
    """
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found at '{predictions_path}'")

    if predictions_path.endswith(".json"):
        pred_df = pd.read_json(predictions_path, dtype={"id": str})
    else:
        pred_df = pd.read_csv(predictions_path, dtype={"id": str})

    # Rename 'example_id' to 'id' if present
    if "example_id" in pred_df.columns and "id" not in pred_df.columns:
        pred_df = pred_df.rename(columns={"example_id": "id"})

    if "id" not in pred_df.columns:
        raise ValueError(
            f"Predictions file '{predictions_path}' must contain an 'id' column to join with ground truth."
        )

    pred_df["id"] = pred_df["id"].astype(str)

    # Identify prediction column
    candidate_cols = [
        "judge_prediction",
        "judge_label",
        "is_sycophantic_pred",
        "predicted_sycophantic",
        "is_sycophantic",
    ]
    pred_col = None
    for col in candidate_cols:
        if col in pred_df.columns:
            pred_col = col
            break

    if not pred_col:
        raise ValueError(
            f"Could not find a valid prediction column in '{predictions_path}'. "
            f"Expected one of: {candidate_cols}"
        )

    # Merge on string 'id'
    merged = test_df.merge(
        pred_df[["id", pred_col]],
        on="id",
        how="inner",
        suffixes=("", "_pred_file")
    )

    if len(merged) < len(test_df):
        print(
            f"[WARNING] Predictions only cover {len(merged)} / {len(test_df)} test examples."
        )

    merged["judge_prediction"] = parse_boolean_series(merged[pred_col])
    return merged, pred_col


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Computes standard binary classification metrics and confusion matrix."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
        "raw_matrix": cm.tolist(),
    }


def print_evaluation_report(
    metrics: dict,
    total_count: int,
    true_sycophantic: int,
    true_truthful: int,
    is_mock: bool,
    source_name: str,
):
    """Prints a clear, formatted evaluation report to stdout."""
    print("\n===============================================================")
    print("                 LLM JUDGE EVALUATION REPORT                   ")
    print("===============================================================")

    if is_mock:
        print("[!] NOTICE: RUNNING IN PIPELINE VERIFICATION MODE (LOCAL MOCK)")
        print("    Predictions are from local test data, NOT actual LLM output.")
        print("    These metrics verify the evaluation code is working properly.")
        print("    Do NOT report these as real Gemini model performance.")
        print("---------------------------------------------------------------")
    else:
        print(f"Predictions source: {source_name}")
        print("---------------------------------------------------------------")

    print(f"Total test examples evaluated: {total_count}")
    print(f"Ground truth distribution     : {true_sycophantic} Sycophantic (True), {true_truthful} Non-Sycophantic (False)")
    print("---------------------------------------------------------------")
    print(f"Accuracy : {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall   : {metrics['recall']:.2%}")
    print(f"F1 Score : {metrics['f1_score']:.2%}")
    print("---------------------------------------------------------------")

    cm = metrics["confusion_matrix"]
    print("Confusion Matrix:")
    print("                         Predicted FALSE    Predicted TRUE")
    print(f"  Actual FALSE (Truthful)   {cm['true_negatives']:<16}   {cm['false_positives']:<16}")
    print(f"  Actual TRUE (Sycophantic) {cm['false_negatives']:<16}   {cm['true_positives']:<16}")
    print()
    print("  - True Negatives  (TN):", cm["true_negatives"], "(Truthful correctly classified as Truthful)")
    print("  - False Positives (FP):", cm["false_positives"], "(Truthful incorrectly classified as Sycophantic)")
    print("  - False Negatives (FN):", cm["false_negatives"], "(Sycophantic incorrectly classified as Truthful)")
    print("  - True Positives  (TP):", cm["true_positives"], "(Sycophantic correctly classified as Sycophantic)")
    print("===============================================================\n")


def save_results(
    evaluated_df: pd.DataFrame,
    metrics: dict,
    is_mock: bool,
    output_dir: str = RESULTS_DIR,
):
    """Saves evaluation summary JSON and detailed per-sample CSV."""
    os.makedirs(output_dir, exist_ok=True)

    summary_file = os.path.join(output_dir, "evaluation_summary.json")
    detailed_file = os.path.join(output_dir, "evaluation_details.csv")

    summary_payload = {
        "evaluation_mode": "mock_pipeline_verification" if is_mock else "real_llm_judge",
        "is_mock": is_mock,
        "disclaimer": (
            "These results are for local pipeline verification only and DO NOT represent actual LLM performance."
            if is_mock
            else "Evaluated on Phase 1 test set with LLM judge predictions."
        ),
        "test_dataset": DEFAULT_TEST_PATH,
        "total_evaluated": len(evaluated_df),
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        },
        "confusion_matrix": metrics["confusion_matrix"],
    }

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=4)

    # Save detailed CSV for error analysis
    export_cols = [
        col
        for col in [
            "id",
            "user_message",
            "correct_answer",
            "wrong_belief",
            "ai_response",
            "is_sycophantic",
            "judge_prediction",
            "correct",
        ]
        if col in evaluated_df.columns
    ]
    evaluated_df[export_cols].to_csv(detailed_file, index=False)

    print(f"Saved evaluation summary : {summary_file}")
    print(f"Saved detailed results   : {detailed_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM Judge Sycophancy Detection on Phase 1 Test Split"
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=DEFAULT_TEST_PATH,
        help=f"Path to Phase 1 test dataset (default: {DEFAULT_TEST_PATH})",
    )
    parser.add_argument(
        "--predictions",
        "-p",
        type=str,
        default=None,
        help="Path to predictions file (CSV/JSON). If omitted, runs in local pipeline verification mode.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Directory to save evaluation results (default: {RESULTS_DIR})",
    )

    args = parser.parse_args()

    # Step 1: Load and validate test set
    test_df = load_test_dataset(args.test_path)
    print(f"[OK] Loaded test dataset '{args.test_path}' with {len(test_df)} examples.")

    # Step 2: Acquire predictions
    is_mock = False
    source_name = args.predictions

    if args.predictions:
        print(f"[OK] Loading predictions from '{args.predictions}'...")
        evaluated_df, pred_col = load_predictions(args.predictions, test_df)
    else:
        # Fallback to local pipeline verification mode
        is_mock = True
        local_batch_file = "results/local_batch_results.csv"

        if os.path.exists(local_batch_file):
            print(f"[INFO] Using local pipeline batch predictions from '{local_batch_file}' for verification.")
            evaluated_df, _ = load_predictions(local_batch_file, test_df)
            source_name = local_batch_file
        else:
            print("[INFO] No external predictions found. Simulating local baseline for pipeline verification.")
            evaluated_df = test_df.copy()
            # For pipeline testing only: mock matches ground truth
            evaluated_df["judge_prediction"] = evaluated_df["is_sycophantic"]
            source_name = "local_pipeline_test"

    # Step 3: Compute evaluation metrics
    evaluated_df["correct"] = (
        evaluated_df["is_sycophantic"] == evaluated_df["judge_prediction"]
    )
    metrics = compute_metrics(
        y_true=evaluated_df["is_sycophantic"],
        y_pred=evaluated_df["judge_prediction"],
    )

    # Step 4: Display report
    true_sycophantic = int(evaluated_df["is_sycophantic"].sum())
    true_truthful = len(evaluated_df) - true_sycophantic

    print_evaluation_report(
        metrics=metrics,
        total_count=len(evaluated_df),
        true_sycophantic=true_sycophantic,
        true_truthful=true_truthful,
        is_mock=is_mock,
        source_name=source_name or "local_verification",
    )

    # Step 5: Save results to disk
    save_results(
        evaluated_df=evaluated_df,
        metrics=metrics,
        is_mock=is_mock,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()