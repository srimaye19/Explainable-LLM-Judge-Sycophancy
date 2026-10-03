import os
import sys
import pandas as pd
import plotly.express as px

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import load_evaluation_history, compute_analytics_metrics, MODEL_NAME

def test_analytics_and_export():
    print("========================================")
    print("ANALYTICS & EXPORT TEST SUITE")
    print("========================================")

    # Test 1: Zero records safety
    zero_metrics = compute_analytics_metrics([])
    assert zero_metrics["total"] == 0
    assert zero_metrics["sycophancy_rate"] == 0.0
    print("[PASS] Test 1: Zero records handled safely")

    # Test 2: One record safety
    one_record = [{
        "timestamp": "2026-10-03 10:00:00",
        "model": "gemini-3.5-flash-lite",
        "is_sycophantic": True,
        "truthfulness_score": 3,
        "agreement_bias_score": 8,
        "reasoning_quality": 4,
        "safety_score": 9,
        "user_message": "test",
        "wrong_belief": "test",
        "correct_answer": "test",
        "ai_response": "test",
        "explanation": "test",
        "key_issues": ["issue1"],
        "suggested_improved_response": "test"
    }]
    one_metrics = compute_analytics_metrics(one_record)
    assert one_metrics["total"] == 1
    assert one_metrics["sycophantic_count"] == 1
    assert one_metrics["non_sycophantic_count"] == 0
    assert one_metrics["sycophancy_rate"] == 100.0
    assert one_metrics["avg_truthfulness"] == 3.0
    print("[PASS] Test 2: Single record handled safely")

    # Test 3: Load existing records
    records = load_evaluation_history()
    print(f"Loaded {len(records)} existing history records.")
    assert len(records) >= 6, f"Expected at least 6 records, got {len(records)}"

    metrics = compute_analytics_metrics(records)
    print("Calculated Analytics Metrics:")
    for k, v in metrics.items():
        print(f" - {k}: {v}")

    assert metrics["total"] == len(records)
    assert metrics["sycophantic_count"] + metrics["non_sycophantic_count"] == metrics["total"]
    expected_rate = round((metrics["sycophantic_count"] / metrics["total"]) * 100.0, 1)
    assert metrics["sycophancy_rate"] == expected_rate
    assert 0 <= metrics["avg_truthfulness"] <= 10
    assert 0 <= metrics["avg_agreement_bias"] <= 10
    assert 0 <= metrics["avg_reasoning_quality"] <= 10
    assert 0 <= metrics["avg_safety_score"] <= 10
    print("[PASS] Test 3: Analytics calculations verified against real history records")

    # Test 4: Plotly chart creation
    # Pie chart
    class_df = pd.DataFrame({
        "Classification": ["Sycophantic", "Non-Sycophantic"],
        "Count": [metrics["sycophantic_count"], metrics["non_sycophantic_count"]]
    })
    fig_class = px.pie(class_df, names="Classification", values="Count", hole=0.45)
    assert fig_class is not None

    # Bar chart
    avg_score_df = pd.DataFrame({
        "Dimension": ["Truthfulness", "Agreement Bias", "Reasoning Quality", "Safety"],
        "Average Score": [
            metrics["avg_truthfulness"],
            metrics["avg_agreement_bias"],
            metrics["avg_reasoning_quality"],
            metrics["avg_safety_score"]
        ]
    })
    fig_scores = px.bar(avg_score_df, x="Dimension", y="Average Score", range_y=[0, 10])
    assert fig_scores is not None

    # Line distribution chart
    chronological_records = list(reversed(records))
    dist_data = []
    for i, r in enumerate(chronological_records):
        eval_label = f"Eval {i+1}"
        dist_data.append({"Evaluation": eval_label, "Dimension": "Truthfulness", "Score": float(r.get("truthfulness_score", 0))})
        dist_data.append({"Evaluation": eval_label, "Dimension": "Agreement Bias", "Score": float(r.get("agreement_bias_score", 0))})
        dist_data.append({"Evaluation": eval_label, "Dimension": "Reasoning Quality", "Score": float(r.get("reasoning_quality", 0))})
        dist_data.append({"Evaluation": eval_label, "Dimension": "Safety", "Score": float(r.get("safety_score", 0))})
    dist_df = pd.DataFrame(dist_data)
    fig_dist = px.line(dist_df, x="Evaluation", y="Score", color="Dimension", markers=True)
    assert fig_dist is not None
    print("[PASS] Test 4: All Plotly visualizations generated successfully")

    # Test 5: CSV Export generation
    export_rows = []
    for r in records:
        issues = r.get("key_issues", [])
        issues_str = "; ".join(str(iss) for iss in issues) if isinstance(issues, list) else str(issues)
        export_rows.append({
            "timestamp": r.get("timestamp") or r.get("timestamp_iso", ""),
            "model": r.get("model", MODEL_NAME),
            "user_message": r.get("user_message", ""),
            "wrong_belief": r.get("wrong_belief", ""),
            "correct_answer": r.get("correct_answer", ""),
            "ai_response": r.get("ai_response", ""),
            "truthfulness_score": r.get("truthfulness_score", 0),
            "agreement_bias_score": r.get("agreement_bias_score", 0),
            "reasoning_quality": r.get("reasoning_quality", 0),
            "safety_score": r.get("safety_score", 0),
            "is_sycophantic": r.get("is_sycophantic", False),
            "explanation": r.get("explanation", ""),
            "key_issues": issues_str,
            "suggested_improved_response": r.get("suggested_improved_response", "")
        })
    df_export = pd.DataFrame(export_rows)
    csv_text = df_export.to_csv(index=False, encoding="utf-8")
    assert len(df_export) == len(records)
    assert "timestamp" in df_export.columns
    assert "is_sycophantic" in df_export.columns
    assert "truthfulness_score" in df_export.columns
    assert "suggested_improved_response" in df_export.columns
    assert len(csv_text) > 500
    print(f"[PASS] Test 5: CSV export generated successfully ({len(df_export)} rows, {len(csv_text)} bytes)")

    print("\nALL ANALYTICS & EXPORT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_analytics_and_export()
