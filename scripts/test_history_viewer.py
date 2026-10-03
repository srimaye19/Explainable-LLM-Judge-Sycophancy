import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.app import load_evaluation_history, MODEL_NAME

def test_history_viewer():
    history = load_evaluation_history()
    print("Total history records loaded:", len(history))
    assert len(history) == 6, f"Expected 6 records, got {len(history)}"

    # Check newest first
    timestamps = [h.get("timestamp") or h.get("timestamp_iso") for h in history]
    print("Timestamps (newest first):")
    for t in timestamps:
        print(" -", t)
    assert timestamps == sorted(timestamps, reverse=True), "History is not sorted newest first!"

    for idx, item in enumerate(history):
        print(f"\n--- Record {idx+1}: {item.get('_filename')} ---")
        ts = item.get("timestamp") or item.get("timestamp_iso")
        model = item.get("model")
        is_syco = item.get("is_sycophantic")
        t_score = item.get("truthfulness_score")
        a_score = item.get("agreement_bias_score")
        r_score = item.get("reasoning_quality")
        s_score = item.get("safety_score")

        u_msg = item.get("user_message")
        w_bel = item.get("wrong_belief")
        c_ans = item.get("correct_answer")
        a_resp = item.get("ai_response")
        expl = item.get("explanation")
        issues = item.get("key_issues")
        sugg = item.get("suggested_improved_response")

        print(f"Timestamp: {ts}")
        print(f"Model: {model}")
        print(f"Classification: {'Sycophantic' if is_syco else 'Non-Sycophantic'}")
        print(f"Scores: Truth={t_score}, Bias={a_score}, Reason={r_score}, Safety={s_score}")
        print(f"User Message: {u_msg[:50]}...")
        print(f"Wrong Belief: {w_bel[:50]}...")
        print(f"Correct Answer: {c_ans[:50]}...")
        print(f"AI Response: {a_resp[:50]}...")
        print(f"Explanation: {expl[:50]}...")
        print(f"Key Issues Count: {len(issues)}")
        print(f"Suggested Response: {sugg[:50]}...")

        # Assertions for required fields
        assert ts is not None, "Missing timestamp"
        assert model is not None, "Missing model"
        assert is_syco is not None, "Missing classification"
        assert t_score is not None, "Missing truthfulness_score"
        assert a_score is not None, "Missing agreement_bias_score"
        assert r_score is not None, "Missing reasoning_quality"
        assert s_score is not None, "Missing safety_score"
        assert u_msg, "Missing user_message"
        assert w_bel, "Missing wrong_belief"
        assert c_ans, "Missing correct_answer"
        assert a_resp, "Missing ai_response"
        assert expl, "Missing explanation"
        assert isinstance(issues, list), "Missing key_issues list"
        assert sugg, "Missing suggested_improved_response"

    print("\nSUCCESS: All 6 history records contain all required display fields and are sorted newest first!")

if __name__ == "__main__":
    test_history_viewer()
