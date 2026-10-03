# Explainable LLM Judge for Sycophancy Detection

An explainable LLM-as-a-Judge framework and interactive web application designed to evaluate, classify, and explain sycophantic agreement bias in Large Language Model (LLM) responses.

---

## 📌 Problem Overview

Large Language Models (LLMs) often exhibit **sycophancy**—a behavioral bias where the assistant agrees with or validates a user's misconception or incorrect premise, even when doing so contradicts factual truth. In critical domains like healthcare, science, and education, sycophantic behavior undermines user trust and spreads misinformation.

This project implements an **Explainable LLM Judge** that:
1. Compares an AI response against the user's misconception and a factual reference answer.
2. Formally classifies whether the response is **sycophantic** or **non-sycophantic**.
3. Calculates dimensional scores (Truthfulness, Agreement Bias, Reasoning Quality, and Safety).
4. Delivers natural-language explanations, identifies key issues, and suggests an improved, factual response.

---

## 🚀 Key Features

* **LLM-as-a-Judge Architecture**: Powered by Google's `gemini-3.5-flash-lite` model for automated, high-precision evaluation.
* **Strict Pydantic Validation**: Guarantees structured, type-safe JSON schema enforcement on all evaluation results.
* **Multi-Dimensional Scoring (0–10 Scale)**:
  * **Truthfulness**: Factual correctness against ground truth.
  * **Agreement Bias**: Degree of inappropriate validation of incorrect beliefs.
  * **Reasoning Quality**: Depth and logical coherence of the explanation.
  * **Safety Score**: Avoidance of misleading or hazardous validation.
* **Interactive Streamlit Web App**:
  * One-click demo examples (Astronomy, Physics, Medical).
  * Arbitrary text inputs for custom evaluation.
  * Real-time classification badges and metric scorecards.
* **Evaluation History Viewer**:
  * Persistent JSON logging in `results/app_history/`.
  * Deduplication and newest-first chronological sorting.
  * Classification filtering (`All`, `Sycophantic Only`, `Non-Sycophantic Only`).
* **Analytics Dashboard**:
  * Real-time metrics: Sycophancy rate, total runs, and score averages.
  * Interactive Plotly visualizations (Classification Donut, Score Comparison Bar, and Trend Line charts).
* **CSV Data Export**: One-click download of all evaluation history for reporting and offline analysis.

---

## 🏗️ System Workflow

```
[User Statement & Misconception] + [AI Response] + [Ground Truth]
                             │
                             ▼
              [Prompt Engineering & System Rules]
                             │
                             ▼
              [Gemini 3.5 Flash Lite Judge]
                             │
                             ▼
              [JSON Cleaner & Pydantic Validation]
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
   [Structured Scores & Class]       [Explainable Insights]
   • Sycophantic: True/False         • Rationale Explanation
   • Truthfulness: 0-10              • Key Issues Identified
   • Agreement Bias: 0-10            • Suggested Improved Response
   • Reasoning Quality: 0-10
   • Safety Score: 0-10
            │                                 │
            └────────────────┬────────────────┘
                             ▼
            [Streamlit UI + Plotly Analytics + History + CSV Export]
```

---

## 🛠️ Technology Stack

* **Language**: Python 3.10+
* **LLM Engine**: Google Gemini API (`google-genai` SDK, `gemini-3.5-flash-lite`)
* **Validation**: Pydantic v2
* **Frontend**: Streamlit
* **Visualizations**: Plotly Express & Plotly Graph Objects
* **Data Processing**: Pandas, NumPy
* **Environment**: python-dotenv

---

## 📂 Project Structure

```
Explainable-LLM-Judge-Sycophancy/
├── app/
│   ├── app.py                     # Main interactive Streamlit application
│   └── app_phase3_backup.py       # Verified backup of the application
├── data/
│   ├── processed/
│   │   └── final_dataset.csv      # 200 curated benchmark examples
│   └── splits/
│       ├── train.csv              # 140 training examples
│       ├── dev.csv                # 30 development examples
│       └── test.csv               # 30 test examples
├── docs/
│   └── PHASE3_COMPLETION.md       # Comprehensive Phase 3 completion report
├── results/
│   ├── app_history/               # Saved live application evaluation records
│   ├── judge_results.csv          # Phase 2 test set evaluation results
│   ├── judge_evaluation.json      # Phase 2 benchmark metrics (100% accuracy)
│   ├── judge_comparison.csv       # Predictions vs ground-truth comparison
│   └── error_analysis_summary.json# Error analysis report
├── scripts/
│   ├── check_phase2.py            # Phase 1 & 2 integrity status check
│   ├── test_phase3.py             # Phase 3 core logic & persistence test suite
│   ├── test_analytics.py          # Analytics calculations & chart tests
│   ├── test_history_viewer.py     # History loader & display test
│   ├── test_live_judge.py         # Live Gemini API evaluation test
│   └── test_e2e_sycophancy.py     # E2E sycophantic vs non-sycophantic test
├── src/
│   └── judge/
│       ├── judge_prompt.py        # System prompt and prompt builder
│       └── schema.py              # Pydantic JudgeResult model
├── .env.example                   # Environment variable template
├── .gitignore                     # Git ignore rules (secrets & history protected)
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/srimaye19/Explainable-LLM-Judge-Sycophancy.git
cd Explainable-LLM-Judge-Sycophancy
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and insert your Google Gemini API key:
```bash
copy .env.example .env
```
Inside `.env`:
```ini
GEMINI_API_KEY=your_gemini_api_key_here
```

---

## 💻 Running the Application

Launch the Streamlit web application:

```bash
.\.venv\Scripts\python.exe -m streamlit run app\app.py
```

Open your browser at `http://localhost:8501`.

---

## 🧪 Running Phase 3 Tests

Run the automated test suites using the project virtual environment:

```bash
# 1. Phase 3 Core Tests (Schema, prompt, history, metrics)
.\.venv\Scripts\python.exe scripts\test_phase3.py

# 2. Analytics & Visualizations Test
.\.venv\Scripts\python.exe scripts\test_analytics.py

# 3. History Viewer Test
.\.venv\Scripts\python.exe scripts\test_history_viewer.py

# 4. Live Sycophancy Discrimination Test
.\.venv\Scripts\python.exe scripts\test_e2e_sycophancy.py

# 5. Phase 1 & 2 Integrity Verification
.\.venv\Scripts\python.exe scripts\check_phase2.py
```

---

## 📊 Phase 2 Benchmark Results

The LLM Judge evaluated against the 30-sample ground-truth test split (`data/splits/test.csv`) achieved:
* **Accuracy**: 100.0%
* **Precision**: 100.0%
* **Recall**: 100.0%
* **F1 Score**: 100.0%
* **Confusion Matrix**: TN = 14, FP = 0, FN = 0, TP = 16

---

## 🔒 Security & Privacy

* Sensitive files (`.env`, virtual environments, and local evaluation sessions in `results/app_history/`) are strictly excluded in `.gitignore`.
* API keys are masked in exception outputs and are never exposed in CSV exports or UI components.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](file:///c:/Users/HP/Desktop/Explainable-LLM-Judge-Sycophancy/LICENSE) for details.
