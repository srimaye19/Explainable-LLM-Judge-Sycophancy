Explainable LLM Judge for Sycophancy Detection

📌 Project Overview

Explainable LLM Judge for Sycophancy Detection is an AI-based framework designed to detect and explain sycophantic behavior in Large Language Model (LLM) responses.

Sycophancy occurs when an AI assistant agrees with or validates a user's incorrect belief instead of providing a truthful and factual response. This can lead to misinformation, especially in areas such as healthcare, education, science, and other knowledge-based applications.

This project uses an LLM-as-a-Judge approach to evaluate an AI-generated response by comparing it with the user's statement and a factual reference answer.

🎯 What the Project Does

The system evaluates an AI response and determines whether it is:

- Sycophantic – the response incorrectly agrees with or validates the user's misconception.
- Non-Sycophantic – the response provides a factual and corrective answer.

Along with the classification, the system provides an explanation of the judgment and evaluates the response across multiple dimensions such as:

- Truthfulness
- Agreement Bias
- Reasoning Quality
- Safety

The project also provides an interactive web interface where users can enter a statement, misconception, AI response, and factual reference answer to receive an explainable evaluation.

💡 Main Idea

The overall concept of the project is:

User Misconception + AI Response + Ground Truth → LLM Judge → Sycophancy Detection + Explanation

The goal is to make LLM evaluation more transparent by not only identifying whether an AI response is sycophantic, but also explaining why the response was classified that way.

🛠️ Technologies Used

- Python
- Google Gemini
- Pydantic
- Streamlit
- Pandas
- Plotly

🎯 Project Goal

The primary goal of this project is to develop an explainable and structured method for detecting sycophancy in LLM responses, helping researchers and developers understand how AI systems respond to incorrect user beliefs and evaluate the factual reliability of those responses.