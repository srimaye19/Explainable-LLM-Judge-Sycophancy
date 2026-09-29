# Dataset Card — Explainable LLM Judge Sycophancy

## 1. Dataset Name

Explainable LLM Judge Sycophancy Controlled Dataset

## 2. Purpose

This dataset is designed to prototype an explainable LLM-as-a-Judge system for detecting sycophantic responses.

The dataset contains pairs of responses to user messages containing incorrect factual beliefs.

The two response types are:

1. Non-sycophantic / corrective responses
2. Sycophantic / agreement responses

The goal is to allow an LLM judge to distinguish between truthful corrective behavior and agreement with an incorrect user belief.

## 3. Dataset Size

Total examples: 200

Sycophantic examples: 100

Non-sycophantic examples: 100

The dataset is balanced between the two classes.

## 4. Dataset Structure

Each example contains:

- id
- user_message
- correct_answer
- wrong_belief
- ai_response
- truthfulness_score
- sycophancy_score
- is_sycophantic
- notes
- annotator

## 5. Data Construction

The dataset was constructed from factual questions and corresponding incorrect user beliefs.

For each user belief, two controlled response types were created:

### Non-sycophantic response

The response explicitly corrects the user's incorrect belief and provides the correct information.

### Sycophantic response

The response agrees with the user's incorrect belief rather than correcting it.

This controlled construction creates matched examples where the underlying user belief remains the same while the response behavior changes.

## 6. Labels

### is_sycophantic

`true` indicates that the response agrees with or supports the user's incorrect belief.

`false` indicates that the response corrects the incorrect belief.

### truthfulness_score

The score represents how accurately the response reflects the known correct answer.

### sycophancy_score

The score represents the degree to which the response demonstrates agreement with the incorrect user belief.

## 7. Dataset Splits

The dataset is divided into:

- Training: 140 examples
- Development: 30 examples
- Test: 30 examples

A fixed random seed was used when creating the splits.

## 8. Intended Use

The dataset is intended for:

- LLM-as-a-Judge research
- Sycophancy detection experiments
- Judge prompt development
- Explainable evaluation
- Prototype benchmarking

## 9. Limitations

This is a controlled synthetic dataset created for a research prototype.

The responses are not a representative sample of naturally occurring conversations.

The sycophantic responses are deliberately constructed to agree with incorrect beliefs.

Therefore, performance on this dataset should not be interpreted as evidence that a judge will perform equally well on naturally occurring user conversations.

The dataset is also relatively small compared with large-scale language-model evaluation benchmarks.

## 10. Important Methodological Note

The controlled dataset is intended to test whether an LLM judge can recognize an obvious difference between:

- correcting an incorrect belief, and
- agreeing with an incorrect belief.

Future versions should include naturally occurring examples, multiple domains, ambiguous cases, borderline cases, and independently annotated examples.

## 11. Ethical Considerations

The dataset is intended for research and evaluation of model behavior.

It should not be used as the sole basis for making decisions about the safety, reliability, or suitability of a deployed AI system.

## 12. Phase 1 Status

Phase 1 dataset preparation is complete.

Dataset size: 200

Class balance:

- Non-sycophantic: 100
- Sycophantic: 100

Train/dev/test:

- Train: 140
- Dev: 30
- Test: 30