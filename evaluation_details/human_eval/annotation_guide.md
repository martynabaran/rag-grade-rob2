# Human Annotation Guide

## Task
You will evaluate chatbot answers to questions about medical methodology (GRADE or RoB2).
For each row in the CSV, score the **Chatbot Answer** on four criteria.

## Files to annotate
- `human_eval_grade_sample.csv` — questions about the GRADE methodology
- `human_eval_rob2_sample.csv` — questions about the Risk of Bias 2.0 tool

## Scoring Criteria (1–5 scale)

### 1. Accuracy
Does the answer correctly reflect the source material?

| Score | Meaning |
|-------|---------|
| 1 | Completely incorrect or contradicts the reference |
| 2 | Mostly incorrect; a few correct elements |
| 3 | Partially correct; mixes accurate and inaccurate information |
| 4 | Mostly correct; minor errors or imprecisions |
| 5 | Fully correct; all claims are accurate |

### 2. Completeness
Does the answer cover all key points present in the reference answer?

| Score | Meaning |
|-------|---------|
| 1 | Major gaps — misses most key points |
| 2 | Covers only 1–2 key points, misses the rest |
| 3 | Covers roughly half the key points |
| 4 | Covers most key points; only minor omissions |
| 5 | Covers all key points from the reference |

### 3. Clarity
Is the answer well-structured and easy to understand?

| Score | Meaning |
|-------|---------|
| 1 | Confusing, disorganized, very hard to follow |
| 2 | Hard to follow; major structural issues |
| 3 | Reasonably clear but could be better organized |
| 4 | Clear and well-structured; minor issues |
| 5 | Excellent clarity — concise, logical, easy to follow |

### 4. Faithfulness
Is every claim in the answer supported by the retrieved context (not invented)?

| Score | Meaning |
|-------|---------|
| 1 | Most claims are not supported by any context (hallucination) |
| 2 | Several unsupported claims |
| 3 | Mixed — some supported, some not |
| 4 | Almost all claims are supported; one minor unsupported detail |
| 5 | Fully grounded — every claim traces to the context |

## How to fill in the CSV

1. Open the CSV in Excel or Google Sheets
2. Read the **Question**, **Reference Answer**, and **Chatbot Answer** for each row
3. Fill in integer scores (1–5) in the four score columns
4. Add optional free-text comments in the **notes** column
5. Leave the `chatbot_answer` and other pre-filled columns unchanged

## Important Notes
- Judge the **Chatbot Answer** against the **Reference Answer** — the reference is the gold standard
- The **Retrieved Context** column (if present) shows what the chatbot was given; use it to assess Faithfulness
- Do not penalize for style differences if the content is correct
- If you are unsure about a score, use the **notes** column to explain
- Aim for 15–30 minutes per 10 questions
