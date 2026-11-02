# GRADE & RoB2 RAG Chatbots

Retrieval-Augmented Generation chatbots for two medical methodology domains:

- **GRADE** — methodology for rating certainty of evidence and developing healthcare guidelines
  (GRADE Book pages + GRADE guideline articles).
- **RoB2** — Cochrane Risk of Bias 2.0 tool for randomised trials.

Each chatbot answers strictly from a local FAISS index built from its sources. The project
includes a Gradio web app and an automated + human evaluation pipeline.

Project requirements, the work plan and audit findings are in [`docs/`](docs/README.md).

## Setup

Requirements: Python 3.10+, Google Chrome (Selenium scraping of JS-rendered pages).

```bash
pip install -r requirements.txt
cp .env.example .env   # set OPENROUTER_API_KEY
```

All LLM calls go through [OpenRouter](https://openrouter.ai); model names use the
`provider/model-name` format.

## Running the app

```bash
python app.py
```

Opens at http://127.0.0.1:7860. Indices are loaded from `.faiss_cache/` (built on first run).

## Running the notebooks

Run from the repository root, in this order:

1. `grade_rag.ipynb` — builds/loads the GRADE index, defines `ask_grade()` / `get_grade_contexts()`
2. `rob2_rag.ipynb` — builds/loads the RoB2 index, defines `ask_rob2()` / `get_rob2_contexts()`
3. `generate_rob2_dataset.ipynb` — generates RoB2 Q&A pairs (run once)
4. `automated_evaluation_extended.ipynb` — RAGAS, ROUGE, BLEU, semantic similarity, G-Eval
5. `llm_judge.ipynb` — two-judge LLM evaluation with Cohen's Kappa
6. `prepare_human_eval.ipynb` — pre-filled CSVs for human annotation

`run_screening_eval.py` runs the GRADE chatbot on the screening questions.

## Repository structure

```
app.py                         Gradio web app
rag_core.py                    Shared RAG pipeline (scrape/parse, chunk, index, answer)
grade_rag.ipynb / grade_rag.py GRADE chatbot (notebook + importable script)
rob2_rag.ipynb / rob2_rag.py   RoB2 chatbot (notebook + importable script)
generate_rob2_dataset.ipynb    RoB2 Q&A generation
automated_evaluation_extended.ipynb
llm_judge.ipynb
prepare_human_eval.ipynb
run_screening_eval.py
data/
  grade_pdfs/, rob2_pdfs/      Source PDFs
  parsed_grade/, parsed_rob2/  Per-PDF parse cache (JSON)
.faiss_cache/grade, rob2/      FAISS indices
evaluation/
  datasets/                    Evaluation questions and reference answers
  results/                     Metric CSVs (grade, rob2, grade_initial, screening, llm_judge)
  human_eval/                  Annotation guide and annotation CSVs
docs/                          Requirements, plan, findings, technical report
legacy/                        Earlier agentic pipeline evaluated on ML papers (not maintained)
feedback.jsonl                 User feedback collected by the app
```
