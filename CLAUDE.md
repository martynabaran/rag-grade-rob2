# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot system for two medical methodology domains:
- **GRADE** — methodology for developing evidence-based healthcare guidelines
- **RoB2** — Cochrane Risk of Bias 2.0 tool for assessing bias in randomised trials

Both chatbots scrape their source websites, build local FAISS vector indices, and answer questions strictly from retrieved content. The system includes comprehensive automated and human evaluation pipelines.

## Environment Setup

Copy `.env.example` to `.env` and fill in:
```
OPENROUTER_API_KEY=...
MODEL_NAME=google/gemini-2.5-flash        # chatbot model
EVAL_MODEL_NAME=google/gemini-2.5-flash   # evaluation model
SOURCE_KEY=GRADE Book                     # for legacy notebook
```

Chrome must be installed (Selenium uses it for JS-rendered pages). `webdriver-manager` handles the ChromeDriver automatically.

## Running the Notebooks

**Recommended execution order:**
1. `grade_rag.ipynb` — build GRADE index, defines `ask_grade()` / `get_grade_contexts()`
2. `rob2_rag.ipynb` — build RoB2 index, defines `ask_rob2()` / `get_rob2_contexts()`
3. `generate_rob2_dataset.ipynb` — generate RoB2 Q&A pairs (run once; output cached in CSV)
4. `automated_evaluation_extended.ipynb` — full metric suite on both domains
5. `llm_judge.ipynb` — two-judge LLM evaluation with Cohen's Kappa
6. `prepare_human_eval.ipynb` — generate pre-filled CSVs for human annotation

**Legacy notebook** (`agentic_rag_chatbot.ipynb`) requires Docker + GROBID only for the PDF/figure path; the GRADE Book path works without Docker.

## Architecture

### Shared Core (`rag_core.py`)

Three-step pipeline with no agent framework:
```
scrape_and_chunk(urls) or load_pdfs_and_chunk(pdf_paths) → build_index(docs) → rag_answer(query, index, llm)
```

Key functions:
- `fetch_visible_text(url)` — headless Chrome + BeautifulSoup, handles JS-rendered pages
- `load_or_build_index(urls, cache_path, parsed_pdf_cache_dir=...)` — builds FAISS index and saves to disk; loads from cache on subsequent runs (FAISS cache stored in `.faiss_cache/grade/` and `.faiss_cache/rob2/`)
- `load_pdfs_and_chunk(pdf_paths, parsed_cache_dir=...)` — parses PDFs to `Document`s and stores per-PDF JSON cache files; reparses only when a PDF is new or modified
- `rag_answer(query, index, llm, k=5)` — returns `(answer, contexts)` tuple; `contexts` is the list of retrieved chunk strings needed for RAGAS evaluation
- `rag_answer_with_sources(query, index, llm, k=5)` — returns `(answer, contexts, sources)` tuple; `sources` is a deduplicated ordered list of source URLs/paths from retrieved chunk metadata. Used by `app.py` to show source links in the UI. Do not remove or merge with `rag_answer` — evaluation notebooks depend on the 2-tuple signature.
- `build_llm()` — constructs `ChatOpenAI` pointed at OpenRouter from env vars

### Chatbot Notebooks (`grade_rag.ipynb`, `rob2_rag.ipynb`)

Each notebook defines a domain-specific system prompt and source list (web URLs and/or PDF paths), then exposes two functions used by all evaluation notebooks:
- `ask_<domain>(question)` → answer string
- `get_<domain>_contexts(question, k)` → list of retrieved chunk strings

The evaluation notebooks import these via `%run` or direct import — **run the chatbot notebook before the evaluation notebooks** in the same kernel session.

### Evaluation Stack

| Notebook | What it measures |
|---|---|
| `automated_evaluation_extended.ipynb` | RAGAS (Context Recall, Faithfulness, Factual Correctness, Answer Relevancy), ROUGE-1/2/L, BLEU, cosine semantic similarity, NLP complexity, G-Eval |
| `llm_judge.ipynb` | Two independent LLM judges scoring Accuracy/Completeness/Faithfulness (1–5), Cohen's Kappa inter-judge agreement |
| `prepare_human_eval.ipynb` | Stratified sample CSVs for human annotation (see `evaluation_details/human_eval/annotation_guide.md`) |

### Gradio App (`app.py`)

Run locally with `python app.py` — opens at http://127.0.0.1:7860. No external hosting required.

Features added (2026-05-30):
- **Per-user session isolation** — each browser tab gets its own `gr.State` copy (chat history, last Q&A). A unique UUID is generated per connection via `gr.State(lambda: str(uuid.uuid4()))`.
- **Source links in replies** — every assistant message ends with a `📚 Sources:` block listing clickable links back to the original pages/PDFs retrieved during RAG.
- **Thumbs up / down** — Gradio 5 built-in like event (`chatbot.like()`). Hover over any assistant reply to see the thumbs. Rating is saved immediately.
- **Optional comment** — a text box below the chat; its content is included when a thumb is clicked.
- **Feedback log** — all ratings written as newline-delimited JSON to `feedback.jsonl` in the project root. Schema: `timestamp`, `session_id`, `domain`, `question`, `answer`, `sources`, `rating` (`"up"`/`"down"`), `comment`.

Analysing feedback:
```python
import pandas as pd
df = pd.read_json("feedback.jsonl", lines=True)
df.groupby(["domain", "rating"]).size()          # overall satisfaction
df[df["rating"] == "down"][["question", "comment"]]  # failing questions
```

### Data Layout

```
evaluation_details/
  evaluation_gradebook/   # GRADE results (Q&A pairs, per-model metrics, RAGAS details)
  evaluation_rob2/        # RoB2 results (same structure; full_dataset_rob2.csv generated by notebook)
  llm_judge/              # Per-judge CSVs + summary
  human_eval/             # Pre-filled annotation CSVs + annotation_guide.md
feedback.jsonl            # Live user feedback from the Gradio app (created on first rating)
```

Evaluation CSVs follow the naming pattern `{metric_type}_{chat_model}_{eval_model}.csv`.

## Key Design Decisions

- **No LangGraph** — the original agentic architecture (4-node state graph) was replaced with plain function calls in `rag_core.py`. Do not re-introduce LangGraph.
- **No LangGraph-era PDF figure pipeline** — the old GROBID/figure-extraction path is not part of the new architecture. Keep the new PDF support focused on text parsing to `Document`s plus cacheable JSON outputs.
- **FAISS cache** — scraping is slow (~5–8 s/page). Always pass `cache_path` to `load_or_build_index()` so the index is not rebuilt on every run.
- **Parsed PDF cache** — store parsed PDF JSON outputs in a separate folder and pass it as `parsed_pdf_cache_dir` so only new or modified PDFs are reparsed.
- **OpenRouter** — all LLM calls go through `https://openrouter.ai/api/v1`. Model names use OpenRouter format (`provider/model-name`).
- **Chunk size** — default 800 chars with 100-char overlap in `rag_core.chunk_text()`. Adjust via `scrape_and_chunk(urls, chunk_size=..., overlap=...)`.
