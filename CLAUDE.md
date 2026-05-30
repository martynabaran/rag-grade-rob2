# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A RAG (Retrieval-Augmented Generation) chatbot system for two medical methodology domains:
- **GRADE** — methodology for developing evidence-based healthcare guidelines
- **RoB2** — Cochrane Risk of Bias 2.0 tool for assessing bias in randomised trials

Both chatbots scrape their source websites, build local FAISS vector indices, and answer questions strictly from retrieved content. The system includes a Gradio web UI and a comprehensive automated and human evaluation pipeline.

## Environment Setup

Copy `.env.example` to `.env` and fill in:
```
OPENROUTER_API_KEY=...
MODEL_NAME=google/gemini-2.5-flash        # default chatbot model
EVAL_MODEL_NAME=google/gemini-2.5-flash   # evaluation model
SOURCE_KEY=GRADE Book                     # for legacy notebook
```

Chrome must be installed (Selenium uses it for JS-rendered pages). `webdriver-manager` handles the ChromeDriver automatically.

## Running the App

```bash
python app.py
# opens at http://127.0.0.1:7860
```

No external hosting required. All state is local.

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
- `rag_answer(query, index, llm, k=5)` — returns `(answer, contexts)` tuple. **Do not change the return signature** — evaluation notebooks depend on it.
- `rag_answer_with_sources(query, index, llm, k=5, system_prompt, history)` — used by `app.py`. Returns `(answer, contexts, sources)`. Accepts optional `history` (Gradio message list) for multi-turn context: last 2 user turns enrich the FAISS query; last 6 messages are included in the LLM prompt. **Do not merge with `rag_answer`.**
- `build_llm(model_name=None)` — constructs `ChatOpenAI` pointed at OpenRouter. Pass `model_name` directly to override `MODEL_NAME` env var.

### Gradio App (`app.py`)

Standalone local app. All features are per-session via `gr.State`.

**Audience selector (radio buttons):**

| Option | Behaviour |
|---|---|
| 🩺 Expert (clinician / researcher) | Full technical terminology, concise, no basic explanations |
| 📋 Stakeholder (policy / guideline) | Practical implications, moderate language, brief definitions |
| 👤 Layperson | Plain language, everyday analogies, jargon-free |

Implemented as `AUDIENCE_SUFFIXES` dict appended to the base system prompt at request time.

**Model selector (radio buttons):**

| Label | OpenRouter ID |
|---|---|
| ⚡ Gemini 2.5 Flash | `google/gemini-2.5-flash` |
| 🔬 Gemini 2.0 Flash Lite | `google/gemini-2.0-flash-lite` |
| 🦙 Llama 3.1 8B | `meta-llama/llama-3.1-8b-instruct` |
| 🌬️ Mistral 7B | `mistralai/mistral-7b-instruct` |
| 🔷 Qwen 2.5 7B | `qwen/qwen-2.5-7b-instruct` |

LLM instances are cached in `_llm_cache` (dict keyed by label). The default model is pre-warmed at startup.

**Multi-turn conversation context:**
- FAISS retrieval: current question enriched with last 2 user turns for correct resolution of follow-up questions
- LLM prompt: last 6 messages included as `Conversation history:` block; source blocks stripped from assistant messages before inclusion to avoid prompt bloat

**Feedback system:**
- Thumbs up/down via Gradio 5 built-in `chatbot.like()` event (appears on hover over any message)
- Optional free-text comment box always visible below the chat
- All ratings saved to `feedback.jsonl` (newline-delimited JSON)

`feedback.jsonl` schema:
```
timestamp, session_id, domain, audience, model, question, answer, sources, rating, comment
```

Analysing feedback:
```python
import pandas as pd
df = pd.read_json("feedback.jsonl", lines=True)
df.groupby(["domain", "rating"]).size()
df.groupby(["model", "rating"]).size()
df[df["rating"] == "down"][["question", "comment"]]
```

**Source links:** every assistant message ends with a `📚 Sources:` markdown block with clickable links to the original pages/PDFs retrieved during RAG.

**Per-session isolation:** every browser tab gets independent copies of chat history, `last_qa`, audience selection, and model selection via `gr.State`. Session ID generated as `gr.State(lambda: str(uuid.uuid4()))`.

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

### Data Layout

```
.faiss_cache/
  grade/              # FAISS index for GRADE (built from web + PDFs)
  rob2/               # FAISS index for RoB2
evaluation_details/
  evaluation_gradebook/   # GRADE results (Q&A pairs, per-model metrics, RAGAS details)
  evaluation_rob2/        # RoB2 results (full_dataset_rob2.csv generated by notebook)
  llm_judge/              # Per-judge CSVs + summary
  human_eval/             # Pre-filled annotation CSVs + annotation_guide.md
feedback.jsonl            # Live user feedback from the Gradio app (created on first rating)
```

Evaluation CSVs follow the naming pattern `{metric_type}_{chat_model}_{eval_model}.csv`.

## Key Design Decisions

- **No LangGraph** — the original agentic architecture (4-node state graph) was replaced with plain function calls in `rag_core.py`. Do not re-introduce LangGraph.
- **No LangGraph-era PDF figure pipeline** — the old GROBID/figure-extraction path is not part of the new architecture.
- **FAISS cache** — scraping is slow (~5–8 s/page). Always pass `cache_path` to `load_or_build_index()` so the index is not rebuilt on every run.
- **Parsed PDF cache** — store parsed PDF JSON outputs in a separate folder and pass it as `parsed_pdf_cache_dir` so only new or modified PDFs are reparsed.
- **OpenRouter** — all LLM calls go through `https://openrouter.ai/api/v1`. Model names use OpenRouter format (`provider/model-name`).
- **Chunk size** — default 800 chars with 100-char overlap in `rag_core.chunk_text()`. Adjust via `scrape_and_chunk(urls, chunk_size=..., overlap=...)`.
- **Two RAG functions** — `rag_answer` (2-tuple, used by notebooks) and `rag_answer_with_sources` (3-tuple + history, used by app). Keep them separate.
- **LLM cache in app** — `_llm_cache` dict prevents recreating the same `ChatOpenAI` instance on every request. Add new models to the `MODELS` dict in `app.py`; no other changes needed.
