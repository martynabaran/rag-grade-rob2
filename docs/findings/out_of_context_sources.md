# References Shown for Out-of-Context Questions

Requirement **R3**: when a question is outside the knowledge base, the chatbot must not list
references.

## Current behaviour

1. `rag_core.rag_answer_with_sources()` always retrieves the top-k chunks with
   `index.similarity_search(...)`. FAISS always returns k results, however unrelated they are.
2. `sources` is built directly from the metadata of those k chunks.
3. The system prompt tells the LLM to say when the context is insufficient, and it usually
   does.
4. `app.py` then always appends `_format_sources(sources)` to the answer.

Result: a question like "What is the capital of Poland?" gets a correct refusal followed by a
"Sources" block with GRADE/RoB2 links that were never used.

## Proposed fix

Two complementary checks, both inside `rag_answer_with_sources()` so the app (and later the
backend API) only has to handle an empty source list:

1. **Retrieval threshold.** Switch to `similarity_search_with_score()` and treat the question
   as out of scope when even the best chunk is below a relevance threshold. In that case skip
   the LLM call, return a fixed refusal and `sources = []`. The threshold should be tuned on
   the in-scope and out-of-scope evaluation questions.
2. **Explicit answer status from the LLM.** Ask the model to start its reply with a marker
   (e.g. `[NO_ANSWER]`) when the context does not support an answer. Strip the marker and
   return `sources = []`. Optionally ask the model to cite which context blocks it used and
   only return those sources.

Constraints:

- Keep the return signature `(answer, contexts, sources)`; `rag_answer()` used by the
  evaluation notebooks stays unchanged.
- `_format_sources([])` already returns an empty string, so the UI change is minimal.
- Log out-of-scope decisions in `feedback.jsonl` so false refusals can be reviewed.

## How to verify

- Out-of-scope questions (general knowledge, other medical topics): no "Sources" block.
- In-scope questions from the evaluation set: answers still include sources; measure the
  false-refusal rate.

## Implemented (Phase 2)

Both checks are in `rag_core.rag_answer_with_sources()`:

| Check | Behaviour |
|---|---|
| `MAX_RELEVANCE_DISTANCE = 1.35` | If the closest chunk is further away, the LLM is not called; `OUT_OF_SCOPE_ANSWER` is returned with no contexts and no sources. Pass `max_distance=None` to disable. |
| `NO_ANSWER_MARKER` (`[NO_ANSWER]`) | Added to the system prompt. If the reply contains it, the marker is removed and no sources are returned (contexts are kept for evaluation). |

The app additionally marks Partially Up-To-Date GRADE sources with a warning in the Sources
block.

### Threshold tuning

`python tune_relevance_threshold.py` (embeddings only, no LLM calls) on the rebuilt indices:

| Domain | In-scope questions | Max in-scope distance | Out-of-scope refused at 1.35 |
|---|---|---|---|
| GRADE | 1104 | 1.319 | 76.7% (23 / 30) |
| RoB2 | 65 | 0.835 | 93.3% (28 / 30) |

No in-scope question is refused by the threshold. All general-knowledge questions are refused
by it; the questions that pass are close to the domain (e.g. "How do I calculate a sample size
for a survey?", "How does chemotherapy work?") and are left to the `[NO_ANSWER]` marker.

### Verified

- Out-of-scope question in the app ("What is the capital of Poland?"): refusal, no Sources block.
- Stubbed LLM: in-scope questions keep their sources; a `[NO_ANSWER]` reply returns no sources;
  vague follow-ups still resolve through the conversation history.

**Not yet verified:** how reliably each model in the app emits `[NO_ANSWER]` — needs a real
API key; measure it with the out-of-scope set during Phase 4.
