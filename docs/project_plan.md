# Project Plan

Work plan for meeting the [instructor requirements](instructor_requirements.md).
Owners are assigned at team meetings; tick tasks off as they are merged.

## Phase 0 — Clarify scope (first class)

- [ ] Present the source audit ([findings/grade_source_audit.md](findings/grade_source_audit.md)).
- [ ] Present the current evaluation state ([findings/evaluation_audit.md](findings/evaluation_audit.md)).
- [ ] Ask the open questions listed in [instructor_requirements.md](instructor_requirements.md#open-questions-for-the-instructor)
      and record the answers there.

## Phase 1 — Source cleanup (R1) — done

**Owner:** Martyna

- [x] Remove outdated PDFs 1, 3, 4, 5, 6, 7, 8, 14, 15 from `data/grade_pdfs/` and their
      JSON caches from `data/parsed_grade/`.
- [ ] Apply the instructor's decision on PDFs 9 and 11 (mixed status) — kept as partially
      up-to-date until then.
- [x] Add a source status map (`GRADE_SOURCE_STATUS` in `sources.py`) and store it in
      chunk metadata (`metadata["status"]`).
- [x] Add missing Current GRADE Book pages to `GRADE_URLS` (indirectness, decision
      thresholds, outcomes, EtD introduction, recommendations, history).
- [x] Rebuild `.faiss_cache/grade` (14 pages + 34 PDFs, 1145 chunks) and commit the new index.
- [x] Remove the unconditional `shutil.rmtree(".faiss_cache/rob2")` cell from `rob2_rag.ipynb`.
- [ ] Optional: add Current guidance PDFs 42, 43, 45, 46.

**Done when:** no Outdated source is in the GRADE index, and the index is rebuilt.

## Phase 2 — Out-of-context answers without references (R3) — done

**Owner:** Martyna · Details: [findings/out_of_context_sources.md](findings/out_of_context_sources.md)

- [x] Add a relevance threshold using `similarity_search_with_score()` in
      `rag_answer_with_sources()`.
- [x] Add an explicit "no answer" marker to the system prompt and return `sources = []`
      when it is present.
- [x] Tune the threshold on in-scope vs. out-of-scope evaluation questions
      (`tune_relevance_threshold.py`, `evaluation/datasets/out_of_scope_questions.csv`).
- [x] Flag Partially Up-To-Date GRADE sources in the app's Sources block.
- [ ] Measure the `[NO_ANSWER]` behaviour per model with a real API key (part of Phase 4).

**Done when:** out-of-scope questions never show a "Sources" block, and in-scope answers
still do.

## Phase 3 — Backend / frontend split (R2)

**Owner:** _TBD_

- [ ] `backend/` — FastAPI service that loads both FAISS indices and the LLM cache once at
      startup and exposes:
  - `POST /chat` — `{domain, question, audience, model, history}` → `{answer, sources, contexts}`
  - `POST /feedback` — stores a rating in `feedback.jsonl`
  - `GET /health`, `GET /models`
- [ ] `frontend/` — Gradio UI (current layout, audience and model selectors, feedback) that
      calls the backend over HTTP; backend address from `BACKEND_URL`.
- [ ] Keep `rag_core.py` as the shared library used by the backend and notebooks.
- [ ] `docker-compose.yml` with two services (`backend`, `frontend`), plus run instructions
      without Docker.
- [ ] Separate `requirements` for backend and frontend.

**Done when:** backend and frontend start as two processes on different ports and the full
chat + feedback flow works through the API.

## Phase 4 — Extended evaluation by difficulty (R4)

**Owner:** _TBD_ · Details: [findings/evaluation_audit.md](findings/evaluation_audit.md)

- [ ] Agree on difficulty definitions (easy / medium / hard) and the dataset schema.
- [ ] GRADE: 20 easy + 20 medium + 20 hard questions from current sources (web pages and PDFs),
      manually reviewed.
- [ ] RoB2: 20 easy + 20 medium + 20 hard questions from the sources actually in the RoB2
      index, manually reviewed.
- [ ] Out-of-scope set (≈10 per domain) for R3.
- [ ] Point `automated_evaluation_extended.ipynb`, `llm_judge.ipynb` and
      `prepare_human_eval.ipynb` at the same datasets.
- [ ] Report all metrics per domain × difficulty (tables + plots).
- [ ] Re-run after Phases 1–2 and summarise results in `docs/technical_report.md`.

**Done when:** each domain has 60 reviewed questions and results are reported per difficulty
level.

## Dependencies

```
Phase 0 ──► Phase 1 ──► Phase 4 (final run)
        └─► Phase 2 ──► Phase 4 (out-of-scope metrics)
        └─► Phase 3 (independent, can run in parallel)
```

Phases 2 and 3 both touch the app, so they should agree on the `rag_answer_with_sources()`
contract early: the backend returns an empty `sources` list for out-of-scope answers.
