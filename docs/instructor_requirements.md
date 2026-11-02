# Instructor Requirements

Feedback received from the course instructor at the start of the semester continuation.
These are the requirements the project has to meet. The work plan that addresses them is in
[project_plan.md](project_plan.md).

## R1. Use only up-to-date GRADE sources

The [GRADE Living Map](https://book.gradepro.org/map) labels every GRADE article as
**Current**, **Partially Up-To-Date** or **Outdated**. Sources marked as outdated must be
removed from the GRADE knowledge base.

- Outdated articles are listed in the map's **Archive** view (the default *Repository* view
  only shows Current and Partially Up-To-Date articles).
- Findings for our corpus: [findings/grade_source_audit.md](findings/grade_source_audit.md).

## R2. Split the application into two separate instances

The application must run as two independent services:

- **Backend** — retrieval, LLM calls, feedback storage, exposed through an API.
- **Frontend** — the user interface, talking to the backend over HTTP.

Currently `app.py` runs everything (index loading, LLM calls, UI, feedback) in a single
Gradio process.

## R3. No references for out-of-context answers

When the user asks a question that is outside the scope of the knowledge base, the chatbot
must not list references. Currently it still shows a "Sources" block even when it answers
that the context does not contain the information.

- Findings: [findings/out_of_context_sources.md](findings/out_of_context_sources.md).

## R4. Extend the evaluation with difficulty levels

Prepare and verify **easy, medium and hard questions — 20 per category** — and report
evaluation results per difficulty level.

- Findings on the current evaluation: [findings/evaluation_audit.md](findings/evaluation_audit.md).

## Open questions for the instructor

1. **R4:** does "20 per category" mean 20 easy / 20 medium / 20 hard **per domain**
   (GRADE and RoB2, i.e. 120 questions in total), or per topic?
2. **R4:** may questions be LLM-generated if every pair is manually reviewed, or must they be
   written by the team?
3. **R1:** what should happen with **Partially Up-To-Date** sources — keep them (optionally
   with a warning in the answer), or remove them as well?
4. **R1:** GRADE guidelines 9 and 11 are *Partially Up-To-Date* under one topic and
   *Outdated* under another. Keep or remove?
5. **R2:** is a REST backend (FastAPI) + Gradio frontend acceptable, or is a specific
   frontend technology expected?
