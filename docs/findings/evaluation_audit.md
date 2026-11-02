# Evaluation Audit

State of the evaluation pipeline before addressing requirement **R4** (20 easy / 20 medium /
20 hard questions, results reported per difficulty level).

## Datasets

| File | Rows | Difficulty labels | Notes |
|---|---|---|---|
| `evaluation/datasets/full_dataset_gradebook.csv` | 117 | Basic 30, Intermediate 30, Advanced 27, ExpertStyled 30 | 97 AI-generated, 20 expert-written. Covers only the 8 scraped GRADE Book pages — no questions about the PDFs. |
| `evaluation/datasets/questions_gradebook.csv` | 970 | none | `question;expected_response`. Used by `automated_evaluation_extended.ipynb`. |
| `evaluation/datasets/questions_rob2.csv` | 65 | none | `question;expected_response`. Used by `automated_evaluation_extended.ipynb`. |
| `evaluation/datasets/screening_questions.csv` | 17 | none | Used by `run_screening_eval.py`. |
| `evaluation/datasets/full_dataset_rob2.csv` | — | — | **Never generated.** Expected by `llm_judge.ipynb` and `prepare_human_eval.ipynb`, which skip RoB2 when it is missing. |

## Notebooks

| Notebook | Status |
|---|---|
| `generate_rob2_dataset.ipynb` | Generates 5 Basic + 5 Intermediate + 4 Advanced pairs for each of 7 riskofbias.info pages (~98 pairs). Not run yet. |
| `automated_evaluation_extended.ipynb` | Runs RAGAS, ROUGE, BLEU, semantic similarity, complexity and G-Eval on `questions_*.csv`. Reports **averages only — no breakdown by difficulty**. Results in `evaluation/results/grade/` and `evaluation/results/rob2/`. |
| `llm_judge.ipynb` | Two-judge scoring + Cohen's Kappa. **No results committed** (`evaluation/results/llm_judge/` does not exist yet). |
| `prepare_human_eval.ipynb` | Stratified sampling by difficulty exists. No annotation CSVs committed yet. |

## Issues

1. **No balanced difficulty set.** GRADE has 27–30 per level (mostly AI-generated, unreviewed);
   RoB2 has none.
2. **Results are not split by difficulty**, so we cannot show how the chatbot performs on
   easy vs. hard questions.
3. **Different notebooks use different datasets** (`questions_*.csv` vs. `full_dataset_*.csv`),
   so automated metrics and LLM-judge scores are not comparable.
4. **GRADE questions only cover the web pages**, while the index also contains 43 PDFs.
5. **Some GRADE questions may target outdated sources** that will be removed
   (see [grade_source_audit.md](grade_source_audit.md)).
6. **RoB2 mismatch:** the RoB2 index is built from the 3 PDFs in `data/rob2_pdfs/`, but
   `generate_rob2_dataset.ipynb` generates questions from riskofbias.info web pages that are
   not in the index.
7. **`rob2_rag.ipynb` deletes the RoB2 FAISS cache on every run** (`shutil.rmtree`), forcing a
   full rebuild.
8. **No out-of-scope questions**, so we cannot measure whether the chatbot correctly refuses
   (relevant to R3).

## Recommended actions

1. Create one dataset per domain with a single schema:
   `id, domain, difficulty (easy|medium|hard), topic, source, question, reference_answer, origin (expert|generated), reviewed`.
2. 20 easy + 20 medium + 20 hard per domain, generated from **current** sources only and
   manually reviewed. Reuse good existing GRADE pairs where possible.
3. Define difficulty explicitly:
   - **easy** — single fact or definition stated directly in one source passage;
   - **medium** — explanation or comparison that combines several statements from one source;
   - **hard** — reasoning across sources, applying the methodology to a scenario, or nuanced
     judgements.
4. Add a set of out-of-scope questions (e.g. 10 per domain) with expected behaviour
   "refuse, no references".
5. Use the same datasets in all evaluation notebooks and add per-difficulty aggregation
   (`groupby("difficulty")`) to the reports.
6. Re-run the evaluation after the source cleanup (R1) so results reflect the final corpus.
