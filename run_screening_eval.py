"""
run_screening_eval.py

Runs the GRADE RAG chatbot on all 17 questions from
  evaluation/Screening questions(Arkusz1).csv
and saves results to evaluation/screening_answers.csv.

Usage:
    python run_screening_eval.py

Output columns:
    question, expected_response, answer, contexts (pipe-separated chunks)
"""

import csv
import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

load_dotenv()

from rag_core import get_pdf_paths, load_or_build_index, rag_answer, build_llm

# ---------------------------------------------------------------------------
# GRADE config (mirrors grade_rag.py)
# ---------------------------------------------------------------------------

GRADE_URLS = [
    "https://book.gradepro.org/guideline/overview-of-the-grade-approach",
    "https://book.gradepro.org/guideline/the-development-methods-of-grade",
    "https://book.gradepro.org/guideline/requirements-for-claiming-the-use-of-grade",
    "https://book.gradepro.org/guideline/questions-about-interventions-diagnostic-test-prognosis-and-exposures",
    "https://book.gradepro.org/guideline/risk-of-bias-randomized-trials",
    "https://book.gradepro.org/guideline/inconsistency",
    "https://book.gradepro.org/guideline/imprecision",
    "https://book.gradepro.org/guideline/dissemination-bias",
]

GRADE_SYSTEM_PROMPT = (
    "You are an expert assistant specialising in the GRADE methodology for evidence-based "
    "healthcare guidelines. Answer the user's question based ONLY on the retrieved GRADE Book "
    "content below. Do not fabricate or speculate beyond what is in the context. "
    "If the context does not contain enough information, say so explicitly. "
    "Use precise GRADE terminology (certainty of evidence, rating down/up, domains, etc.)."
)

GRADE_PDF_DIR = "data/grade_pdfs"
PARSED_GRADE_PDF_DIR = "data/parsed_grade"
CACHE_DIR = ".faiss_cache/grade"

INPUT_CSV = Path("evaluation/Screening questions(Arkusz1).csv")
OUTPUT_CSV = Path("evaluation/screening_answers.csv")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Build / load index
    grade_pdf_paths = get_pdf_paths(GRADE_PDF_DIR) if Path(GRADE_PDF_DIR).is_dir() else []
    print(f"Found {len(grade_pdf_paths)} GRADE PDFs")

    index = load_or_build_index(
        urls=GRADE_URLS,
        pdf_paths=grade_pdf_paths,
        cache_path=CACHE_DIR,
        parsed_pdf_cache_dir=PARSED_GRADE_PDF_DIR,
    )
    llm = build_llm()
    print("Index ready.\n")

    # Read questions
    rows = []
    with open(INPUT_CSV, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            rows.append(row)

    print(f"Loaded {len(rows)} questions from {INPUT_CSV}\n")

    # Run RAG for each question
    results = []
    for i, row in enumerate(rows, 1):
        question = row["question"].strip()
        expected = row.get("expected_response", "").strip()
        print(f"[{i}/{len(rows)}] {question[:80]}...")

        answer, contexts = rag_answer(
            question, index, llm, k=5, system_prompt=GRADE_SYSTEM_PROMPT
        )
        results.append(
            {
                "question": question,
                "expected_response": expected,
                "answer": answer,
                "contexts": " | ".join(ctx.replace("\n", " ") for ctx in contexts),
            }
        )
        print(f"        -> {len(answer)} chars\n")

    # Save results
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["question", "expected_response", "answer", "contexts"]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} answers to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
