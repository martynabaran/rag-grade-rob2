"""
tune_relevance_threshold.py

Compares the distance of the closest retrieved chunk for in-scope evaluation
questions and for out-of-scope questions
(evaluation/datasets/out_of_scope_questions.csv), for both indices.
Use it to choose rag_core.MAX_RELEVANCE_DISTANCE after changing sources.
No LLM calls are made.

Usage:
    python tune_relevance_threshold.py
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from langchain_community.vectorstores import FAISS

from rag_core import MAX_RELEVANCE_DISTANCE, get_embeddings
from sources import GRADE_CACHE_DIR, ROB2_CACHE_DIR

DATASETS = "evaluation/datasets"
THRESHOLDS = [1.0, 1.1, 1.2, 1.25, 1.3, 1.35, 1.4]


def in_scope_questions() -> dict[str, list[str]]:
    grade = pd.concat([
        pd.read_csv(f"{DATASETS}/full_dataset_gradebook.csv")["Question"],
        pd.read_csv(f"{DATASETS}/questions_gradebook.csv", sep=";")["question"],
        pd.read_csv(f"{DATASETS}/screening_questions.csv", sep=";")["question"],
    ])
    rob2 = pd.read_csv(f"{DATASETS}/questions_rob2.csv", sep=";")["question"]
    return {"grade": grade.dropna().tolist(), "rob2": rob2.dropna().tolist()}


def best_distances(index: FAISS, questions: list[str]) -> np.ndarray:
    return np.array([index.similarity_search_with_score(q, k=1)[0][1] for q in questions])


def main():
    embeddings = get_embeddings()
    indices = {
        "grade": FAISS.load_local(GRADE_CACHE_DIR, embeddings, allow_dangerous_deserialization=True),
        "rob2": FAISS.load_local(ROB2_CACHE_DIR, embeddings, allow_dangerous_deserialization=True),
    }
    out_of_scope = pd.read_csv(f"{DATASETS}/out_of_scope_questions.csv")["question"].tolist()

    print(f"Current MAX_RELEVANCE_DISTANCE = {MAX_RELEVANCE_DISTANCE}")
    for domain, questions in in_scope_questions().items():
        ins = best_distances(indices[domain], questions)
        oos = best_distances(indices[domain], out_of_scope)
        print(f"\n{domain}: {len(ins)} in-scope, {len(oos)} out-of-scope questions")
        print(f"  in-scope max distance: {ins.max():.3f}, out-of-scope min distance: {oos.min():.3f}")
        print("  threshold | in-scope refused | out-of-scope refused")
        for t in THRESHOLDS:
            print(f"  {t:9.2f} | {np.mean(ins > t):15.1%} | {np.mean(oos > t):19.1%}")


if __name__ == "__main__":
    main()
