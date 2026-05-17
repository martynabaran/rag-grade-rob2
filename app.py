"""
app.py — Gradio web app for the GRADE and RoB2 RAG chatbots.

Loads pre-built FAISS indices from .faiss_cache/ at startup.
Requires OPENROUTER_API_KEY in the environment (or a .env file).

Deploy to Hugging Face Spaces:
  - Set OPENROUTER_API_KEY as a Space secret.
  - Commit .faiss_cache/ alongside this file and rag_core.py.
  - Use requirements_app.txt as requirements.txt in the Space.
"""

import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("MODEL_NAME", "google/gemini-2.5-flash")

import gradio as gr
from langchain_community.vectorstores import FAISS

from rag_core import get_embeddings, rag_answer, build_llm

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

GRADE_SYSTEM_PROMPT = (
    "You are an expert assistant specialising in the GRADE methodology for evidence-based "
    "healthcare guidelines. Answer the user's question based ONLY on the retrieved GRADE Book "
    "content below. Do not fabricate or speculate beyond what is in the context. "
    "If the context does not contain enough information, say so explicitly. "
    "Use precise GRADE terminology (certainty of evidence, rating down/up, domains, etc.)."
)

ROB2_SYSTEM_PROMPT = (
    "You are an expert assistant specialising in the Cochrane Risk of Bias tool 2.0 (RoB2). "
    "RoB2 assesses risk of bias in randomized trials across five domains: "
    "(1) randomization process, (2) deviations from intended interventions, "
    "(3) missing outcome data, (4) outcome measurement, (5) selection of the reported result. "
    "Answer the user's question based ONLY on the retrieved RoB2 content below. "
    "Do not fabricate or speculate beyond what is in the context. "
    "If the context does not contain enough information, say so explicitly. "
    "Use precise RoB2 terminology (signalling questions, bias judgments: Low/Some concerns/High, etc.)."
)

# ---------------------------------------------------------------------------
# Load indices and LLM once at startup
# ---------------------------------------------------------------------------

print("Loading GRADE index...")
grade_index = FAISS.load_local(
    ".faiss_cache/grade", get_embeddings(), allow_dangerous_deserialization=True
)
print("Loading RoB2 index...")
rob2_index = FAISS.load_local(
    ".faiss_cache/rob2", get_embeddings(), allow_dangerous_deserialization=True
)
print("Building LLM...")
llm = build_llm()
print("Ready.")

# ---------------------------------------------------------------------------
# Chat handlers
# ---------------------------------------------------------------------------


def answer_grade(question: str, history: list) -> str:
    answer, _ = rag_answer(question, grade_index, llm, system_prompt=GRADE_SYSTEM_PROMPT)
    return answer


def answer_rob2(question: str, history: list) -> str:
    answer, _ = rag_answer(question, rob2_index, llm, system_prompt=ROB2_SYSTEM_PROMPT)
    return answer


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Evidence-Based Medicine Assistants") as demo:
    gr.Markdown(
        """# Evidence-Based Medicine RAG Assistants
Answers are grounded strictly in official methodology documentation — no fabrication or speculation beyond the retrieved content."""
    )

    with gr.Tabs():
        with gr.TabItem("GRADE Methodology"):
            gr.ChatInterface(
                fn=answer_grade,
                title="GRADE Book Assistant",
                description=(
                    "Ask questions about the GRADE methodology for developing evidence-based "
                    "healthcare guidelines. Covers certainty of evidence, rating domains "
                    "(risk of bias, inconsistency, imprecision, indirectness, dissemination bias), "
                    "evidence profiles, and more."
                ),
                examples=[
                    "What are the four levels of certainty of evidence in GRADE?",
                    "What are the five domains for rating down certainty of evidence?",
                    "When can certainty of evidence be rated up?",
                    "What is the difference between imprecision and inconsistency in GRADE?",
                ],
                cache_examples=False,
            )

        with gr.TabItem("RoB2 Tool"):
            gr.ChatInterface(
                fn=answer_rob2,
                title="RoB2 Assistant",
                description=(
                    "Ask questions about the Cochrane Risk of Bias 2.0 (RoB2) tool for "
                    "assessing bias in randomised trials. Covers the five domains, "
                    "signalling questions, domain-level and overall bias judgments."
                ),
                examples=[
                    "What are the five domains assessed in RoB2?",
                    "What does 'Some concerns' mean in an overall RoB2 judgment?",
                    "If one domain is rated High, what is the overall bias judgment?",
                    "What are the signalling questions for the randomization domain?",
                ],
                cache_examples=False,
            )

if __name__ == "__main__":
    demo.launch()
