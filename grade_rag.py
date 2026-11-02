# %% [markdown]
# # GRADE RAG Chatbot (Simplified)
# 
# Clean RAG pipeline over the GRADE Book — no LangGraph, no agents, no Docker.
# 
# **Architecture:**
# ```
# scrape_and_chunk(urls) → build_index() → rag_answer(query)
# ```
# 
# **Setup:** copy `.env.example` to `.env` and fill in:
# ```
# OPENROUTER_API_KEY=...
# MODEL_NAME=google/gemini-2.5-flash
# ```

# %%
import os, warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dotenv import load_dotenv
load_dotenv()

from rag_core import get_pdf_paths, load_or_build_index, rag_answer, build_llm
from sources import (
    GRADE_URLS, GRADE_PDF_DIR, PARSED_GRADE_PDF_DIR, GRADE_CACHE_DIR, GRADE_SOURCE_STATUS,
)

# %% [markdown]
# ## 1. Configuration

# %%
GRADE_SYSTEM_PROMPT = (
    "You are an expert assistant specialising in the GRADE methodology for evidence-based "
    "healthcare guidelines. Answer the user's question based ONLY on the retrieved GRADE Book "
    "content below. Do not fabricate or speculate beyond what is in the context. "
    "If the context does not contain enough information, say so explicitly. "
    "Use precise GRADE terminology (certainty of evidence, rating down/up, domains, etc.)."
)

grade_pdf_paths = get_pdf_paths(GRADE_PDF_DIR)
print(f"Found {len(grade_pdf_paths)} GRADE PDFs")

# %% [markdown]
# ## 2. Build / Load Index
# 
# First run scrapes all GRADE Book pages listed in `sources.py` (~1–2 min) and saves a local FAISS cache.  
# Subsequent runs load from cache instantly.

# %%
index = load_or_build_index(
    urls=GRADE_URLS,
    pdf_paths=grade_pdf_paths,
    cache_path=GRADE_CACHE_DIR,
    parsed_pdf_cache_dir=PARSED_GRADE_PDF_DIR,
    source_status=GRADE_SOURCE_STATUS,
)
llm = build_llm()
print("Ready.")

# %% [markdown]
# ## 3. Ask a Question

# %%
query = "What are the four levels of certainty of evidence in GRADE?"
answer, contexts = rag_answer(query, index, llm, k=5, system_prompt=GRADE_SYSTEM_PROMPT)
print("Answer:\n", answer)
print("\n--- Retrieved contexts ---")
for i, ctx in enumerate(contexts, 1):
    print(f"[{i}] {ctx[:200]}...")

# %% [markdown]
# ## 4. Convenience Functions (for evaluation notebooks)

# %%
def ask_grade(question: str) -> str:
    answer, _ = rag_answer(question, index, llm, system_prompt=GRADE_SYSTEM_PROMPT)
    return answer

def get_grade_contexts(question: str, k: int = 5) -> list[str]:
    _, contexts = rag_answer(question, index, llm, k=k, system_prompt=GRADE_SYSTEM_PROMPT)
    return contexts

# %%
questions = [
      "What is a potential risk of combining tests with therapeutic interventions?",   # Basic
      "What can a multiple intervention comparison framework help with?",                        # Intermediate
      "What challenge may arise when comparing different diagnostic tests?",  #
      "What are the causes of incoherence?"

  ]
for q in questions:
      ans, ctx = rag_answer(q, index, llm, k=5, system_prompt=GRADE_SYSTEM_PROMPT)
      print(f"Q: {q}\nA: {ans}\n")

# %% [markdown]
# ## 5. Optional Gradio UI

# %%
import gradio as gr

def gradio_fn(question, history):
    answer, _ = rag_answer(question, index, llm, system_prompt=GRADE_SYSTEM_PROMPT)
    return answer

demo = gr.ChatInterface(
    fn=gradio_fn,
    title="GRADE Book Assistant",
    description="Ask questions about the GRADE methodology. Answers are grounded in the GRADE Book.",
)
# Uncomment to launch:
demo.launch()


