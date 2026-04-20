# %% [markdown]
# # RoB2 RAG Chatbot
# 
# Same RAG pipeline as `grade_rag.ipynb` — only the source URLs and system prompt differ.
# 
# **Source:** [Risk of Bias 2.0 Tool](https://www.riskofbias.info/welcome/rob-2-0-tool)
# 
# **RoB2 domains covered:**
# 1. Bias arising from the randomization process
# 2. Bias due to deviations from intended interventions
# 3. Bias due to missing outcome data
# 4. Bias in measurement of the outcome
# 5. Bias in selection of the reported result

# %%
import os, warnings
warnings.filterwarnings('ignore')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dotenv import load_dotenv
load_dotenv()

from rag_core import get_pdf_paths, load_or_build_index, rag_answer, build_llm

# %% [markdown]
# ## 1. Configuration

# %%
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

ROB2_PDF_DIR = "data/rob2_pdfs"
PARSED_ROB2_PDF_DIR = "data/parsed_rob2"
CACHE_DIR = ".faiss_cache/rob2"

rob2_pdf_paths = get_pdf_paths(ROB2_PDF_DIR)
print(f"Found {len(rob2_pdf_paths)} RoB2 PDFs")

# %% [markdown]
# ## 2. Build / Load Index
# 
# First run scrapes all RoB2 pages and saves a local FAISS cache.  
# Subsequent runs load from cache instantly.
# 
# > **Note:** some subpages on riskofbias.info may not be JavaScript-rendered.  
# > If a page returns empty text, increase `wait_time` below.

# %%
import shutil
shutil.rmtree(".faiss_cache/rob2", ignore_errors=True)                                               
print("Cache cleared.")

# %%
index = load_or_build_index(
    pdf_paths=rob2_pdf_paths,
    cache_path=CACHE_DIR,
    parsed_pdf_cache_dir=PARSED_ROB2_PDF_DIR,
)
llm = build_llm()
print("Ready.")

# %% [markdown]
# ## 3. Inspect Scraped Content
# 
# Verify the index contains meaningful RoB2 text before evaluation.

# %%
# Quick sanity check: retrieve top chunks for a known RoB2 concept
probe_query = "signalling questions randomization domain"
docs = index.similarity_search(probe_query, k=3)
for i, doc in enumerate(docs, 1):
    print(f"--- Chunk {i} (source: {doc.metadata['source']}) ---")
    print(doc.page_content[:400])
    print()

# %% [markdown]
# ## 4. Ask a Question

# %%
query = "What are the five domains assessed in RoB2?"
answer, contexts = rag_answer(query, index, llm, k=5, system_prompt=ROB2_SYSTEM_PROMPT)
print("Answer:\n", answer)
print("\n--- Retrieved contexts ---")
for i, ctx in enumerate(contexts, 1):
    print(f"[{i}] {ctx[:200]}...")

# %% [markdown]
# ## 5. Sanity-Check Questions

# %%
sanity_questions = [
    "In my risk of bias assessment, can I add an extra domain corresponding to the quality of statistical results presentation?",
    "Do the different risk of bias domains have different implications in the overall result classification?",
    "I have five domains classified as 'Low risk of bias' and one domain classified as 'High risk of bias'. What is the overall risk of bias judgement?",
    "Is it adequate to stop risk of bias assessment once one of the domains is judged at high risk of bias?",
    "In my systematic review, I am assessing an educational intervention. Can I assume upfront that it is not possible to implement allocation concealment?",
    # "What makes an outcome measurement blinded vs unblinded in RoB2?",
    # "What is selective outcome reporting and which domain covers it?",
    # "When would you rate a trial as 'High' risk of bias for Domain 2?",
    # "What is the overall RoB2 judgment if one domain is rated High?",
    # "How does RoB2 differ from the original Cochrane Risk of Bias tool (RoB1)?",
]

for q in sanity_questions:
    ans, _ = rag_answer(q, index, llm, k=5, system_prompt=ROB2_SYSTEM_PROMPT)
    print(f"Q: {q}")
    print(f"A: {ans[:300]}")
    print()

# %%
docs = index.similarity_search("risk of bias domain signalling questions", k=5)                      
for i, doc in enumerate(docs, 1):                                                                    
      print(f"--- Chunk {i} ---")                                                                      
      print(f"Source: {doc.metadata['source']}")                                                       
      print(doc.page_content[:300])                                                                    
      print()  

# %% [markdown]
# ## 6. Convenience Functions (for evaluation notebooks)

# %%
def ask_rob2(question: str) -> str:
    answer, _ = rag_answer(question, index, llm, system_prompt=ROB2_SYSTEM_PROMPT)
    return answer

def get_rob2_contexts(question: str, k: int = 5) -> list[str]:
    _, contexts = rag_answer(question, index, llm, k=k, system_prompt=ROB2_SYSTEM_PROMPT)
    return contexts

# %% [markdown]
# ## 7. Optional Gradio UI

# %%
import gradio as gr

def gradio_fn(question, history):
    answer, _ = rag_answer(question, index, llm, system_prompt=ROB2_SYSTEM_PROMPT)
    return answer

demo = gr.ChatInterface(
    fn=gradio_fn,
    title="RoB2 Assistant",
    description="Ask questions about the Cochrane Risk of Bias 2.0 tool. Answers are grounded in official RoB2 documentation.",
)
# Uncomment to launch:
demo.launch()


