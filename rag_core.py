"""
rag_core.py — Shared RAG pipeline for GRADE and RoB2 chatbots.

Three main steps:
  1. scrape_and_chunk(urls)  -> list[Document]
  2. build_index(docs)       -> FAISS vectorstore
  3. rag_answer(query, ...)  -> (answer: str, contexts: list[str])

No LangGraph, no agents, no GROBID — just a clean retrieval + generation loop.
"""

import re
import time
import os

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

# ---------------------------------------------------------------------------
# Embeddings (shared, loaded once)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_embeddings: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embeddings


# ---------------------------------------------------------------------------
# Web scraping
# ---------------------------------------------------------------------------

def fetch_visible_text(url: str, wait_time: int = 5) -> str:
    """Render a JavaScript-heavy page with headless Chrome and return visible text."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()), options=options
    )
    try:
        driver.get(url)
        time.sleep(wait_time)
        page_source = driver.page_source
    finally:
        driver.quit()

    soup = BeautifulSoup(page_source, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)
    text = text.replace("\xa0", " ")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def scrape_and_chunk(
    urls: list[str],
    chunk_size: int = 800,
    overlap: int = 100,
    wait_time: int = 5,
) -> list[Document]:
    """
    Scrape each URL, chunk the text, and return LangChain Documents
    with metadata (source URL).
    """
    documents = []
    for url in urls:
        print(f"  Scraping: {url}")
        try:
            text = fetch_visible_text(url, wait_time=wait_time)
        except Exception as exc:
            print(f"  WARNING: failed to fetch {url}: {exc}")
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"source": url, "chunk_index": i},
                )
            )
        print(f"    -> {len(chunks)} chunks")

    print(f"Total documents: {len(documents)}")
    return documents


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------

def build_index(documents: list[Document]) -> FAISS:
    """Build a FAISS vector index from a list of Documents."""
    return FAISS.from_documents(documents, get_embeddings())


def load_or_build_index(
    urls: list[str],
    cache_path: str | None = None,
    **scrape_kwargs,
) -> FAISS:
    """
    Build the index (optionally saving/loading a local FAISS cache).
    If cache_path exists, load from disk; otherwise scrape + build + save.
    """
    if cache_path and os.path.isdir(cache_path):
        print(f"Loading index from cache: {cache_path}")
        return FAISS.load_local(
            cache_path, get_embeddings(), allow_dangerous_deserialization=True
        )

    print("Building index from scratch...")
    docs = scrape_and_chunk(urls, **scrape_kwargs)
    index = build_index(docs)

    if cache_path:
        index.save_local(cache_path)
        print(f"Index saved to: {cache_path}")

    return index


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert assistant. Answer the user's question based ONLY on the "
    "retrieved context below. Do not fabricate or speculate beyond what is in the "
    "context. If the context does not contain enough information to answer "
    "confidently, say so explicitly. Cite which part of the context supports each "
    "claim where possible."
)


def rag_answer(
    query: str,
    index: FAISS,
    llm: ChatOpenAI,
    k: int = 5,
    system_prompt: str = SYSTEM_PROMPT,
) -> tuple[str, list[str]]:
    """
    Retrieve top-k chunks for `query`, then generate an answer with `llm`.

    Returns:
        answer   — the LLM-generated answer string
        contexts — list of retrieved chunk texts (for evaluation)
    """
    docs = index.similarity_search(query, k=k)
    contexts = [doc.page_content for doc in docs]
    sources = list({doc.metadata.get("source", "") for doc in docs})

    context_block = "\n\n---\n\n".join(contexts)
    prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\nAnswer:"
    )

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)
    return answer, contexts


# ---------------------------------------------------------------------------
# Convenience: build LLM from env
# ---------------------------------------------------------------------------

def build_llm(
    model_env_var: str = "MODEL_NAME",
    base_url: str = "https://openrouter.ai/api/v1",
    api_key_env_var: str = "OPENROUTER_API_KEY",
    temperature: float = 0.0,
) -> ChatOpenAI:
    return ChatOpenAI(
        base_url=base_url,
        api_key=os.getenv(api_key_env_var),
        model=os.getenv(model_env_var),
        temperature=temperature,
    )
