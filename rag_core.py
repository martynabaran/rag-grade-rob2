"""
rag_core.py — Shared RAG pipeline for GRADE and RoB2 chatbots.

Main steps:
  1. scrape_and_chunk(urls) or load_pdfs_and_chunk(pdf_paths) -> list[Document]
  2. build_index(docs)                                        -> FAISS vectorstore
  3. rag_answer(query, ...)                                   -> (answer: str, contexts: list[str])

The module keeps the new simplified architecture while supporting both web pages
and PDF-based sources.
"""

import re
import time
import os
import json
from pathlib import Path

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


def _extract_authors(authors_raw) -> str:
    """Normalize SciPDF author metadata into a readable string."""
    if isinstance(authors_raw, list):
        return ", ".join(
            author.get("name") or author.get("full_name", "")
            for author in authors_raw
            if isinstance(author, dict)
        )
    return str(authors_raw or "")


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
# PDF parsing
# ---------------------------------------------------------------------------

def get_pdf_paths(folder: str | os.PathLike[str]) -> list[Path]:
    """Return a list of PDF file paths in the given folder."""
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"Provided path is not a directory: {folder}")
    return sorted(folder_path.glob("*.pdf"))


def _document_to_dict(doc: Document) -> dict:
    return {
        "page_content": doc.page_content,
        "metadata": doc.metadata,
    }


def _document_from_dict(data: dict) -> Document:
    return Document(
        page_content=data["page_content"],
        metadata=data.get("metadata", {}),
    )


def _parsed_pdf_cache_path(
    pdf_path: str | os.PathLike[str],
    parsed_cache_dir: str | os.PathLike[str],
) -> Path:
    pdf_path = Path(pdf_path)
    parsed_cache_dir = Path(parsed_cache_dir)
    return parsed_cache_dir / f"{pdf_path.stem}.json"


def _is_pdf_cache_stale(
    pdf_path: str | os.PathLike[str],
    cache_path: str | os.PathLike[str],
) -> bool:
    pdf_path = Path(pdf_path)
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return True
    return cache_path.stat().st_mtime < pdf_path.stat().st_mtime


def save_parsed_pdf_documents(
    documents: list[Document],
    cache_path: str | os.PathLike[str],
) -> None:
    """Save parsed PDF documents to a JSON cache file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "documents": [_document_to_dict(doc) for doc in documents],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))


def load_parsed_pdf_documents(cache_path: str | os.PathLike[str]) -> list[Document]:
    """Load parsed PDF documents from a JSON cache file."""
    cache_path = Path(cache_path)
    payload = json.loads(cache_path.read_text())
    return [_document_from_dict(item) for item in payload.get("documents", [])]

def parse_pdf_to_documents(pdf_path: str | os.PathLike[str]) -> list[Document]:
    """
    Parse a PDF with SciPDF Parser and convert it to LangChain Documents.

    Output mirrors the old notebook logic:
    - one summary chunk with title/authors/abstract/pub date
    - one chunk per detected section
    """
    try:
        import scipdf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PDF parsing requires the 'scipdf-parser' package. "
            "Install it before calling parse_pdf_to_documents()."
        ) from exc

    pdf_path = Path(pdf_path)
    article = scipdf.parse_pdf_to_dict(str(pdf_path))

    filename = pdf_path.name
    doc_id = pdf_path.stem
    title = article.get("title") or filename
    authors = _extract_authors(article.get("authors"))
    abstract = article.get("abstract", "")
    pub_date = article.get("pub_date", "")

    documents: list[Document] = []

    summary = (
        f"Title: {title}\n"
        f"Authors: {authors}\n"
        f"Abstract: {abstract}\n"
        f"PubDate: {pub_date}"
    )
    documents.append(
        Document(
            page_content=summary,
            metadata={
                "source": filename,
                "source_type": "pdf",
                "doc_id": doc_id,
                "title": title,
                "authors": authors,
                "pub_date": pub_date,
                "type": "summary",
            },
        )
    )

    sections = article.get("sections") or []
    for idx, section in enumerate(sections, 1):
        heading = section.get("heading") or f"Section {idx}"
        body = section.get("text") or ""
        if not body.strip():
            continue

        documents.append(
            Document(
                page_content=f"{heading}\n{body}",
                metadata={
                    "source": filename,
                    "source_type": "pdf",
                    "doc_id": doc_id,
                    "title": title,
                    "authors": authors,
                    "pub_date": pub_date,
                    "section": heading,
                    "type": "section",
                },
            )
        )

    return documents


def load_pdfs_and_chunk(
    pdf_paths: list[str] | list[os.PathLike[str]],
    parsed_cache_dir: str | os.PathLike[str] | None = None,
) -> list[Document]:
    """
    Parse multiple PDFs and return one combined list of Documents.

    If `parsed_cache_dir` is provided, parsed JSON files are reused unless the
    source PDF is missing from cache or newer than its cache file.
    """
    documents: list[Document] = []
    cache_dir = Path(parsed_cache_dir) if parsed_cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        cache_path = (
            _parsed_pdf_cache_path(pdf_path, cache_dir) if cache_dir else None
        )

        if cache_path and not _is_pdf_cache_stale(pdf_path, cache_path):
            print(f"  Loading parsed PDF from cache: {pdf_path.name}")
            try:
                pdf_docs = load_parsed_pdf_documents(cache_path)
            except Exception as exc:
                print(f"  WARNING: failed to load cache for {pdf_path}: {exc}")
                pdf_docs = []
            else:
                documents.extend(pdf_docs)
                print(f"    -> {len(pdf_docs)} documents")
                continue

        print(f"  Parsing PDF: {pdf_path}")
        try:
            pdf_docs = parse_pdf_to_documents(pdf_path)
        except Exception as exc:
            print(f"  WARNING: failed to parse {pdf_path}: {exc}")
            continue

        if cache_path:
            try:
                save_parsed_pdf_documents(pdf_docs, cache_path)
            except Exception as exc:
                print(f"  WARNING: failed to save cache for {pdf_path}: {exc}")

        documents.extend(pdf_docs)
        print(f"    -> {len(pdf_docs)} documents")

    print(f"Total documents: {len(documents)}")
    return documents


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
                    metadata={
                        "source": url,
                        "source_type": "web",
                        "chunk_index": i,
                    },
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
    urls: list[str] | None = None,
    pdf_paths: list[str] | list[os.PathLike[str]] | None = None,
    cache_path: str | None = None,
    parsed_pdf_cache_dir: str | os.PathLike[str] | None = None,
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
    docs: list[Document] = []

    if urls:
        docs.extend(scrape_and_chunk(urls, **scrape_kwargs))
    if pdf_paths:
        docs.extend(
            load_pdfs_and_chunk(pdf_paths, parsed_cache_dir=parsed_pdf_cache_dir)
        )

    if not docs:
        raise ValueError("No sources provided. Pass at least one URL or PDF path.")

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
