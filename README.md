# Agentic RAG: A Framework for Context-Based Chatbots

This repository contains the source code and evaluation framework for the Bachelor of Science Thesis: **"Agentic RAG: A Framework for Context-Based Chatbots"**.

The project implements an autonomous **Agentic Retrieval-Augmented Generation (RAG)** system designed to handle complex scientific and medical queries. Unlike traditional linear RAG pipelines, this system utilizes a multi-agent architecture to orchestrate retrieval, process visual data (tables/figures), and enforce strict adherence to context to minimize hallucinations.

## Installation & Setup

### Prerequisites
*   Python 3.10+
*   **Docker Desktop** (required for running the GROBID parser).
*   Chrome Browser (for Selenium web scraping).

### 1. Start the GROBID Service
The system relies on [GROBID](https://github.com/kermitt2/grobid) to parse PDF structures (headers, sections, citations). GROBID instance must be running locally before starting the chatbot.

The following command starts the server:

```bash
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.7.3
```

## Key Features

*   **Multi-Agent Orchestration:** Uses `LangGraph` to manage state and coordination between the Retriever, Image Analyst, and Answer Generator agents.
*   **Dual-Domain Support:**
    *   **Unstructured Data:** Ingests and analyzes PDF research papers (Text + Figures).
    *   **Structured Data:** Scrapes and processes web-based medical guidelines (e.g., GRADE Book).
*   **Multimodal Capabilities:** Detects questions about charts/tables, extracts the images from PDFs, and generates descriptions.
*   **Strict Context Adherence:** Engineered system prompts designed to refuse out-of-domain questions (e.g., "Capital of Poland") to ensure scientific rigor.
*   **Hybrid Evaluation:** Includes a comprehensive evaluation setup using **RAGAS** metrics and NLP complexity scores.