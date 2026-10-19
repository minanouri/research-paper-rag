# Research Paper RAG

A retrieval-augmented generation (RAG) system for searching research papers and answering questions using evidence retrieved from their original content.

## Overview

Research papers contain valuable information about methodologies, experiments, and findings, but locating specific information across multiple papers can be time-consuming. This project provides a question-answering system that combines document processing, hybrid information retrieval, and large language models (LLMs) to make research papers easier to search and analyze.

Built with Python and LlamaIndex, the system uses Docling to parse PDF documents and extract their structural information. Paper content is divided into structure-aware chunks enriched with metadata, including paper identifiers, section headings, publication years, and page numbers. ChromaDB enables dense vector retrieval, complemented by BM25 for keyword-based search. The retrieval pipeline combines results using reciprocal rank fusion (RRF) and applies a Sentence Transformers cross-encoder reranker to prioritize passages relevant to each query. An LLM generates answers from the retrieved excerpts, with numbered inline citations linking claims to their supporting evidence. The system also supports metadata extraction, user-provided metadata corrections, and filtering by paper, section type, and publication year.

## Key Features

- **PDF ingestion:** Parse research papers and extract text and document structure using Docling.
- **Structure-aware chunking:** Preserve section and subsection context in searchable document chunks.
- **Metadata management:** Extract, normalize, cache, and override paper metadata.
- **Hybrid retrieval:** Combine dense vector search with BM25 keyword retrieval.
- **Rank fusion and reranking:** Use reciprocal rank fusion and cross-encoder reranking to improve passage relevance.
- **Metadata filtering:** Narrow searches by paper, section type, and minimum publication year.
- **Evidence-grounded question answering:** Generate answers using retrieved excerpts and numbered citations.
- **Persistent storage:** Maintain vector indexes, document data, and paper metadata across runs.

## Project Structure

```text
paper-rag/
├── src/
│   └── paper_rag/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── config.py
│       ├── generate.py
│       ├── index.py
│       ├── ingest.py
│       ├── metadata.py
│       ├── retrieve.py
│       └── sections.py
├── papers/              # Input research papers (PDFs)
├── data/                # Persistent indexes and metadata
├── README.md
└── .gitignore
```

## Configuration and Data

The `config.py` module defines application paths and model and retrieval settings. The `data/` directory stores persistent indexes, document data, the paper registry, and metadata caches.

Metadata sidecar files can be used to correct information that cannot be reliably extracted automatically. Existing index data should be preserved between runs unless a rebuild is intended.

## Usage

Install the project dependencies, configure the required models and LLM settings, and place research paper PDFs in the `papers/` directory.

### Ingest Papers

Ingest all PDFs in the configured directory:

```bash
PYTHONPATH=src python -m paper_rag ingest
```

Ingest a specific paper or refresh metadata:

```bash
PYTHONPATH=src python -m paper_rag ingest <path-to-pdf>
PYTHONPATH=src python -m paper_rag ingest --refresh-meta
```

### Ask Questions

Ask a question across indexed papers:

```bash
PYTHONPATH=src python -m paper_rag ask "<your-question>"
```

Optional filters:

```bash
PYTHONPATH=src python -m paper_rag ask "<your-question>" --paper <paper-id>
PYTHONPATH=src python -m paper_rag ask "<your-question>" --section-type <section-type>
PYTHONPATH=src python -m paper_rag ask "<your-question>" --year-min <year>
```

### List Indexed Papers

```bash
PYTHONPATH=src python -m paper_rag list
```
