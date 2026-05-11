# Vector Creator

A document ingestion pipeline that parses, classifies, embeds, and stores documents into a PostgreSQL pgvector database for retrieval-augmented generation (RAG).

## Overview

Vector Creator automates the full ingestion workflow for unstructured documents:

1. **Load** — scans a folder for supported documents (PDF, DOCX, TXT, CSV)
2. **Parse** — converts raw files into structured text via LlamaParse
3. **Classify** — tags each document with a type (`contract`, `policy`, `faq`, etc.) using rule-based matching, with an Ollama LLM fallback
4. **Chunk** — splits parsed content into overlapping 512-token chunks
5. **Embed** — generates 384-dimensional vectors using a HuggingFace sentence-transformer model
6. **Store** — upserts nodes into a PostgreSQL table with the `pgvector` extension

A lightweight RAG retriever (`RagRetriever`) can then query the stored vectors for semantic search.

## Project Structure

```
vector_creator/
├── app/
│   ├── core/
│   │   ├── config.py          # Pydantic settings loaded from .env
│   │   ├── constants.py       # Document type labels and classification rules
│   │   └── logging.py         # Logging setup
│   ├── database/
│   │   ├── pgvector_service.py  # PGVectorStore factory
│   │   └── postgres_service.py  # Raw SQLAlchemy connection
│   ├── embeddings/
│   │   └── embedding_service.py # HuggingFace embedding wrapper
│   ├── loaders/
│   │   └── folder_loader.py   # Recursive folder scanner
│   ├── parsers/
│   │   └── llama_parse_service.py  # LlamaParse document parser
│   ├── pipelines/
│   │   └── ingestion_pipeline.py   # Orchestrates the full parse→store flow
│   ├── rag/
│   │   ├── retriever.py       # Async semantic retriever
│   │   └── query_test.py      # Manual retrieval test script
│   ├── enrichment.py          # DocumentMetadataEnricher transform
│   └── llm_classifier.py      # Ollama-backed LLM classifier
├── docker/
│   └── postgres/init/
│       └── 01-enable-pgvector.sql
├── docker-compose.yml
├── main.py                    # Entry point
└── requirements.txt
```

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- [Ollama](https://ollama.com) running locally with a model loaded (e.g. `llama3.2:3b-instruct-q4_K_M`)
- A [LlamaCloud](https://cloud.llamaindex.ai) API key for LlamaParse

## Setup

### 1. Start the database

Copy the example env file and fill in your values:

```bash
cp .env.docker.example .env.docker
```

Then start Postgres with pgvector:

```bash
docker compose up -d
```

### 2. Configure environment

Create a `.env` file in the project root (see `.env.docker.example` for the database fields). Required variables:

| Variable | Description |
|---|---|
| `POSTGRES_URL` | Full connection URL, e.g. `postgresql://user:pass@localhost:5433/dbname` |
| `VECTOR_TABLE` | Name of the table to store embeddings |
| `EMBEDDING_MODEL` | HuggingFace model name, e.g. `BAAI/bge-small-en-v1.5` |
| `EMBEDDING_DIM` | Embedding dimension (default: `384`) |
| `LLAMA_CLOUD_API_KEY` | LlamaParse API key |
| `LLAMA_MODEL_NAME` | Ollama model name for LLM classification |
| `DOCUMENT_PATH` | Absolute path to the folder containing documents to ingest |
| `CHROMA_DB_PATH` | Path for ChromaDB (optional, not used in default pipeline) |

### 3. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 4. Run ingestion

```bash
python main.py
```

The pipeline scans `DOCUMENT_PATH`, processes each file, and upserts the resulting chunks and embeddings into the configured pgvector table. Duplicate documents are deduplicated via LlamaIndex's `UPSERTS` docstore strategy.

## Document Classification

Each document chunk is tagged with a `doc_type` metadata field. Classification runs in two stages:

1. **Rule-based** — fast keyword matching against a priority-ordered rule set
2. **LLM fallback** — the first 2,000 characters are sent to the local Ollama model if no rule matches

Supported types: `faq`, `policy`, `guide`, `contract`, `report`, `notice`, `invoice`, `generic`

## RAG Retrieval

`RagRetriever` wraps the pgvector index for async semantic search:

```python
from app.database.pgvector_service import PgVectorStoreService
from app.embeddings.embedding_service import EmbeddingService
from app.rag.retriever import RagRetriever

retriever = RagRetriever(
    vector_store_service=PgVectorStoreService(),
    embedding_service=EmbeddingService(),
    similarity_top_k=5,
)

results = await retriever.aretrieve("What is the notice period for lease termination?")
```

## Supported File Types

| Extension | Parser |
|---|---|
| `.pdf` | LlamaParse |
| `.docx` | LlamaParse |
| `.txt` | LlamaParse |
| `.csv` | LlamaParse |
