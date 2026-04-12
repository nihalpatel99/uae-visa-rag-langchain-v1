# UAE Visa Assistant

An AI-powered chatbot that answers questions about UAE visa policies, requirements, and procedures using Retrieval-Augmented Generation (RAG) with LangChain.

## Overview

This application uses an LLM agent backed by a ChromaDB vector store indexed from the official UAE Services Guide. Users can ask natural language questions about visas for residents, investors, and visitors, and receive contextual answers with cited source documents.

**Example questions:**
- What visa do I need to work in Dubai?
- How long can tourists stay in the UAE?
- Can investors get long-term residency?
- What documents are required for a family visa?

## Architecture

```
User Query
    │
    ▼
Streamlit UI (main.py)
    │
    ▼
LangChain Agent (OpenAI LLM)
    │
    ▼
ChromaDB Vector Store  ◄──  rag.py (indexing script)
    │                            │
    ▼                            ▼
Retrieved Chunks          UAE Services Guide PDF
    │
    ▼
LLM Response
```

- **Frontend**: Streamlit with a custom dark/gold UAE-themed UI
- **LLM**: OpenAI ChatGPT (gpt-4o, gpt-4o-mini, gpt-4-turbo, or gpt-3.5-turbo)
- **Embeddings**: OpenAI text-embedding models (runtime) / Ollama mxbai-embed-large (indexing)
- **Vector Store**: ChromaDB (persistent, stored in `uae_visa_index/`)
- **Framework**: LangChain agent with tool-use pattern for retrieval

## Prerequisites

- Python 3.13
- An [OpenAI API key](https://platform.openai.com/api-keys)
- [Ollama](https://ollama.com) with `mxbai-embed-large` model _(only needed to re-index the PDF)_

## Setup

### 1. Clone and enter the repository

```bash
git clone <repo-url>
cd uae-visa-rag-langchain-v1
```

### 2. Create a virtual environment and install dependencies

Using `uv` (recommended):

```bash
uv sync
```

Or using `pip`:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. (Optional) Re-index the source PDF

The `uae_visa_index/` vector store is already included. Run this only if you want to rebuild it or swap in a different PDF:

```bash
# Requires Ollama running with mxbai-embed-large
ollama pull mxbai-embed-large
python rag.py
```

### 4. Run the application

```bash
streamlit run main.py
```

The app opens at `http://localhost:8501`.

## Usage

1. Open the app in your browser.
2. In the **sidebar**, enter your OpenAI API key.
3. Select your preferred **Chat Model** and **Embedding Model**.
4. Click **Initialize Agent**.
5. Type your visa question in the chat input and press Enter.

The agent will show its reasoning steps and the source document chunks it used to formulate the answer.

## Project Structure

```
uae-visa-rag-langchain-v1/
├── main.py                          # Streamlit chatbot application
├── rag.py                           # PDF indexing script (builds vector store)
├── دليل-الخدمات-إصدار-4.2-2024EN.pdf  # Source: UAE Services Guide v4.2 (2024)
├── uae_visa_index/                  # ChromaDB persistent vector store
│   └── chroma.sqlite3
├── pyproject.toml                   # Project metadata & dependencies (uv)
├── requirements.txt                 # Pip-compatible dependency list
├── uv.lock                          # Locked dependency versions
├── .env                             # Environment variables (optional)
└── .python-version                  # Python version pin (3.13)
```

## Configuration

All settings are configurable from the Streamlit sidebar at runtime. No `.env` file is required.

| Setting | Options | Default |
|---|---|---|
| OpenAI API Key | Your `sk-...` key | — |
| Chat Model | `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-3.5-turbo` | `gpt-4o` |
| Embedding Model | `text-embedding-3-small`, `text-embedding-3-large`, `text-embedding-ada-002` | `text-embedding-3-small` |
| Chroma Directory | Path to vector store | `uae_visa_index` |

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain` / `langchain-core` | LLM agent orchestration |
| `langchain-community` | Document loaders, vector store integrations |
| `langchain-ollama` | Ollama embeddings (used during indexing) |
| `chromadb` | Persistent vector database |
| `pymupdf` | PDF parsing |
| `python-dotenv` | Environment variable loading |

## Data Source

The knowledge base is built from the **UAE Services Guide, Edition 4.2 (2024)** — the official government reference for visa and residency services. Documents are split into 500-character chunks with 100-character overlap before indexing.
