# DocMind-RAG-Powered-Document-Intelligence-Chatbot


DocMind is a Retrieval-Augmented Generation (RAG) chatbot that lets you chat with your documents. It searches your local files first, falls back to web search when needed, and remembers your conversation throughout the session.

Built entirely with open-source tools. Runs locally. Zero API costs for LLM inference.

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Llama](https://img.shields.io/badge/LLM-Llama%203.1-purple)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## How It Works

DocMind uses a 3-tier knowledge architecture to answer questions:

```
User asks a question
        │
        ▼
┌─────────────────────┐
│  1. Document Search  │ ── ChromaDB vector search (cosine similarity)
│     Distance < 0.8?  │
└────────┬────────────┘
         │ No
         ▼
┌─────────────────────┐
│  2. Web Search       │ ── Tavily API (real-time web results)
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  3. Conv. Memory     │ ── Remembers past questions in the session
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Llama 3.1 via       │ ── Generates answer grounded in retrieved context
│  Ollama (local)      │
└─────────────────────┘
```

### RAG Pipeline

**Indexing (one-time):**
```
Documents (.txt, .md) → Chunk (300 words, 50-word overlap) → Embed (all-MiniLM-L6-v2) → Store (ChromaDB)
```

**Querying (every question):**
```
Question → Embed → Semantic search in ChromaDB → Top 3 chunks → LLM generates answer
```

## Features

- **Semantic Search** — finds relevant content by meaning, not just keywords
- **Web Search Fallback** — automatically searches the web when documents don't have the answer
- **Conversation Memory** — remembers previous questions and answers within a session
- **Hallucination Reduction** — model answers only from retrieved context, with a tunable relevance threshold
- **Fully Local LLM** — runs Llama 3.1 via Ollama, no data leaves your machine
- **Zero LLM Cost** — no API keys needed for inference

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Llama 3.1 (8B) via Ollama |
| Embeddings | all-MiniLM-L6-v2 (384-dim) |
| Vector Database | ChromaDB (persistent, cosine similarity) |
| Web Search | Tavily API |
| Language | Python 3.11+ |

## Project Structure

```
docmind/
├── docs/                          # Your documents go here
│   └── sample.txt                 # Example document
├── chroma_db/                     # Vector database (auto-created)
├── step1_index_docs.py            # Indexes documents into ChromaDB
├── step3_rag_with_memory_and_search.py  # Main chatbot with all features
└── README.md
```

## Quick Start

### Prerequisites

- [Python 3.11+](https://python.org/downloads)
- [Ollama](https://ollama.com/download)

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/docmind.git
cd docmind

# 2. Pull the LLM model (~4.7GB download)
ollama pull llama3.1

# 3. Install Python dependencies
pip3 install ollama chromadb sentence-transformers tavily-python

# 4. Add your Tavily API key (free at tavily.com)
#    Edit step3_rag_with_memory_and_search.py and replace:
#    tvly-YOUR-KEY-HERE
```

### Usage

```bash
# Step 1: Add documents to the docs/ folder
#         Supports .txt and .md files

# Step 2: Index your documents (run once, or when docs change)
python3 step1_index_docs.py

# Step 3: Start chatting
python3 step3_rag_with_memory_and_search.py
```

### Example Session

```
You: what is an AI agent?
  📎 Searching your documents...
  📎 Best match distance: 0.6140
  ✓  Found relevant chunks in your documents
Assistant (📎): An AI agent is a system that uses a large language model
combined with tools to take actions autonomously...

You: what was my last question?
  💬 This is about our conversation — using memory
Assistant (💬): You asked about what an AI agent is.

You: today's IPL match results
  📎 Searching your documents...
  📎 Best match distance: 0.9542
  ✗  Documents don't seem relevant (distance 0.95)
  🌐 Searching the web instead...
  🌐 Got 3 web results
Assistant (🌐): Based on web search results, today's IPL matches include...
```

## Architecture Deep Dive

### Document Chunking

Documents are split into 300-word chunks with 50-word overlap to prevent sentence splitting at boundaries:

```
Chunk 1: words 1-300
              ├── 50 word overlap ──┤
Chunk 2:           words 251-550
                        ├── 50 word overlap ──┤
Chunk 3:                     words 501-800
```

### Embedding & Similarity

Each chunk is converted into a 384-dimensional vector using the all-MiniLM-L6-v2 model. At query time, the user's question is embedded with the same model, and ChromaDB finds the closest vectors using cosine distance:

- **Distance < 0.8** → Documents are relevant, use them
- **Distance ≥ 0.8** → Documents aren't relevant, fall back to web search

### Query Routing

Intent classification determines the knowledge source:

```python
# Memory keywords trigger conversation recall
memory_keywords = ["last question", "you said", "i asked", "remember", ...]

# Decision flow:
if memory_question → use conversation history
elif doc_distance < 0.8 → use document chunks
else → use web search
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 300 | Words per chunk |
| `CHUNK_OVERLAP` | 50 | Overlapping words between chunks |
| `n_results` | 3 | Number of chunks retrieved per query |
| `best_distance` threshold | 0.8 | Cosine distance cutoff for document relevance |
| Embedding model | all-MiniLM-L6-v2 | 384-dim, fast, good quality |
| LLM model | llama3.1 | 8B parameters, supports tool calling |

## Possible Improvements

- [ ] PDF and DOCX support for document ingestion
- [ ] Streaming responses for better UX
- [ ] Hybrid search (vector + keyword) for better retrieval
- [ ] Reranking with a cross-encoder for improved precision
- [ ] Web UI with Streamlit or Gradio
- [ ] Persistent conversation memory across sessions using ChromaDB

## License

MIT
