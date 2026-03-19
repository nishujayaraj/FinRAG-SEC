# FinRAG-SEC: Production-Grade Retrieval-Augmented Generation for SEC Financial Document Analysis

A production-ready RAG pipeline that enables natural language Q&A over SEC 10-K annual filings from Apple, Microsoft, Tesla, Google, and Amazon with semantic search, LLM-powered answer generation, custom evaluation metrics, and MLflow experiment tracking.

---

## What is this?

Financial analysts at firms like Goldman Sachs and Bloomberg spend hours manually reading through 100-page Securities and Exchange Commission(SEC) 10-K filings to answer questions like:

- *"What risks did Tesla mention about electric vehicles this year?"*
- *"How does Microsoft describe its cloud business?"*
- *"What changed in Apple's revenue mix from 2023 to 2025?"*

**FinRAG-SEC automates this** : ask any question in plain English and get a precise, cited answer drawn directly from the official SEC filings.

---

## Architecture

```
User Question
      │
      ▼
┌─────────────────────┐
│   Query Embedder    │  Converts question to 384-dim vector
│  (MiniLM-L6-v2)     │  using sentence-transformers
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Qdrant Vector DB  │  HNSW index searches 1,427 chunk vectors
│   (local mode)      │  Returns top-K most similar chunks
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Groq LLM          │  Llama-3.1-8b instant reads retrieved
│  (Llama 3.1 8B)     │  chunks and generates a grounded answer
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│   Answer +          │  Final answer with company, date,
│   Citations         │  and similarity scores cited
└─────────────────────┘
```

---

## Pipeline Stages

### Stage 1 — Data Ingestion
Fetches real 10-K filings directly from the **SEC EDGAR API** for 5 companies across 3 years (15 filings total). Each company is identified by its CIK number and filings are downloaded as HTML.

### Stage 2 — Parsing
Raw HTML is cleaned using **BeautifulSoup** — removing JavaScript, CSS, XBRL/XML accounting tags, and SEC boilerplate to extract clean human readable text.

### Stage 3 — Chunking
Each document (30,000–60,000 words) is split into **overlapping chunks** of 500 words with 100 words overlap. Overlap ensures important sentences at chunk boundaries are not missed.

```
[chunk 1: words 0-500  ]
[chunk 2: words 400-900]   ← 100 word overlap
[chunk 3: words 800-1300]  ← 100 word overlap
```

### Stage 4 — Embedding
Each chunk is converted into a **384-dimensional vector** using `sentence-transformers/all-MiniLM-L6-v2`. Vectors capture semantic meaning, not just keywords enabling meaning based search.

### Stage 5 — Vector Storage
All 1,427 chunk vectors are stored in **Qdrant** with their metadata (text, company, date). Qdrant builds an HNSW index that enables millisecond similarity search across all vectors.

### Stage 6 — Generation
Retrieved chunks are formatted into a **carefully engineered prompt** with source labels and sent to **Groq's Llama 3.1 8B** for answer generation. Temperature is set to 0.1 for factual precision.

### Stage 7 — Evaluation
Custom evaluation metrics measure system quality across 10 test questions:
- **Context Relevancy** — do retrieved chunks contain keywords from the question?
- **Faithfulness** — does the answer only use information from the chunks?
- **Answer Relevancy** — does the answer semantically match the question?

All results tracked with **MLflow** for experiment comparison.

---

## Dataset

| Company | Filing Years | Avg Words/Filing |
|---------|-------------|-----------------|
| Apple | 2023, 2024, 2025 | ~32,000 |
| Microsoft | 2023, 2024, 2025 | ~53,000 |
| Tesla | 2024, 2025, 2026 | ~62,000 |
| Google | 2024, 2025, 2026 | ~55,000 |
| Amazon | 2024, 2025, 2026 | ~44,000 |
| **Total** | **15 filings** | **1,427 chunks** |

---

## Evaluation Results

Evaluated on 10 hand-crafted financial analysis questions using custom metrics:

| Metric | Score |
|--------|-------|
| Context Relevancy | 0.6517 |
| Faithfulness | 0.7300 |
| Answer Relevancy | 0.7437 |
| Avg Retrieval Score | 0.6056 |
| **Overall Score** | **0.6828** |

**Grade: Good 👍**

All experiments tracked in MLflow — parameters, metrics, and per-question CSV artifacts logged per run.

---

## Sample Q&A Output

**Question:** *What risks does Tesla mention related to electric vehicles?*

```
ANSWER:
Based on Tesla's 2026 and 2024 SEC filings, Tesla mentions the following 
risks related to electric vehicles:

1. Perceptions about EV features, quality, safety, performance, and cost
2. Limited range on a single battery charge and access to charging facilities
3. Competition from plug-in hybrid and high fuel-economy ICE vehicles
4. Volatility in the cost of oil, gasoline, and energy
5. Government regulations and economic incentives
6. Concerns about Tesla's future viability

SOURCES USED:
  1. TESLA | 2026-01-29 | Similarity Score: 0.6635
  2. TESLA | 2024-01-29 | Similarity Score: 0.6452
  3. TESLA | 2024-01-29 | Similarity Score: 0.5257

Tokens used: 2267
```

---

## Project Structure

```
FinRAG-SEC/
├── src/
│   ├── ingestion/
│   │   ├── sec_downloader.py    # SEC EDGAR API client, fetches 10-K filings
│   │   └── parser.py            # HTML cleaner, extracts readable text
│   ├── chunking/
│   │   └── chunker.py           # Overlapping window chunker (500/100)
│   ├── embeddings/
│   │   └── embedder.py          # MiniLM-L6-v2 batch embedding pipeline
│   ├── retrieval/
│   │   └── vector_store.py      # Qdrant client, collection management, search
│   ├── generation/
│   │   └── generator.py         # Prompt engineering + Groq LLM inference
│   └── evaluation/
│       └── evaluator.py         # Custom metrics + MLflow experiment tracking
├── data/
│   ├── raw/                     # Downloaded SEC HTML filings
│   └── processed/               # Chunks JSON, embeddings JSON, eval CSV
├── config/                      # YAML configs
├── requirements.txt
└── .env.example
```

---

## Key Design Decisions

### Why overlapping chunks?
Important sentences often fall at chunk boundaries. A 100-word overlap between consecutive chunks ensures no critical information is split across two chunks and lost during retrieval.

### Why cosine similarity?
For text embeddings, cosine similarity measures the **angle** between vectors rather than their magnitude. This makes it robust to document length — a short chunk and a long chunk about the same topic score similarly.

### Why Groq over OpenAI?
Groq runs Llama 3.1 on custom LPU hardware at ~500 tokens/second — significantly faster than OpenAI for inference. For a financial Q&A system where latency matters, this is the practical production choice.

### Why custom evaluation metrics over RAGAS?
Custom metrics give full transparency into what is being measured and eliminate dependency on third-party APIs for evaluation. Context relevancy, faithfulness, and answer relevancy are implemented from scratch using keyword overlap and cosine similarity — the same techniques used in production evaluation pipelines.

### Why Qdrant over Pinecone/Weaviate?
Qdrant runs locally with zero cost in development, and the same client code points to a cloud instance in production — no code changes needed. Its HNSW index provides sub-millisecond search at scale.

---

## Research Questions

| Question | How we measure it |
|----------|-------------------|
| Does semantic search find relevant chunks? | Context relevancy score across 10 questions |
| Does the LLM stay grounded in the source? | Faithfulness score (keyword overlap with chunks) |
| Does the answer address the question asked? | Answer relevancy via cosine similarity |
| How fast is end-to-end retrieval + generation? | Tokens/sec via Groq, logged in MLflow |

---

## Quickstart

### 1. Clone and setup
```bash
git clone https://github.com/nishujayaraj/FinRAG-SEC
cd FinRAG-SEC
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Add API keys
```bash
cp .env.example .env
# Add your HUGGINGFACE_API_TOKEN and GROQ_API_KEY
```

### 3. Run the pipeline
```bash
# Download 15 SEC filings
python3 src/ingestion/sec_downloader.py

# Parse + chunk all filings
python3 src/chunking/chunker.py

# Embed all chunks (downloads ~80MB model first time)
python3 src/embeddings/embedder.py

# Upload to Qdrant vector DB
python3 src/retrieval/vector_store.py

# Test Q&A generation
python3 src/generation/generator.py

# Run evaluation + log to MLflow
python3 src/evaluation/evaluator.py
```

### 4. View MLflow results
```bash
mlflow ui
# Open http://localhost:5000
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| SEC Data | EDGAR REST API |
| HTML Parsing | BeautifulSoup4 |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | Qdrant (local mode) |
| LLM Inference | Groq (Llama-3.1-8b-instant) |
| Experiment Tracking | MLflow |
| Language | Python 3.13 |

---

## About

Built as a production-grade portfolio project demonstrating end-to-end RAG system design from raw government data ingestion to evaluated, cited LLM answers — with full MLOps instrumentation.

**Author:** Nischitha S Jayaraja | [LinkedIn](https://linkedin.com/in/nischithajayaraja) | [GitHub](https://github.com/nishujayaraj)
