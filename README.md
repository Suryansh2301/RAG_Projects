# RAG Copilot — Project Documentation

## 1. Overview
RAG Copilot is a Streamlit app that lets a user upload a PDF and either:
- **Ask specific questions** about it (Q&A Copilot Mode), or
- **Get a full executive summary** of it (Executive Summarization Mode)

Both modes are built as **LCEL Runnables** (LangChain Expression Language) —
composed with the `|` pipe operator — on top of a **hybrid retrieval**
pipeline (dense + sparse search, fused with Reciprocal Rank Fusion) and
**cross-encoder re-ranking**. Embeddings, re-ranking, and generation all run
through the Hugging Face Inference API using a single access token.

## 2. File Structure
```
rag-copilot/
├── app.py                    # Streamlit UI — entry point & mode router
├── config.py                 # All tunable constants
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py           # Python package initializer
│   ├── ingestion.py          # PDF loading, character chunking & dual indexing
│   ├── retriever.py          # Hybrid Ensemble (BM25 + Chroma) & Cross-Encoder
│   └── chain.py               # LCEL RAG chain & source citation formatter
├── data/
│   ├── uploads/
│   └── vectorstore/
└── docs/
```

## 3. Requirements
| Package | Purpose |
|---|---|
| streamlit | Web UI |
| langchain | Main package (composability helpers) |
| langchain-core | Runnable/LCEL base abstractions, prompts, output parsers |
| langchain-classic | `EnsembleRetriever` (moved here post-LangChain-1.0 split) |
| langchain-community | `PyPDFLoader`, `BM25Retriever` |
| langchain-text-splitters | `RecursiveCharacterTextSplitter` |
| langchain-huggingface | `HuggingFaceEndpoint`, `ChatHuggingFace`, HF embeddings |
| langchain-google-genai | `ChatGoogleGenerativeAI` `GoogleGenerativeAIEmbeddings`, Gemini LLM |
| langchain-chroma | Dedicated Chroma vector store integration |
| sentence-transformers | Inference (re-ranker) |
| chromadb | Dense vector database |
| rank_bm25 | Sparse/lexical BM25 index |
| pypdf | PDF text extraction |
| tiktoken | Token counting for the Map-Reduce vs Stuffing decision |
| python-dotenv | Loads `.env` |

```bash
pip install -r requirements.txt
```

## 4. Hugging Face Models
| Role | Model | Access |
|---|---|---|
| Dense embeddings | `BAAI/bge-m3` | HF Inference API (`HuggingFaceEndpointEmbeddings`) |
| Cross-encoder re-ranker | `BAAI/bge-reranker-base` | HF Inference API (raw `InferenceClient.post`) |
| Generation LLM | `meta-llama/Meta-Llama-3.1-8B-Instruct` (swap: `Qwen/Qwen2.5-7B-Instruct`) | HF Inference API (`HuggingFaceEndpoint` + `ChatHuggingFace`) |
| Dense embedding | `gemini-embedding-2` | Gemini Inference API (`GoogleGenerativeAIEmbeddings`) |
| Generation LLM | `gemini-3.5-flash-lite` (swap: `gemini-3.6-flash`) | Gemini Inference API (`ChatGoogleGenerativeAI`) |

## 5. Setup
```bash
git clone <your-repo>
cd rag-copilot
python -m venv venv
source .venv/bin/activate.ps1        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # paste your HUGGINGFACEHUB_API_TOKEN
streamlit run app.py
```

## 6. How each file works

### `config.py`
Every tunable constant: chunk size/overlap, top-k values, RRF weights, model
IDs, token thresholds. Loads `.env` and fails fast if the HF token is missing.

### `src/ingestion.py` — PDF loading, character chunking & dual indexing
- `get_embedding_model()` — cached `BAAI/bge-m3` wrapper (HF Inference API).
- `load_and_split_pdf()` — `PyPDFLoader` reads pages, `RecursiveCharacterTextSplitter`
  (800 chars / 150 overlap) chunks them, stamping `source`/`page` metadata.
- `build_dense_index()` / `load_dense_index()` — write/reopen the persisted
  `langchain-chroma` collection.
- `build_sparse_index()` / `load_sparse_index()` — build/rebuild the BM25
  index, pickling the source chunks so it survives a restart.
- `ingest_pdf()` — the single entry point `app.py` calls.

### `src/retriever.py` — Hybrid Ensemble (BM25 + Chroma) & Cross-Encoder
- `get_hybrid_retriever()` — wraps the Chroma retriever (top 10) and BM25
  retriever (top 10) in `langchain_classic`'s `EnsembleRetriever`, which
  performs Reciprocal Rank Fusion at the configured 40% BM25 / 60% dense
  weights.
- `rerank_with_cross_encoder()` — sends the ~20 fused candidates as
  `(query, chunk)` pairs to `BAAI/bge-reranker-base` and keeps the top 3.
- `get_reranked_retriever()` — wraps the whole retrieve-then-rerank flow in
  a `RunnableLambda`, so it's a `Runnable[str, list[Document]]` that
  composes with `|` inside `chain.py` just like a built-in retriever.

### `src/chain.py` — LCEL RAG chain & source citation formatter
- `get_llm()` — cached ` ChatGoogleGenerativeAI` singleton (gemini-3.5-flash-lite by
  default, gemini-3.6-flash as a drop-in swap), shared by both chains.
- `format_docs_with_citations()` — the source citation formatter: turns
  retrieved chunks into a labeled context string plus a structured citation
  list (`filename` + `page`).
- `build_qa_chain()` — **Option A**, a true LCEL pipeline:
  `RunnableLambda(retrieve+format) | RunnableParallel(answer=prompt|llm|parser, citations=passthrough)`.
  Call `.invoke({"question": "..."})` → `{"answer": ..., "citations": [...]}`.
- `build_summary_chain()` — **Option B**, also LCEL: a `RunnableLambda`
  routes on token count to either a Stuffing chain (`prompt | llm | parser`,
  one call) or a Map-Reduce chain (`map_chain.batch(...)` over sections,
  then one `reduce_chain.invoke(...)`). Call `.invoke(chunks)` →
  `{"summary": ..., "strategy_used": ..., "token_count": ...}`.

### `app.py`
The mode router: file uploader → `ingest_pdf()`; radio button → Option A or
B; each option builds its chain with `build_qa_chain()` /
`build_summary_chain()` and calls `.invoke(...)`, then renders the result.

## 7. Design notes
- **LangChain 1.0 split**: `EnsembleRetriever` now lives in `langchain-classic`
  (not `langchain.retrievers` as in the 0.x days) — `src/retriever.py`
  imports it from there. `BM25Retriever` and `PyPDFLoader` remain in
  `langchain-community`. Chroma has its own dedicated `langchain-chroma`
  package now instead of living under `langchain-community.vectorstores`.
- **Re-ranker call** uses a raw `InferenceClient.post()`. If your HF plan
  lacks Inference API access to `bge-reranker-base`, swap in a local
  `sentence_transformers.CrossEncoder` inside `rerank_with_cross_encoder` —
  same function signature, nothing else changes.
- **Gated models**: Llama-3.1-8B-Instruct requires accepting Meta's license
  on Hugging Face first; Qwen2.5-7B-Instruct is ungated and works immediately.
- **Persistence**: delete `data/vectorstore/` to force a clean re-index.
