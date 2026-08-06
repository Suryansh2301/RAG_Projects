"""
config.py
---------
Central place for every tunable constant in the project. Every module reads
from here so there's exactly one place to change chunk sizes, model IDs, etc.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Hugging Face credentials
# ---------------------------------------------------------------------------
# HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# if not HF_TOKEN:
#     raise EnvironmentError(
#         "HUGGINGFACEHUB_API_TOKEN not found. Create a .env file "
#         "(copy .env.example) and paste your Hugging Face access token."
#     )

# google api key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY not found. Create a .env file "
        "(copy .env.example) and paste your Google API key."
    )

# ---------------------------------------------------------------------------
# Ingestion & Dual Indexing (src/ingestion.py)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 500 # or 800
CHUNK_OVERLAP = 75 # or 150

UPLOAD_DIR = "data/uploads"
VECTORSTORE_DIR = "data/vectorstore"
CHROMA_COLLECTION_NAME = "rag_copilot_docs"

# EMBEDDING_MODEL_ID = "BAAI/bge-m3"  # served via HF Inference API
EMBEDDING_MODEL_ID = "gemini-embedding-2"  # served via HF Inference API

# ---------------------------------------------------------------------------
# Hybrid Search & Re-Ranking (src/retriever.py)
# ---------------------------------------------------------------------------
TOP_K_DENSE = 4 # or 10
TOP_K_SPARSE = 4 # or 10
RRF_WEIGHT_BM25 = 0.4
RRF_WEIGHT_DENSE = 0.6
RERANK_MODEL_ID = "BAAI/bge-reranker-small"  # or "BAAI/bge-reranker-base"
TOP_N_RERANKED = 2 # or 3, but 2 is faster and still high-precision

# ---------------------------------------------------------------------------
# LCEL Chain & Generation (src/chain.py)
# ---------------------------------------------------------------------------
# LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct") # Option B: meta-llama/Meta-Llama-3.1-8B-Instruct
# # Alternative: "Qwen/Qwen2.5-7B-Instruct" (ungated)

# -------------------------------------------------------------------

LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "gemini-3.5-flash-lite") # Option B: gemini-3.6-flash


# LLM_TEMPERATURE = 0.2
LLM_MAX_NEW_TOKENS = 1024

# Executive Summarization: > threshold -> Map-Reduce, else -> Stuffing
LONG_DOC_TOKEN_THRESHOLD = 10_000
