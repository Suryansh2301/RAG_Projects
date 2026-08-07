"""
src/retriever.py
-----------------
Hybrid Ensemble (BM25 + Chroma) & Local Cross-Encoder re-ranking.
Optimized specifically for CPU-only, low-memory execution environments
(e.g. Render Free/Starter, 512MB RAM).

    User Query
        -> Dense search (Chroma, top 10)   \\
        -> Sparse search (BM25, top 10)     } EnsembleRetriever (RRF, 40% BM25 / 60% Dense)
        -> ~20 fused candidates
        -> Cross-Encoder Re-Ranker (cross-encoder/ms-marco-MiniLM-L-6-v2, running LOCALLY, ~80MB)
        -> Top N high-precision chunks

Everything is exposed through get_reranked_retriever(), which returns a
LangChain Runnable[str, list[Document]] -- so it composes with `|` inside
an LCEL chain exactly like any built-in retriever.
"""

import os
from typing import List

# Pre-emptively optimize CPU execution configuration before loading torch
# This prevents PyTorch from aggressively hoarding all virtual CPU cores
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from sentence_transformers import CrossEncoder

from config import (
    TOP_K_DENSE,
    TOP_K_SPARSE,
    RRF_WEIGHT_BM25,
    RRF_WEIGHT_DENSE,
    RERANK_MODEL_ID,
    TOP_N_RERANKED,
)
from src.ingestion import load_dense_index, load_sparse_index

# Global variable to cache the model across queries to avoid reload overhead
_LOCAL_CROSS_ENCODER = None


def get_hybrid_retriever() -> EnsembleRetriever:
    """
    Combines the dense (Chroma) and sparse (BM25) retrievers into a single
    EnsembleRetriever, which performs Reciprocal Rank Fusion internally at
    the configured 40% BM25 / 60% Dense weights.
    """
    dense_retriever = load_dense_index().as_retriever(search_kwargs={"k": TOP_K_DENSE})

    sparse_retriever = load_sparse_index()
    if sparse_retriever is None:
        raise RuntimeError("No BM25 index found. Ingest a document first.")
    sparse_retriever.k = TOP_K_SPARSE

    return EnsembleRetriever(
        retrievers=[sparse_retriever, dense_retriever],
        weights=[RRF_WEIGHT_BM25, RRF_WEIGHT_DENSE],
    )


def rerank_with_cross_encoder(
    query: str, candidates: List[Document], top_n: int = TOP_N_RERANKED
) -> List[Document]:
    """
    Scores each (query, chunk) pair with a local cross-encoder model.
    Runs completely offline, locked to CPU mode, cached across queries.
    """
    global _LOCAL_CROSS_ENCODER

    if not candidates:
        return []

    # Initialize and cache the model on the first request
    if _LOCAL_CROSS_ENCODER is None:
        _LOCAL_CROSS_ENCODER = CrossEncoder(RERANK_MODEL_ID, device="cpu")
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)

    pairs = [[query, doc.page_content] for doc in candidates]
    raw_scores = _LOCAL_CROSS_ENCODER.predict(pairs, show_progress_bar=False)
    scores = [float(s) for s in raw_scores]

    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in scored[:top_n]]


def get_reranked_retriever() -> RunnableLambda:
    """
    Returns a Runnable[str, list[Document]]: query in, top-N re-ranked
    chunks out. Because it's a RunnableLambda it can be piped with `|`
    directly inside the LCEL chain built in src/chain.py.
    """
    hybrid_retriever = get_hybrid_retriever()

    def _retrieve_and_rerank(query: str) -> List[Document]:
        candidates = hybrid_retriever.invoke(query)
        return rerank_with_cross_encoder(query, candidates)

    return RunnableLambda(_retrieve_and_rerank)


# --- HF Inference API version (kept for reference / paid-endpoint setups) --
# Use this if you switch RERANK_MODEL_ID to a larger model (e.g.
# BAAI/bge-reranker-v2-m3) served via a dedicated paid HF Inference
# Endpoint. Note: api-inference.huggingface.co is retired -- use
# router.huggingface.co/hf-inference instead.
#
# import requests
# from config import HF_TOKEN
#
# def _flatten_scores(raw_response):
#     scores = []
#     for item in raw_response:
#         if isinstance(item, list):
#             item = item[0]
#         scores.append(float(item["score"]))
#     return scores
#
# def rerank_with_cross_encoder(query, candidates, top_n=TOP_N_RERANKED):
#     if not candidates:
#         return []
#     pairs = [[query, doc.page_content] for doc in candidates]
#     api_url = f"https://router.huggingface.co/hf-inference/models/{RERANK_MODEL_ID}"
#     headers = {"Authorization": f"Bearer {HF_TOKEN}"}
#     try:
#         response = requests.post(api_url, headers=headers, json={"inputs": pairs}, timeout=30)
#         response.raise_for_status()
#         scores = _flatten_scores(response.json())
#     except Exception as exc:
#         raise RuntimeError(
#             f"Cross-encoder re-ranking call failed ({exc}). Check that "
#             f"'{RERANK_MODEL_ID}' is deployed on a paid HF Inference Endpoint."
#         ) from exc
#     scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
#     return [doc for doc, _score in scored[:top_n]]

