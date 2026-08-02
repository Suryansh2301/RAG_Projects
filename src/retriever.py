"""
src/retriever.py
-----------------
Hybrid Ensemble (BM25 + Chroma) & Local Cross-Encoder re-ranking.
Optimized specifically for CPU-only execution environments.

    User Query
        -> Dense search (Chroma, top 10)   \\
        -> Sparse search (BM25, top 10)     } EnsembleRetriever (RRF, 40% BM25 / 60% Dense)
        -> ~20 fused candidates
        -> Cross-Encoder Re-Ranker (BAAI/bge-reranker-base, running permanently LOCALLY)
        -> Top 3 high-precision chunks

Everything is exposed through get_reranked_retriever(), which returns a
LangChain Runnable[str, list[Document]] -- so it composes with `|` inside
an LCEL chain exactly like any built-in retriever.
"""

import os
from typing import List

# Pre-emptively optimize CPU execution configuration before loading torch
# This prevents PyTorch from aggressively hoarding 100% of all virtual CPU cores
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
    Scores each (query, chunk) pair with a local BAAI/bge-reranker-base
    cross-encoder model. Runs completely offline, locked to CPU mode.
    """
    global _LOCAL_CROSS_ENCODER

    if not candidates:
        return []

    # Initialize and cache the model on the first request
    if _LOCAL_CROSS_ENCODER is None:
        # Explicitly pass device='cpu' to skip GPU validation checks
        _LOCAL_CROSS_ENCODER = CrossEncoder(RERANK_MODEL_ID, device="cpu")
        
        # Configure inter-op threads directly inside the active torch runtime
        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)

    # Format text pairs as a standard list of lists for sentence_transformers
    pairs = [[query, doc.page_content] for doc in candidates]
    
    # Calculate inference scores locally on the CPU
    # disabled progress bar to keep clean production server application logs
    raw_scores = _LOCAL_CROSS_ENCODER.predict(pairs, show_progress_bar=False)
    scores = [float(s) for s in raw_scores]

    # Map scores back to documents and sort by highest relevance
    scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _score in scored[:top_n]]


def get_reranked_retriever() -> RunnableLambda:
    """
    Returns a Runnable[str, list[Document]]: query in, top-3 re-ranked
    chunks out. Because it's a RunnableLambda it can be piped with `|`
    directly inside the LCEL chain built in src/chain.py.
    """
    hybrid_retriever = get_hybrid_retriever()

    def _retrieve_and_rerank(query: str) -> List[Document]:
        candidates = hybrid_retriever.invoke(query)  # ~20 fused candidates
        return rerank_with_cross_encoder(query, candidates)  # top 3

    return RunnableLambda(_retrieve_and_rerank)
















# """
# src/retriever.py
# -----------------
# Hybrid Ensemble (BM25 + Chroma) & Cross-Encoder re-ranking.

#     User Query
#         -> Dense search (Chroma, top 10)   \\
#         -> Sparse search (BM25, top 10)     } EnsembleRetriever (RRF, 40% BM25 / 60% Dense)
#         -> ~20 fused candidates
#         -> Cross-Encoder Re-Ranker (BAAI/bge-reranker-base, via HF Inference API)
#         -> Top 3 high-precision chunks

# Everything is exposed through get_reranked_retriever(), which returns a
# LangChain Runnable[str, list[Document]] -- so it composes with `|` inside
# an LCEL chain exactly like any built-in retriever.
# """

# import json
# from typing import List

# from huggingface_hub import InferenceClient
# from langchain_core.documents import Document
# from langchain_core.runnables import RunnableLambda
# from langchain_classic.retrievers.ensemble import EnsembleRetriever

# from config import (
#     HF_TOKEN,
#     TOP_K_DENSE,
#     TOP_K_SPARSE,
#     RRF_WEIGHT_BM25,
#     RRF_WEIGHT_DENSE,
#     RERANK_MODEL_ID,
#     TOP_N_RERANKED,
# )
# from src.ingestion import load_dense_index, load_sparse_index


# def get_hybrid_retriever() -> EnsembleRetriever:
#     """
#     Combines the dense (Chroma) and sparse (BM25) retrievers into a single
#     EnsembleRetriever, which performs Reciprocal Rank Fusion internally at
#     the configured 40% BM25 / 60% Dense weights.
#     """
#     dense_retriever = load_dense_index().as_retriever(search_kwargs={"k": TOP_K_DENSE})

#     sparse_retriever = load_sparse_index()
#     if sparse_retriever is None:
#         raise RuntimeError("No BM25 index found. Ingest a document first.")
#     sparse_retriever.k = TOP_K_SPARSE

#     return EnsembleRetriever(
#         retrievers=[sparse_retriever, dense_retriever],
#         weights=[RRF_WEIGHT_BM25, RRF_WEIGHT_DENSE],
#     )


# def _flatten_scores(raw_response) -> List[float]:
#     """Normalizes the HF Inference API JSON response into a flat score list."""
#     if isinstance(raw_response, (bytes, bytearray)):
#         raw_response = json.loads(raw_response)
#     scores = []
#     for item in raw_response:
#         if isinstance(item, list):
#             item = item[0]
#         scores.append(float(item["score"]))
#     return scores


# def rerank_with_cross_encoder(
#     query: str, candidates: List[Document], top_n: int = TOP_N_RERANKED
# ) -> List[Document]:
#     """
#     Scores each (query, chunk) pair with the BAAI/bge-reranker-base
#     cross-encoder on the HF Inference API and keeps only the top-N.
#     """
#     if not candidates:
#         return []

#     client = InferenceClient(model=RERANK_MODEL_ID, token=HF_TOKEN)
#     pairs = [[query, doc.page_content] for doc in candidates]

#     try:
#         raw = client.post(json={"inputs": pairs}, task="text-classification")
#         scores = _flatten_scores(raw)
#     except Exception as exc:
#         raise RuntimeError(
#             f"Cross-encoder re-ranking call failed ({exc}). Check your HF "
#             f"token has Inference API access to '{RERANK_MODEL_ID}', or "
#             "swap in a local sentence_transformers.CrossEncoder as a fallback."
#         ) from exc

#     scored = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
#     return [doc for doc, _score in scored[:top_n]]


# def get_reranked_retriever() -> RunnableLambda:
#     """
#     Returns a Runnable[str, list[Document]]: query in, top-3 re-ranked
#     chunks out. Because it's a RunnableLambda it can be piped with `|`
#     directly inside the LCEL chain built in src/chain.py.
#     """
#     hybrid_retriever = get_hybrid_retriever()

#     def _retrieve_and_rerank(query: str) -> List[Document]:
#         candidates = hybrid_retriever.invoke(query)  # ~20 fused candidates
#         return rerank_with_cross_encoder(query, candidates)  # top 3

#     return RunnableLambda(_retrieve_and_rerank)
