"""
src/ingestion.py
-----------------
PDF loading, character chunking & dual indexing.

    User Uploads PDF
        -> RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
        -> Dense index   : ChromaDB, embedded with Google Gemini Embedding API
        -> Sparse index  : BM25 (rank_bm25 under the hood)

Also owns the embedding-model singleton, since embeddings are only ever
needed at ingest time (to write the dense index) and at retrieval time
(to embed the query) -- both of which start from this module.
"""

import os
import pickle
import time  # <--- Added for sleep delays
from functools import lru_cache
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL_ID,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    UPLOAD_DIR,
    VECTORSTORE_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K_SPARSE,
)

BM25_STORE_PATH = os.path.join(VECTORSTORE_DIR, "bm25_docs.pkl")


@lru_cache(maxsize=1)
def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
    """Cached Google Gemini embedding model."""
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_ID,
        google_api_key=GOOGLE_API_KEY,  # Optional if GOOGLE_API_KEY env var is set
        dimensionality=1024,
    )

def load_and_split_pdf(file_path: str) -> List[Document]:
    """Reads a PDF page-by-page and character-chunks it, keeping source/page metadata."""
    loader = PyPDFLoader(file_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    filename = os.path.basename(file_path)
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1  # 1-indexed for humans

    return chunks


def build_dense_index(chunks: List[Document]) -> Chroma:
    """
    Embeds chunks in rate-limited batches to prevent 429 RESOURCE_EXHAUSTED errors
    and writes them into a persisted ChromaDB collection.
    """
    if not chunks:
        raise ValueError("The chunk list is empty. Nothing to index.")

    BATCH_SIZE = 50       # Safely below the 100 requests per minute limit
    DELAY_SECONDS = 60    # Wait 1 minute between API calls
    
    embeddings_model = get_embedding_model()
    
    # 1. Initialize Chroma with the first batch
    first_batch = chunks[0:BATCH_SIZE]
    print(f"Initializing Chroma collection '{CHROMA_COLLECTION_NAME}' with the first batch ({len(first_batch)} chunks)...")
    
    db = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings_model,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )

    # 2. Add remaining batches sequentially with time delays
    if len(chunks) > BATCH_SIZE:
        print(f"First batch completed. Sleeping for {DELAY_SECONDS} seconds to respect API rate limits...")
        time.sleep(DELAY_SECONDS)

    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        
        print(f"Adding batch {current_batch_num} ({len(batch)} chunks) to Chroma...")
        db.add_documents(documents=batch)
        
        # Only sleep if there are more chunks left to process
        if i + BATCH_SIZE < len(chunks):
            print(f"Batch {current_batch_num} added. Sleeping for {DELAY_SECONDS} seconds...")
            time.sleep(DELAY_SECONDS)

    print("Chroma dense index built successfully!")
    return db


def build_sparse_index(chunks: List[Document]) -> BM25Retriever:
    """Builds a BM25 index and pickles the source chunks so it can be rebuilt without re-embedding."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    with open(BM25_STORE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K_SPARSE
    return bm25


def load_dense_index() -> Chroma:
    """Reopens the persisted Chroma collection (no re-embedding needed)."""
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=VECTORSTORE_DIR,
    )


def load_sparse_index() -> Optional[BM25Retriever]:
    """Rebuilds the BM25 retriever from the pickled chunk list, if one exists."""
    if not os.path.exists(BM25_STORE_PATH):
        return None
    with open(BM25_STORE_PATH, "rb") as f:
        chunks = pickle.load(f)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K_SPARSE
    return bm25


def ingest_pdf(uploaded_file_bytes: bytes, original_filename: str) -> List[Document]:
    """
    Single entry point app.py calls: save the upload, chunk it, and build
    both the dense and sparse indexes.
    """
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, original_filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_bytes)

    chunks = load_and_split_pdf(file_path)
    build_dense_index(chunks)
    build_sparse_index(chunks)
    return chunks

















# """
# src/ingestion.py
# -----------------
# PDF loading, character chunking & dual indexing.

#     User Uploads PDF
#         -> RecursiveCharacterTextSplitter (chunk_size=800, overlap=150)
#         -> Dense index   : ChromaDB, embedded with BAAI/bge-m3 (HF Inference API)
#         -> Sparse index  : BM25 (rank_bm25 under the hood)

# Also owns the embedding-model singleton, since embeddings are only ever
# needed at ingest time (to write the dense index) and at retrieval time
# (to embed the query) -- both of which start from this module.
# """

# import os
# import pickle
# from functools import lru_cache
# from typing import List, Optional

# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.retrievers import BM25Retriever
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document

# from config import (
#     GOOGLE_API_KEY,
#     EMBEDDING_MODEL_ID,
#     CHUNK_SIZE,
#     CHUNK_OVERLAP,
#     UPLOAD_DIR,
#     VECTORSTORE_DIR,
#     CHROMA_COLLECTION_NAME,
#     TOP_K_SPARSE,
# )

# BM25_STORE_PATH = os.path.join(VECTORSTORE_DIR, "bm25_docs.pkl")



# @lru_cache(maxsize=1)
# def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
#     """Cached Google Gemini embedding model."""
#     return GoogleGenerativeAIEmbeddings(
#         model=EMBEDDING_MODEL_ID,
#         google_api_key=GOOGLE_API_KEY,  # Optional if GOOGLE_API_KEY env var is set
#     )

# def load_and_split_pdf(file_path: str) -> List[Document]:
#     """Reads a PDF page-by-page and character-chunks it, keeping source/page metadata."""
#     loader = PyPDFLoader(file_path)
#     pages = loader.load()

#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=CHUNK_SIZE,
#         chunk_overlap=CHUNK_OVERLAP,
#         separators=["\n\n", "\n", ". ", " ", ""],
#     )
#     chunks = splitter.split_documents(pages)

#     filename = os.path.basename(file_path)
#     for chunk in chunks:
#         chunk.metadata["source"] = filename
#         chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1  # 1-indexed for humans

#     return chunks


# def build_dense_index(chunks: List[Document]) -> Chroma:
#     """Embeds every chunk with bge-m3 and writes them into a persisted ChromaDB collection."""
#     return Chroma.from_documents(
#         documents=chunks,
#         embedding=get_embedding_model(),
#         collection_name=CHROMA_COLLECTION_NAME,
#         persist_directory=VECTORSTORE_DIR,  # chromadb's PersistentClient auto-persists
#     )


# def build_sparse_index(chunks: List[Document]) -> BM25Retriever:
#     """Builds a BM25 index and pickles the source chunks so it can be rebuilt without re-embedding."""
#     os.makedirs(VECTORSTORE_DIR, exist_ok=True)
#     with open(BM25_STORE_PATH, "wb") as f:
#         pickle.dump(chunks, f)

#     bm25 = BM25Retriever.from_documents(chunks)
#     bm25.k = TOP_K_SPARSE
#     return bm25


# def load_dense_index() -> Chroma:
#     """Reopens the persisted Chroma collection (no re-embedding needed)."""
#     return Chroma(
#         collection_name=CHROMA_COLLECTION_NAME,
#         embedding_function=get_embedding_model(),
#         persist_directory=VECTORSTORE_DIR,
#     )


# def load_sparse_index() -> Optional[BM25Retriever]:
#     """Rebuilds the BM25 retriever from the pickled chunk list, if one exists."""
#     if not os.path.exists(BM25_STORE_PATH):
#         return None
#     with open(BM25_STORE_PATH, "rb") as f:
#         chunks = pickle.load(f)
#     bm25 = BM25Retriever.from_documents(chunks)
#     bm25.k = TOP_K_SPARSE
#     return bm25


# def ingest_pdf(uploaded_file_bytes: bytes, original_filename: str) -> List[Document]:
#     """
#     Single entry point app.py calls: save the upload, chunk it, and build
#     both the dense and sparse indexes.
#     """
#     os.makedirs(UPLOAD_DIR, exist_ok=True)
#     file_path = os.path.join(UPLOAD_DIR, original_filename)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file_bytes)

#     chunks = load_and_split_pdf(file_path)
#     build_dense_index(chunks)
#     build_sparse_index(chunks)
#     return chunks
