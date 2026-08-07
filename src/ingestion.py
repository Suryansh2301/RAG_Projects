import os
import pickle
import time
from functools import lru_cache
from typing import List, Optional

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
import chromadb  # add this import at the top


from config import (
    HF_TOKEN,
    EMBEDDING_MODEL_ID,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    UPLOAD_DIR,
    VECTORSTORE_DIR,
    CHROMA_COLLECTION_NAME,
    TOP_K_SPARSE,
)

_BM25_CACHE: Optional[BM25Retriever] = None
_DENSE_CACHE: Optional[Chroma] = None

BM25_STORE_PATH = os.path.join(VECTORSTORE_DIR, "bm25_docs.pkl")


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEndpointEmbeddings:
    """Cached BAAI/bge-m3 embedding model served via the HF Inference API."""
    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
    )


# --- Gemini version (kept for reference / easy rollback) -------------------
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from config import GOOGLE_API_KEY
#
# @lru_cache(maxsize=1)
# def get_embedding_model() -> GoogleGenerativeAIEmbeddings:
#     """Cached Google Gemini embedding model."""
#     return GoogleGenerativeAIEmbeddings(
#         model=EMBEDDING_MODEL_ID,
#         google_api_key=GOOGLE_API_KEY,
#         dimensionality=1024,
#     )


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
    Drops any existing collection (avoids stale-dimension errors when the
    embedding model changes, and avoids mixing vectors from previous
    uploads), then embeds chunks in rate-limited batches and writes them
    into a fresh persisted ChromaDB collection.
    """
    global _DENSE_CACHE

    if not chunks:
        raise ValueError("The chunk list is empty. Nothing to index.")

    # Drop the old collection so a new embedding model's dimensionality
    # (or a new document's vectors) never collides with a stale one.
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    try:
        client.delete_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet - fine

    BATCH_SIZE = 80
    DELAY_SECONDS = 60

    embeddings_model = get_embedding_model()

    first_batch = chunks[0:BATCH_SIZE]
    print(f"Initializing Chroma collection '{CHROMA_COLLECTION_NAME}' with the first batch ({len(first_batch)} chunks)...")

    db = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings_model,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
    )

    if len(chunks) > BATCH_SIZE:
        print(f"First batch completed. Sleeping for {DELAY_SECONDS} seconds to respect API rate limits...")
        time.sleep(DELAY_SECONDS)

    for i in range(BATCH_SIZE, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]
        current_batch_num = (i // BATCH_SIZE) + 1
        print(f"Adding batch {current_batch_num} ({len(batch)} chunks) to Chroma...")
        db.add_documents(documents=batch)
        if i + BATCH_SIZE < len(chunks):
            print(f"Batch {current_batch_num} added. Sleeping for {DELAY_SECONDS} seconds...")
            time.sleep(DELAY_SECONDS)

    print("Chroma dense index built successfully!")
    _DENSE_CACHE = db
    return db


def build_sparse_index(chunks: List[Document]) -> BM25Retriever:
    """Builds a BM25 index, pickles the source chunks, and caches the retriever in memory."""
    global _BM25_CACHE

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    with open(BM25_STORE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K_SPARSE
    _BM25_CACHE = bm25         # <-- cache it so load_sparse_index() reuses this object
    return bm25


def load_dense_index() -> Chroma:
    """Returns the cached Chroma client if one exists; otherwise opens the persisted collection."""
    global _DENSE_CACHE
    if _DENSE_CACHE is not None:
        return _DENSE_CACHE
    _DENSE_CACHE = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=get_embedding_model(),
        persist_directory=VECTORSTORE_DIR,
    )
    return _DENSE_CACHE


def load_sparse_index() -> Optional[BM25Retriever]:
    """Returns the cached BM25 retriever if one exists; otherwise rebuilds it from the pickle."""
    global _BM25_CACHE
    if _BM25_CACHE is not None:
        return _BM25_CACHE
    if not os.path.exists(BM25_STORE_PATH):
        return None
    with open(BM25_STORE_PATH, "rb") as f:
        chunks = pickle.load(f)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K_SPARSE
    _BM25_CACHE = bm25
    return bm25



def ingest_pdf(uploaded_file_bytes: bytes, original_filename: str) -> List[Document]:
    """
    Single entry point app.py calls: save the upload, chunk it, and build
    both the dense and sparse indexes.
    """
    global _DENSE_CACHE, _BM25_CACHE
    _DENSE_CACHE = None
    _BM25_CACHE = None

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, original_filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file_bytes)

    chunks = load_and_split_pdf(file_path)
    build_dense_index(chunks)
    build_sparse_index(chunks)
    return chunks