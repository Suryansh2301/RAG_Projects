"""
app.py
------
Streamlit UI - entry point & mode router.

    1. Upload a PDF -> src/ingestion.py: ingest_pdf()
    2. Pick a mode  -> Option A (Q&A Copilot) or Option B (Executive Summary)
    3. Option A     -> src/chain.py: build_qa_chain().invoke({"question": ...})
    4. Option B     -> src/chain.py: build_summary_chain().invoke(chunks)
"""

import streamlit as st

from src.ingestion import ingest_pdf
from src.chain import build_qa_chain, build_summary_chain

st.set_page_config(page_title="RAG Copilot", page_icon="📄", layout="wide")

st.title("📄 RAG Copilot")
st.caption(
    "Hybrid Ensemble (BM25 + Chroma) + Cross-Encoder re-ranking + "
    "an LCEL RAG chain on Hugging Face open LLMs"
)

# Session state
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# 1. Ingestion
st.header("1. Upload a document")
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.ingested:
    # Removed st.spinner here because src.ingestion uses its own dynamic st.status / st.progress bar
    chunks = ingest_pdf(uploaded_file.getvalue(), uploaded_file.name)
    st.session_state.chunks = chunks
    st.session_state.filename = uploaded_file.name
    st.session_state.ingested = True
    st.success(f"Indexed {len(chunks)} chunks from '{uploaded_file.name}'.")

# 2. Mode Router
if st.session_state.ingested:
    st.header("2. Choose a mode")
    mode = st.radio(
        "What do you want to do?",
        ["Option A: Specific Question (Q&A Copilot)", "Option B: Full Document Summary"],
    )

    # -- Option A: Q&A Copilot (LCEL RAG chain) -----------------------------
    if mode.startswith("Option A"):
        query = st.text_input("Ask a question about the document")
        if st.button("Get Answer") and query:
            with st.spinner("Generating"):
                qa_chain = build_qa_chain()
                result = qa_chain.invoke({"question": query})

            st.subheader("💡 Generated Answer")
            st.write(result["answer"])

            st.subheader("📎 Exact Page Citations")
            for c in result["citations"]:
                st.markdown(f"- **{c['source']}**, page {c['page']} (chunk #{c['chunk']})")

    # -- Option B: Executive Summarization (LCEL Map-Reduce / Stuffing) ----
    else:
        if st.button("Generate Executive Summary"):
            with st.spinner("Generating executive summary..."):
                summary_chain = build_summary_chain()
                result = summary_chain.invoke(st.session_state.chunks)

            st.info(
                f"Strategy used: **{result['strategy_used']}** "
                f"(document ≈ {result['token_count']} tokens)"
            )
            st.subheader("📋 Executive Summary + Key Bullet Points")
            st.write(result["summary"])

    st.divider()
    if st.button("Reset / upload a different document"):
        st.session_state.ingested = False
        st.session_state.chunks = None
        st.session_state.filename = None
        st.rerun()
else:
    st.info("Upload a PDF above to get started.")

