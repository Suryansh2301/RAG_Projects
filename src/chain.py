"""
src/chain.py
------------
LCEL RAG chain & source citation formatter.

Owns the LLM singleton plus THREE LangChain Expression Language (LCEL)
Runnables, all built with the `|` pipe operator:

1. build_qa_chain()      -> Option A: retrieve -> format+cite -> prompt -> LLM
2. build_summary_chain() -> Option B: token-length route -> Stuffing | Map-Reduce
3. get_llm()             -> shared ChatGoogleGenerativeAI singleton used by both

Everything here is a Runnable, so app.py only ever calls `.invoke(...)`.
"""

from functools import lru_cache
from typing import List

import tiktoken
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    GOOGLE_API_KEY,
    LLM_MODEL_ID,
    LLM_TEMPERATURE,
    LLM_MAX_NEW_TOKENS,
    LONG_DOC_TOKEN_THRESHOLD,
)
from src.retriever import get_reranked_retriever

_ENCODER = tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """Cached Google Gemini chat LLM (default: gemini-3.5-flash-lite or gemini-3.6-flash)."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL_ID,              
        google_api_key=GOOGLE_API_KEY,    
        temperature=LLM_TEMPERATURE,
        max_output_tokens=LLM_MAX_NEW_TOKENS,  
        max_retries=2,
    )


def format_docs_with_citations(docs: List[Document]):
    """
    Source citation formatter: turns retrieved chunks into a single
    prompt-ready context string, plus a structured citation list the UI
    renders as "Source: filename.pdf, page 4".
    """
    blocks, citations = [], []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(f"[Chunk {i} | Source: {source} | Page: {page}]\n{doc.page_content}")
        citations.append({"chunk": i, "source": source, "page": page})
    return "\n\n".join(blocks), citations


# --- Option A: Q&A Copilot Prompts & Configuration ---
QA_SYSTEM_PROMPT = """
You are an expert AI Research Assistant specializing in question answering over uploaded documents.

Your task is to answer the user's question ONLY using the retrieved document context.

Instructions:

• Base every statement on the retrieved context.
• You may combine information from multiple retrieved chunks to produce a complete answer.
• Do NOT use outside knowledge.
• Do NOT hallucinate.
• Do NOT invent facts.
• Do NOT invent citations.
• Answer confidently whenever the answer is supported by the retrieved context.
• Do NOT say "The context does not explicitly mention..." if the answer is clearly present.
• Only state that the information is unavailable if the retrieved context genuinely lacks the answer.
• Remove repeated information.
• Write naturally as if explaining the document.
• Use Markdown formatting.

Response Format

### Answer
(2–6 concise paragraphs)

### Key Points
• Bullet points if appropriate

### Sources
(Page numbers only)
"""

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", QA_SYSTEM_PROMPT),
    (
        "human",
        """
Context:

{context}

Question:

{question}

Answer using ONLY the retrieved context.
If multiple chunks contain complementary information, combine them naturally.
ask follow-up questions.
""",
    ),
])


def build_qa_chain():
    """
    Builds the full Option A Runnable using LCEL:
        retrieve+rerank -> format+cite -> RunnableParallel(answer=prompt|llm|parser, citations=passthrough)
    """
    retriever = get_reranked_retriever()
    llm = get_llm()

    def _retrieve_and_format(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        context, citations = format_docs_with_citations(docs)
        return {"context": context, "question": inputs["question"], "citations": citations}

    answer_chain = qa_prompt | llm | StrOutputParser()

    chain = (
        RunnableLambda(_retrieve_and_format)
        | RunnableParallel(
            answer=answer_chain,
            citations=RunnableLambda(lambda x: x["citations"]),
        )
    )
    return chain


# --- Option B: Executive Summarization Prompts & Configuration ---
SUMMARY_SYSTEM_PROMPT = """
You are an expert research paper and technical document summarizer.

Summarize ONLY the provided document.

Instructions:

• Do not use outside knowledge.
• Preserve important technical terms.
• Remove repetition.
• Merge similar ideas.
• Keep the author's intent.
• Do not hallucinate.
• Mention limitations if present.
• Mention future work if present.
• Use Markdown headings.
"""

stuff_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM_PROMPT),
    (
        "human",
        """
Document:

{document}

Generate:

# Executive Summary
(4–6 paragraphs)

# Key Contributions

# Important Concepts

# Methodology

# Key Findings

# Advantages

# Limitations (if available)

# Future Work (if available)

# Conclusion

Keep the summary faithful to the document.
""",
    ),
])

map_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM_PROMPT),
    (
        "human",
        """
Summarize the following document section.

Include:

• Main idea
• Important technical concepts
• Important facts
• Important numbers
• Algorithms
• Experimental findings
• Conclusions

Do not omit important information.

Section:

{section}
""",
    ),
])

# Re-constructed the broken prompt from the cut-off
reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM_PROMPT),
    (
        "human",
        """
You are combining individual section summaries of a larger document into a single cohesive Executive Summary.

Consolidate the following intermediate summaries:

{summaries}

Generate a comprehensive final output using the layout requested below. Ensure smooth transitions.

# Executive Summary
(4–6 paragraphs summarizing the entire document context)

# Key Contributions

# Important Concepts

# Methodology

# Key Findings

# Advantages

# Limitations (if available)

# Future Work (if available)

# Conclusion
""",
    ),
])


def _route_summary(chunks: List[Document]) -> dict:
    """
    Dynamic routing logic based on total context token counts.
    Routes between simple Stuffing and Map-Reduce workflows.
    """
    llm = get_llm()
    full_text = "\n\n".join([doc.page_content for doc in chunks])
    total_tokens = len(_ENCODER.encode(full_text))

    # --- Strategy 1: Stuffing (For shorter documents) ---
    if total_tokens <= LONG_DOC_TOKEN_THRESHOLD:
        stuff_chain = stuff_prompt | llm | StrOutputParser()
        summary = stuff_chain.invoke({"document": full_text})
        return {
            "strategy_used": "Stuffing (Single Pass)",
            "token_count": total_tokens,
            "summary": summary,
        }

    # --- Strategy 2: Map-Reduce (For longer documents exceeding threshold) ---
    else:
        # Step 1: Map (Summarize every individual chunk)
        map_chain = map_prompt | llm | StrOutputParser()
        intermediate_summaries = []
        for chunk in chunks:
            out = map_chain.invoke({"section": chunk.page_content})
            intermediate_summaries.append(out)

        # Step 2: Reduce (Synthesize intermediate summaries into one)
        combined_summaries_text = "\n\n=== Section Summary ===\n\n".join(intermediate_summaries)
        reduce_chain = reduce_prompt | llm | StrOutputParser()
        final_summary = reduce_chain.invoke({"summaries": combined_summaries_text})

        return {
            "strategy_used": "Map-Reduce (Multi-Pass Chunk Processing)",
            "token_count": total_tokens,
            "summary": final_summary,
        }


def build_summary_chain():
    """
    Builds the Option B Runnable using an LCEL RunnableLambda routing engine.
    Usage: build_summary_chain().invoke(chunks)
        -> {"strategy_used": "...", "token_count": 1234, "summary": "..."}
    """
    return RunnableLambda(_route_summary)
