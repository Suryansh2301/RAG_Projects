"""
src/chain.py
------------
LCEL RAG chain & source citation formatter.
"""

from functools import lru_cache
from typing import List
import time
import tiktoken
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from config import (
    HF_TOKEN,
    LLM_MODEL_ID,
    LLM_MAX_NEW_TOKENS,
    LONG_DOC_TOKEN_THRESHOLD,
)
from src.retriever import get_reranked_retriever

_ENCODER = tiktoken.get_encoding("cl100k_base")


@lru_cache(maxsize=2)
def get_llm(temperature: float) -> ChatHuggingFace:
    """Cached Hugging Face chat LLM served via the HF Inference API (default: Qwen/Qwen3-8B)."""
    endpoint = HuggingFaceEndpoint(
        repo_id=LLM_MODEL_ID,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=temperature,
        max_new_tokens=LLM_MAX_NEW_TOKENS,
    )
    return ChatHuggingFace(llm=endpoint)


# --- Gemini version (kept for reference / easy rollback) -------------------
# from langchain_google_genai import ChatGoogleGenerativeAI
# from config import GOOGLE_API_KEY
#
# @lru_cache(maxsize=2)
# def get_llm(temperature: float) -> ChatGoogleGenerativeAI:
#     """Cached Google Gemini chat LLM (default: gemini-3.5-flash-lite or gemini-3.6-flash)."""
#     return ChatGoogleGenerativeAI(
#         model=LLM_MODEL_ID,
#         google_api_key=GOOGLE_API_KEY,
#         temperature=temperature,
#         max_output_tokens=LLM_MAX_NEW_TOKENS,
#         max_retries=2,
#     )


def format_docs_with_citations(docs: List[Document]):
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

- Base every statement on the retrieved context.
- You may combine information from multiple retrieved chunks to produce a complete answer.
- Do NOT use outside knowledge.
- Do NOT hallucinate.
- Do NOT invent facts.
- Do NOT invent citations.
- Answer confidently whenever the answer is supported by the retrieved context.
- Do NOT say "The context does not explicitly mention..." if the answer is clearly present.
- Only state that the information is unavailable if the retrieved context genuinely lacks the answer.
- Remove repeated information.
- Write naturally as if explaining the document.
- Use Markdown formatting.

Response Format

### Answer
(2–6 concise paragraphs)

### Key Points
- Bullet points if appropriate

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
    retriever = get_reranked_retriever()
    llm = get_llm(0.0)

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

- Do not use outside knowledge.
- Preserve important technical terms.
- Remove repetition.
- Merge similar ideas.
- Keep the author's intent.
- Do not hallucinate.
- Mention limitations if present.
- Mention future work if present.
- Use Markdown headings.
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

- Main idea
- Important technical concepts
- Important facts
- Important numbers
- Algorithms
- Experimental findings
- Conclusions

Do not omit important information.

Section:

{section}
""",
    ),
])

reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", SUMMARY_SYSTEM_PROMPT),
    (
        "human",
        """
You are combining individual section summaries into a single comprehensive report.

Use ONLY the information provided in the intermediate summaries.

Do not hallucinate.
Do not repeat information.
Merge duplicate ideas.
Write concise, professional Markdown.

Intermediate Summaries:
{summaries}

Generate the report using EXACTLY the following format.

# Executive Summary
Write 2–3 concise paragraphs (maximum 250 words).

# Key Contributions
- 4–6 bullet points

# Important Concepts
- 5–8 bullet points with short explanations

# Methodology
- 4–6 bullet points

# Key Findings
- 5–8 bullet points

# Advantages
- 3–5 bullet points

# Limitations
- 3–5 bullet points
- If not mentioned, write "Not specified in the document."

# Future Work
- 3–5 bullet points
- If not mentioned, write "Not specified in the document."

# Conclusion
Write one concise paragraph (maximum 120 words).
""",
    ),
])


def _route_summary(chunks: List[Document]) -> dict:
    llm = get_llm(0.1)
    full_text = "\n\n".join([doc.page_content for doc in chunks])
    total_tokens = len(_ENCODER.encode(full_text))

    if total_tokens <= LONG_DOC_TOKEN_THRESHOLD:
        stuff_chain = stuff_prompt | llm | StrOutputParser()
        summary = stuff_chain.invoke({"document": full_text})
        return {
            "strategy_used": "Stuffing (Single Pass)",
            "token_count": total_tokens,
            "summary": summary,
        }
    else:
        map_chain = map_prompt | llm | StrOutputParser()
        intermediate_summaries = []
        for chunk in chunks:
            out = map_chain.invoke({"section": chunk.page_content})
            intermediate_summaries.append(out)
            # Stay below the HF Inference API free-tier rate limit
            time.sleep(5)

        combined_summaries_text = "\n\n=== Section Summary ===\n\n".join(intermediate_summaries)
        reduce_chain = reduce_prompt | llm | StrOutputParser()
        final_summary = reduce_chain.invoke({"summaries": combined_summaries_text})

        return {
            "strategy_used": "Map-Reduce (Multi-Pass Chunk Processing)",
            "token_count": total_tokens,
            "summary": final_summary,
        }


def build_summary_chain():
    return RunnableLambda(_route_summary)