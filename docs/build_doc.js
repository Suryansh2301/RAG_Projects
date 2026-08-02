const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, convertInchesToTwip
} = require("docx");

const FONT = "Calibri";
const MONO = "Courier New";
const ACCENT = "1F4E79";
const LIGHT_SHADE = "EDF2F7";
const BORDER_COLOR = "CBD5E0";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, alignment: AlignmentType.LEFT, spacing: { before: 320, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, alignment: AlignmentType.LEFT, spacing: { before: 240, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { after: 140 },
    children: [new TextRun({ text, font: FONT, size: 22, ...opts })],
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, alignment: AlignmentType.LEFT, spacing: { after: 60 } });
}

function code(text) {
  const lines = text.split("\n");
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: [9000],
    rows: [new TableRow({
      children: [new TableCell({
        width: { size: 9000, type: WidthType.DXA },
        shading: { type: ShadingType.CLEAR, fill: LIGHT_SHADE },
        margins: { top: 100, bottom: 100, left: 120, right: 120 },
        borders: {
          top: { style: BorderStyle.SINGLE, size: 4, color: BORDER_COLOR },
          bottom: { style: BorderStyle.SINGLE, size: 4, color: BORDER_COLOR },
          left: { style: BorderStyle.SINGLE, size: 4, color: BORDER_COLOR },
          right: { style: BorderStyle.SINGLE, size: 4, color: BORDER_COLOR },
        },
        children: lines.map(line => new Paragraph({
          alignment: AlignmentType.LEFT,
          spacing: { after: 0 },
          children: [new TextRun({ text: line || " ", font: MONO, size: 18 })],
        })),
      })],
    })],
  });
}

function spacer() { return new Paragraph({ text: "", spacing: { after: 120 } }); }

function makeTable(headerRow, rows, widths) {
  const totalWidth = 9000;
  const colWidths = widths || headerRow.map(() => totalWidth / headerRow.length);
  const headerCells = headerRow.map((text, i) => new TableCell({
    width: { size: colWidths[i], type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: ACCENT },
    children: [new Paragraph({ alignment: AlignmentType.LEFT, children: [new TextRun({ text, bold: true, color: "FFFFFF", font: FONT, size: 20 })] })],
  }));
  const bodyRows = rows.map(cols => new TableRow({
    children: cols.map((text, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      children: [new Paragraph({ alignment: AlignmentType.LEFT, children: [new TextRun({ text, font: FONT, size: 20 })] })],
    })),
  }));
  return new Table({ columnWidths: colWidths, width: { size: totalWidth, type: WidthType.DXA }, rows: [new TableRow({ children: headerCells }), ...bodyRows] });
}

const children = [];

children.push(
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 80 }, children: [new TextRun({ text: "RAG Copilot", bold: true, size: 56, color: ACCENT, font: FONT })] }),
  new Paragraph({ alignment: AlignmentType.LEFT, spacing: { after: 400 }, children: [new TextRun({ text: "Hybrid Ensemble Retrieval + Cross-Encoder Re-Ranking + LCEL RAG Chain — Project Documentation Sheet", size: 26, italics: true, font: FONT, color: "555555" })] }),
);

// 1. Overview
children.push(h1("1. Project Overview"));
children.push(p("RAG Copilot is a Streamlit app that lets a user upload a PDF and either ask specific questions about it (Q&A Copilot Mode) or generate a full executive summary (Executive Summarization Mode)."));
children.push(p("Both modes are built as LCEL Runnables (LangChain Expression Language), composed with the | pipe operator, on top of a hybrid retrieval pipeline (dense + sparse search fused with Reciprocal Rank Fusion) and cross-encoder re-ranking. Embeddings, re-ranking, and generation all run through the Hugging Face Inference API using a single access token."));

children.push(h2("Pipeline at a glance"));
children.push(bullet("Ingestion: PDF → character chunks (800/150) → Dense (Chroma + bge-m3) + Sparse (BM25)"));
children.push(bullet("Mode Router: Streamlit UI, Option A (Q&A) vs Option B (Summary)"));
children.push(bullet("Hybrid Ensemble + Re-Rank: dense+sparse top-10 each → RRF fusion (40% BM25/60% dense) → cross-encoder re-rank → top 3 chunks"));
children.push(bullet("LCEL RAG Chain: retrieve → format+cite → prompt | llm | parser, run via .invoke()"));
children.push(bullet("Executive Summary Chain: token-length routed Stuffing (one call) or Map-Reduce (.batch() over sections, then one reduce call)"));

// 2. File structure
children.push(h1("2. File Structure"));
children.push(code(
`rag-copilot/
├── app.py
├── config.py
├── requirements.txt
├── .env.example
└── src/
    ├── __init__.py
    ├── ingestion.py
    ├── retriever.py
    └── chain.py`
));
children.push(spacer());
children.push(makeTable(
  ["Path", "Description"],
  [
    ["app.py", "Streamlit UI — entry point & mode router"],
    ["config.py", "All tunable constants"],
    ["requirements.txt", "Python dependencies"],
    [".env.example", "Template for your HF token"],
    ["src/__init__.py", "Python package initializer"],
    ["src/ingestion.py", "PDF loading, character chunking & dual indexing"],
    ["src/retriever.py", "Hybrid Ensemble (BM25 + Chroma) & Cross-Encoder"],
    ["src/chain.py", "LCEL RAG chain & source citation formatter"],
  ],
  [3000, 6000]
));
children.push(spacer());

// 3. Requirements
children.push(h1("3. Requirements"));
children.push(makeTable(
  ["Package", "Purpose"],
  [
    ["streamlit", "Web UI"],
    ["langchain", "Main package (composability helpers)"],
    ["langchain-core", "Runnable/LCEL base abstractions, prompts, output parsers"],
    ["langchain-classic", "EnsembleRetriever (moved here post-LangChain-1.0 split)"],
    ["langchain-community", "PyPDFLoader, BM25Retriever"],
    ["langchain-text-splitters", "RecursiveCharacterTextSplitter"],
    ["langchain-huggingface", "HuggingFaceEndpoint, ChatHuggingFace, HF embeddings"],
    ["langchain-chroma", "Dedicated Chroma vector store integration"],
    ["huggingface_hub", "Low-level Inference API client (re-ranker)"],
    ["chromadb", "Dense vector database"],
    ["rank_bm25", "Sparse / lexical BM25 index"],
    ["pypdf", "PDF text extraction"],
    ["tiktoken", "Token counting for the Map-Reduce vs Stuffing decision"],
    ["python-dotenv", "Loads .env into environment variables"],
  ],
  [3600, 5400]
));
children.push(spacer());
children.push(p("Install with:"));
children.push(code("pip install -r requirements.txt"));
children.push(spacer());

// 4. HF Models
children.push(h1("4. Hugging Face Models Used"));
children.push(makeTable(
  ["Role", "Model", "Access Method"],
  [
    ["Dense embeddings", "BAAI/bge-m3", "HF Inference API — HuggingFaceEndpointEmbeddings"],
    ["Cross-encoder re-ranker", "BAAI/bge-reranker-base", "HF Inference API — raw InferenceClient.post()"],
    ["Generation LLM", "meta-llama/Meta-Llama-3.1-8B-Instruct (swap: Qwen/Qwen2.5-7B-Instruct)", "HF Inference API — HuggingFaceEndpoint + ChatHuggingFace"],
  ],
  [2400, 3600, 3000]
));
children.push(spacer());
children.push(p("All three route through one Hugging Face token (HUGGINGFACEHUB_API_TOKEN). Gated models such as Llama-3.1 require accepting the license on the model's Hugging Face page with the same account that owns the token; Qwen2.5-7B-Instruct is ungated and works immediately as a fallback."));

// 5. Setup
children.push(h1("5. Setup Instructions"));
children.push(code(
`git clone <your-repo>
cd rag-copilot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py`
));
children.push(spacer());
children.push(p("Notes:"));
children.push(bullet("On Windows, activate the virtual environment with venv\\Scripts\\activate instead of source venv/bin/activate."));
children.push(bullet("Before running streamlit run app.py, open .env and paste your real HUGGINGFACEHUB_API_TOKEN."));

// 6. How each file works
children.push(h1("6. How Each File Works"));

const fileExplanations = [
  ["config.py", "Every tunable constant: chunk size/overlap, top-k values, RRF weights, model IDs, token thresholds. Loads .env and fails fast if the HF token is missing."],
  ["src/ingestion.py — PDF loading, character chunking & dual indexing", "get_embedding_model() is a cached BAAI/bge-m3 wrapper over the HF Inference API. load_and_split_pdf() reads pages with PyPDFLoader and chunks them (800 chars / 150 overlap) via RecursiveCharacterTextSplitter, stamping source/page metadata. build_dense_index()/load_dense_index() write and reopen the persisted langchain-chroma collection. build_sparse_index()/load_sparse_index() build and rebuild the BM25 index, pickling the source chunks. ingest_pdf() is the single entry point app.py calls."],
  ["src/retriever.py — Hybrid Ensemble (BM25 + Chroma) & Cross-Encoder", "get_hybrid_retriever() wraps the Chroma retriever (top 10) and BM25 retriever (top 10) in langchain_classic's EnsembleRetriever, performing Reciprocal Rank Fusion at the configured 40% BM25 / 60% dense weights. rerank_with_cross_encoder() sends the ~20 fused candidates to BAAI/bge-reranker-base and keeps the top 3. get_reranked_retriever() wraps the whole flow in a RunnableLambda, exposing it as a Runnable[str, list[Document]] that composes with | inside chain.py exactly like a built-in retriever."],
  ["src/chain.py — LCEL RAG chain & source citation formatter", "get_llm() is a cached ChatHuggingFace singleton shared by both chains. format_docs_with_citations() is the source citation formatter, turning retrieved chunks into a labeled context string plus a structured citation list. build_qa_chain() is Option A's LCEL pipeline: RunnableLambda(retrieve+format) | RunnableParallel(answer=prompt|llm|parser, citations=passthrough) — call .invoke({'question': ...}). build_summary_chain() is Option B: a RunnableLambda routes on token count to a Stuffing chain (one call) or a Map-Reduce chain (map_chain.batch() over sections, then one reduce_chain.invoke()) — call .invoke(chunks)."],
  ["app.py", "The mode router: file uploader triggers ingest_pdf() once per new file; a radio button selects Option A or B; each option builds its chain with build_qa_chain() / build_summary_chain(), calls .invoke(...), and renders the result."],
];

fileExplanations.forEach(([file, desc]) => {
  children.push(new Paragraph({
    alignment: AlignmentType.LEFT,
    spacing: { before: 160, after: 60 },
    children: [new TextRun({ text: file, bold: true, font: MONO, size: 20, color: ACCENT })],
  }));
  children.push(p(desc));
});

// 7. Design notes
children.push(h1("7. Design Notes"));
children.push(bullet("LangChain 1.0 split: EnsembleRetriever now lives in langchain-classic (not langchain.retrievers as in the 0.x days). BM25Retriever and PyPDFLoader remain in langchain-community. Chroma has its own dedicated langchain-chroma package now."));
children.push(bullet("Re-ranker call uses a raw InferenceClient.post(). If your HF plan lacks Inference API access to bge-reranker-base, swap in a local sentence_transformers.CrossEncoder inside rerank_with_cross_encoder — same function signature, nothing else changes."));
children.push(bullet("Gated models: Llama-3.1-8B-Instruct requires accepting Meta's license on Hugging Face first; Qwen2.5-7B-Instruct is ungated and works immediately."));
children.push(bullet("Persistence: delete data/vectorstore/ to force a clean re-index."));

const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: convertInchesToTwip(1), bottom: convertInchesToTwip(1), left: convertInchesToTwip(1), right: convertInchesToTwip(1) },
      },
    },
    children,
  }],
  styles: {
    default: {
      document: { run: { font: FONT }, paragraph: { alignment: AlignmentType.LEFT } },
      heading1: { run: { color: ACCENT, size: 32, bold: true, font: FONT }, paragraph: { spacing: { before: 320, after: 160 }, alignment: AlignmentType.LEFT } },
      heading2: { run: { color: "2D3748", size: 26, bold: true, font: FONT }, paragraph: { spacing: { before: 240, after: 120 }, alignment: AlignmentType.LEFT } },
    },
  },
});

Packer.toBuffer(doc).then(buf => {
  require("fs").writeFileSync("RAG_Copilot_Documentation.docx", buf);
  console.log("done");
});
