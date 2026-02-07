from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from Ingestion.loader import load_pdf
from Ingestion.chunker import chunk_documents
from Retrieval.embeddings import Embed
from Retrieval.vector_store import FaissIndex
from main import (
    build_context,
    generate_answer_gemini,
    should_refuse
)
app = FastAPI(
    title="RAG Q&A API",
    description="Query documents with grounded, cited answers",
    version="1.0"
)
print("🔹 Loading documents...")
docs = load_pdf("Computer Networks.pdf")

print("🔹 Chunking documents...")
chunks = chunk_documents(docs)
texts = [c.page_content for c in chunks]

print("🔹 Creating embeddings...")
embedder = Embed()
vectors = embedder.embed(texts)

print("🔹 Building FAISS index...")
index = FaissIndex(vectors.shape[1])
index.add(vectors)

print("✅ RAG pipeline ready")

# -------------------------------
# Request / Response models
# -------------------------------
class QueryRequest(BaseModel):
    question: str


class Citation(BaseModel):
    source: str
    page_index: int
    display_page: int
    chunk_id: int
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]
    confidence: str
def build_citations(indices):
    citations = []

    for idx in indices:
        chunk = chunks[idx]
        citations.append({
            "source": chunk.metadata.get("source"),
            "page_index": chunk.metadata.get("page"),
            "display_page": chunk.metadata.get("page") + 1,
            "chunk_id": idx,
            "text": chunk.page_content[:300]
        })

    return citations
@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    question = req.question.strip()

    # Safety guard
    if len(question) == 0:
        return {
            "question": question,
            "answer": "Please ask a valid question.",
            "citations": [],
            "confidence": "low"
        }

    # Embed question
    q_vec = embedder.embed([question])

    # Retrieve top-k
    distances, indices = index.search(q_vec, k=5)

    # Refusal logic
    if should_refuse(distances[0][0]):
        return {
            "question": question,
            "answer": "I don't know.",
            "citations": [],
            "confidence": "low"
        }

    # Build context + answer
    context = build_context(chunks, indices[0])
    answer = generate_answer_gemini(context, question)

    citations = build_citations(indices[0])

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "confidence": "high"
    }
