# Cited Handbook Q&A

A Retrieval-Augmented Generation (RAG) application that answers questions **strictly from provided documents**, with **clear citations and page references**.

The goal of this project was to build a RAG system **from scratch**, focusing on correctness, grounding, and transparency rather than just “LLM output”.

---

## 🚀 What this does

- Upload a document (PDF)
- Ask natural language questions
- Get:
  - grounded answers
  - exact source references
  - page numbers for verification
- If the answer is not present in the document, the system explicitly responds with **“I don’t know.”**

---

## 🧠 System Architecture

---

## 🔧 Core Components

### 1️⃣ Document Ingestion
- PDF loading using LangChain loaders
- Metadata preserved (source, page number)
- Recursive text chunking

### 2️⃣ Embeddings & Retrieval
- Sentence embeddings using a local embedding model
- Embeddings normalized to unit length
- FAISS index with **cosine similarity** (inner product)
- Top-k semantic retrieval

### 3️⃣ Guardrails & Grounding
- Strict prompt rules: answers must come only from retrieved context
- Refusal mechanism when retrieval confidence is low
- No external or hallucinated knowledge allowed

### 4️⃣ API Layer
- FastAPI `/query` endpoint
- Returns:
  - answer
  - citations (source, page, chunk ID)
  - confidence indicator
- Designed to integrate with a frontend PDF viewer for highlighting cited text

---

## 📦 Example API Response

```json
{
  "question": "What is congestion control?",
  "answer": "Congestion control prevents network overload and packet loss...",
  "citations": [
    {
      "source": "Computer Networks.pdf",
      "page_index": 75,
      "display_page": 76,
      "chunk_id": 417,
      "text": "Congestion is a condition of severe delay..."
    }
  ],
  "confidence": "high"
}

