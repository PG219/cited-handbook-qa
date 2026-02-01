## Cited Handbook Q&A

This project implements a Retrieval-Augmented Generation (RAG) system with explicit grounding and refusal logic. The focus so far has been on building the pipeline incrementally and transparently, validating each stage before adding generation.

✅ Completed Components

1. Ingestion

PDFs are loaded using a deterministic loader.

Documents are recursively chunked with overlap to preserve semantic continuity.

Chunk-level metadata (source file and page number) is retained for traceability.

2. Embeddings

Text chunks are embedded locally using sentence-transformers/all-MiniLM-L6-v2.

Embeddings are normalized to float32 for FAISS compatibility.

Verified embedding shape: (N, 384).

3. Vector Store (FAISS)

Chunk embeddings are indexed using FAISS IndexFlatL2.

Semantic similarity search retrieves top-k relevant chunks.

Retrieval quality was manually inspected and validated.

4. Grounded Generation (Gemini)

Uses Google Gemini (google.genai) strictly as a constrained generator.

Answers are generated only from retrieved chunks.

Each response includes explicit source citations (file name + page number).

Distance-based refusal logic ensures the system says “I don’t know” when retrieval confidence is low.

Hallucination-prone queries are correctly rejected.

🔍 Example Behavior

Supported query → grounded answer with citations.

Unsupported query → system abstains (“I don’t know”).

🧠 Design Philosophy

Retrieval correctness is prioritized before generation.

No agent chains or hidden abstractions are used.

Each step (ingestion, embedding, retrieval, generation) is explicitly implemented and inspectable.

⏭ Next Steps

Normalize embeddings and switch from L2 distance to cosine similarity.

Compare retrieval quality before vs after normalization.

Add evaluation cases to systematically test grounding and refusal behavior.