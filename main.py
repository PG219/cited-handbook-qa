import os
import numpy as np
from google import genai

from Ingestion.loader import load_pdf
from Ingestion.chunker import chunk_documents
from Retrieval.embeddings import Embed
from Retrieval.vector_store import FaissIndex
from google import genai

client = genai.Client(
    api_key="AIzaSyAnkmqgdX3zv6oATQhSk1nPPMZepW5v2B0"
)


def build_prompt(context: str, question: str) -> str:
    return f"""
You are a question-answering system.

RULES:
- Use ONLY the provided CONTEXT.
- If the answer is not present in the context, say exactly: "I don't know."
- Do NOT use external knowledge.
- Cite the source and page number.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


def build_context(chunks, indices, max_chunks=5):
    context_blocks = []
    used = set()

    for idx in indices[:max_chunks]:
        if idx in used:
            continue
        used.add(idx)

        chunk = chunks[idx]
        context_blocks.append(
            f"[Source: {chunk.metadata.get('source')}, "
            f"Page: {chunk.metadata.get('page')}]\n"
            f"{chunk.page_content}"
        )

    return "\n\n".join(context_blocks)


def generate_answer_gemini(context: str, question: str) -> str:
    prompt = build_prompt(context, question)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
        config={
            "temperature": 0.0,
            "top_p": 1.0
        }
    )

    return response.text.strip()


# =========================================================
# 4. Refusal logic
# =========================================================

def should_refuse(best_distance: float, threshold: float = 1.2) -> bool:
    """
    FAISS L2 distance: lower = more similar
    """
    return best_distance > threshold


# =========================================================
# 5. Ingestion + indexing
# =========================================================

print("Loading document...")
docs = load_pdf("Computer Networks.pdf")

print("Chunking document...")
chunks = chunk_documents(docs)

texts = [c.page_content for c in chunks]

print("Embedding chunks...")
embedder = Embed()
vectors = embedder.embed(texts)

print("Embeddings:", vectors.shape, vectors.dtype)

print("Building FAISS index...")
dim = vectors.shape[1]
index = FaissIndex(dim)
index.add(vectors)

print("Index size:", index.index.ntotal)


# =========================================================
# 6. Query → Retrieve → Grounded Generation
# =========================================================

query = "What is congestion control?"
print(f"\nQuery: {query}")

q_vec = embedder.embed([query])
distances, indices = index.search(q_vec, k=5)

best_distance = distances[0][0]

if should_refuse(best_distance):
    print("I don't know.")
else:
    context = build_context(chunks, indices[0])
    answer = generate_answer_gemini(context, query)
    print("\nAnswer:\n")
    print(answer)
