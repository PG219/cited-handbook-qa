import json

from Ingestion.loader import load_pdf
from Ingestion.chunker import chunk_documents
from Retrieval.embeddings import Embed
from Retrieval.vector_store import FaissIndex
from main import (
    build_context,
    generate_answer_gemini,
    should_refuse
)

# --------------------------------------------------
# Load evaluation questions
# --------------------------------------------------


eval_questions = [
  {
    "question": "What is congestion control?",
    "ground_truth": "Congestion control prevents network overload and packet loss by regulating traffic flow.",
    "expected_pages": [73, 76]
  },
  {
    "question": "What is flow control in TCP?",
    "ground_truth": "Flow control ensures that a sender does not overwhelm a receiver with data.",
    "expected_pages": [73]
  },
  {
    "question": "What happens when congestion is not controlled?",
    "ground_truth": "Uncontrolled congestion can lead to severe delays, packet loss, and congestion collapse.",
    "expected_pages": [76]
  }
]


# --------------------------------------------------
# Build RAG pipeline (same as main)
# --------------------------------------------------

docs = load_pdf("Computer Networks.pdf")
chunks = chunk_documents(docs)
texts = [c.page_content for c in chunks]

embedder = Embed()
vectors = embedder.embed(texts)

index = FaissIndex(vectors.shape[1])
index.add(vectors)

# --------------------------------------------------
# Run evaluation
# --------------------------------------------------

eval_results = []

for item in eval_questions:
    question = item["question"]
    ground_truth = item.get("ground_truth", "")

    q_vec = embedder.embed([question])
    distances, indices = index.search(q_vec, k=5)

    contexts = [chunks[i].page_content for i in indices[0]]

    if should_refuse(distances[0][0]):
        answer = "I don't know."
    else:
        context_str = build_context(chunks, indices[0])
        answer = generate_answer_gemini(context_str, question)

    eval_results.append({
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth
    })

# --------------------------------------------------
# Save RAGAS input
# --------------------------------------------------
print(eval_results)
with open("eval/ragas_input.json", "w") as f:
    json.dump(eval_results, f, indent=2)

print("✅ ragas_input.json created successfully")
