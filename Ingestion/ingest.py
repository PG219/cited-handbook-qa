from loader import load_pdf
from chunker import chunk_documents

docs = load_pdf("Computer Networks.pdf")
chunks = chunk_documents(docs)

print(f"Loaded {len(docs)} pages")
print(f"Created {len(chunks)} chunks")

print(chunks[0].page_content[:300])
print(chunks[0].metadata)
