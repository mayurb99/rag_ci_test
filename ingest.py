import chromadb
from sentence_transformers import SentenceTransformer

# Load document
with open("sample_document.txt") as f:
    text = f.read()

# Chunking
chunks = []
chunk_size = 100
overlap = 20

start = 0
while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start = end - overlap

# Embedding
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

# Create DB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("policy_docs")

for i, chunk in enumerate(chunks):
    collection.add(
        ids=[str(i)],
        documents=[chunk],
        embeddings=[embeddings[i]]
    )

print("✅ ChromaDB created")