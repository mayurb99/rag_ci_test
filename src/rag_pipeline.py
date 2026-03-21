import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()

def get_rag_answer(query):

    model = SentenceTransformer("all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("policy_docs")

    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )

    context = "\n".join(results["documents"][0])

    prompt = f"""
Answer using the context below.

Context:
{context}

Question:
{query}
"""

    llm = InferenceClient(
        model="allenai/Olmo-3-7B-Instruct:publicai",
        api_key=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    response = llm.chat_completion(
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content