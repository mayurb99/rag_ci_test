from src.rag_pipeline import get_rag_answer

query = "What are employee benefits?"

print("Running main RAG app...\n")

answer = get_rag_answer(query)

print("Final Answer:\n")
print(answer)