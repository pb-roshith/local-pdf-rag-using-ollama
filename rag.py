import chromadb
import requests

DB_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_collection"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:270m"

def rag_query(question):
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_texts=[question],
        n_results=3
    )

    retrieved = "\n".join(results["documents"][0])

    prompt = f"""
    You are a PDF question-answer bot.
    Use the retrieved context to answer.
    
    CONTEXT:
    {retrieved}
    
    QUESTION: {question}
    
    ANSWER:
    """

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        }
    ).json()

    return response["response"]

if __name__ == "__main__":
    while True:
        q = input("\nAsk: ")
        if q.lower() == "exit":
            break

        ans = rag_query(q)
        print("\nAnswer:", ans)
