import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

PDF_PATH = "x.pdf"
DB_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_collection"

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_database():

    client = chromadb.PersistentClient(path=DB_DIR)

    embedding_func = embedding_functions.OllamaEmbeddingFunction(
        model_name="nomic-embed-text",
        url="http://localhost:11434",
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func
    )

    text = load_pdf_text(PDF_PATH)
    chunks = chunk_text(text)

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    collection.add(
        ids=ids,
        documents=chunks
    )

    print("Database created successfully.")

if __name__ == "__main__":
    build_database()