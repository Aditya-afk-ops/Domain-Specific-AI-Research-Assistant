import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# 1. Directory to load documents from
DATA_PATH = "data/"
# 2. Pinecone index name
INDEX_NAME = os.environ["PINECONE_INDEX_NAME"]
# 3. OpenAI Embedding model
EMBEDDINGS_MODEL = "text-embedding-3-small"
# 4. Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
# ---------------------

def load_documents():
    """Loads documents from the DATA_PATH directory."""
    print(f"Loading documents from {DATA_PATH}...")
    # Use DirectoryLoader, which can handle multiple file types
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*",  # Load all files
        loader_cls=lambda path: PyPDFLoader(path) if path.endswith('.pdf') else None,
        use_multithreading=True,
        show_progress=True
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} document(s).")
    return documents

def split_documents(documents):
    """Splits documents into manageable chunks."""
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")
    return chunks

def initialize_vector_store(chunks):
    """Initializes OpenAI embeddings and stores chunks in Pinecone."""
    print("Initializing OpenAI embeddings...")
    # Use the specified embedding model
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_MODEL)
    
    print(f"Storing {len(chunks)} chunks in Pinecone index '{INDEX_NAME}'...")
    # This command creates a new index if it doesn't exist or updates an existing one
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )
    print("Data ingestion complete!")

def main():
    documents = load_documents()
    if not documents:
        print("No documents found. Please add .txt or .pdf files to the 'data' directory.")
        return
    
    chunks = split_documents(documents)
    initialize_vector_store(chunks)

if __name__ == "__main__":
    main()