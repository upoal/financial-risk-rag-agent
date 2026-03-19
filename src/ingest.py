import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

RAW_DIR = Path("data/raw")
CHROMA_DIR = "data/processed/chroma_db"

def load_pdfs(directory: Path) -> list:
    docs = []
    for pdf_path in directory.glob("*.pdf"):
        print(f"Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs.extend(loader.load())
    print(f"Total pages loaded: {len(docs)}")
    return docs

def chunk_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # chunks for better embedding performance
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

# def embed_and_store(chunks: list) -> Chroma:
#     embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=embeddings,
#         persist_directory=CHROMA_DIR
#     )
#     print(f"Embeddings stored in: {CHROMA_DIR}")
#     return vectorstore

def embed_and_store(chunks: list) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Process in batches to avoid rate limits
    BATCH_SIZE = 100 
    vectorstore = None  # Initialize the storage for vector embeddings 
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i:i + BATCH_SIZE]  # Get the current batch of chunks
        print(f"Embedding batch {i//BATCH_SIZE + 1}/{(len(chunks)-1)//BATCH_SIZE + 1} ({len(batch)} chunks)...")
        
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DIR
            )
        else:
            vectorstore.add_documents(batch)
    
    print(f"Embeddings stored in: {CHROMA_DIR}")
    return vectorstore

if __name__ == "__main__":
    docs = load_pdfs(RAW_DIR)
    chunks = chunk_documents(docs)
    embed_and_store(chunks)
    print("Ingestion complete.")