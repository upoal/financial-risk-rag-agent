"""
Handles retrieval of relevant document chunks from the vector database based on user queries.
"""
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()  #keys

CHROMA_DIR = "data/processed/chroma_db"

def load_vectorstore() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )
    print(f"Vectorstore loaded: {vectorstore._collection.count()} chunks available")
    return vectorstore

def retrieve(query: str, vectorstore: Chroma, k: int = 5) -> list:
    """
    Uses raw cosine similarity instead of the normalized relevance score
    to avoid score compression artifacts from ChromaDB's transformation.
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    
    retrieved = []
    for doc, raw_score in results:
        # Cosine distance from ChromaDB — convert to similarity
        # raw_score is a distance (0 = identical, 2 = opposite)
        # so similarity = 1 - (raw_score / 2)
        similarity = round(1 - (raw_score / 2), 4)
        
        retrieved.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "Unknown"),
            "score": similarity
        })
    
    return retrieved

def format_context(chunks: list) -> str:
    """
    Formats retrieved chunks into a clean context block for the LLM prompt.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source_name = Path(chunk["source"]).name
        context_parts.append(
            f"[Source {i}: {source_name}, Page {chunk['page']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(context_parts)

if __name__ == "__main__":
    vs = load_vectorstore()
    query = "What are the capital requirements under Basel III?"
    results = retrieve(query, vs)
    
    print(f"\nQuery: {query}\n")
    for r in results:
        print(f"Score: {r['score']} | {Path(r['source']).name} | Page {r['page']}")
        print(f"{r['content'][:200]}...\n")