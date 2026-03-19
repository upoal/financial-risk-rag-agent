from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from src.retriever import load_vectorstore, retrieve, format_context

load_dotenv()

# Confidence threshold. below this, refuse to answer rather than hallucinate
CONFIDENCE_THRESHOLD = 0.5  # ---> turn into dynamic variable? try to optimize this, this number can be tweaked based on usage. 

SYSTEM_PROMPT = """You are a financial risk analyst assistant for Santander México.
You answer questions strictly based on the provided regulatory and financial documents.

Rules:
- Only use information from the provided context
- Always cite your sources using [Source N] notation
- If the context does not contain enough information, say so explicitly — do not guess
- Be precise and professional, as your answers may inform risk decisions
- Highlight any regulatory obligations or compliance requirements you identify"""

def ask(query: str, vectorstore, k: int = 5) -> dict:
    """
    Full RAG pipeline: query -> retrieve -> generate -> return answer with citations.
    Returns a dict with: answer, sources, confidence, low_confidence flag.
    """
    #1: Retrieve relevant chunks
    chunks = retrieve(query, vectorstore, k=k)
    
    #2: Check confidence. if best score is too low, don't hallucinate
    top_score = chunks[0]["score"] if chunks else 0
    low_confidence = top_score < CONFIDENCE_THRESHOLD
    
    if low_confidence:
        return {
            "answer": "I could not find sufficient information in the loaded documents to answer this question confidently. Please rephrase or consult the source documents directly.",
            "sources": [],
            "confidence": top_score,
            "low_confidence": True
        }
    
    #3: Format context for the prompt
    context = format_context(chunks)
    
    # 4: Build messages
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"""Context from regulatory documents:

        {context}  # context as dict

        ----
        Question: {query}
        Answer based strictly on the context above. Cite sources using [Source N] notation.""")
    ]
    
    # Step 5: Call GPT-4o-mini
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(messages)
    
    # Step 6: Build source list for audit trail
    sources = [
        {
            "source_id": i + 1,
            "document": chunks[i]["source"],
            "page": chunks[i]["page"],
            "score": chunks[i]["score"]
        }
        for i in range(len(chunks))
    ]
    
    return {
        "answer": response.content,
        "sources": sources,
        "confidence": top_score,
        "low_confidence": False
    }

def print_response(result: dict):
    """Pretty-prints a response for terminal testing."""
    print("\n" + "="*60)
    print("ANSWER:")
    print("="*60)
    print(result["answer"])
    print("\n" + "-"*60)
    print(f"CONFIDENCE: {result['confidence']}")
    print("\nSOURCES (Audit Trail):")
    for s in result["sources"]:
        doc_name = s["document"].split("\\")[-1].split("/")[-1]
        print(f"  [{s['source_id']}] {doc_name} — Page {s['page']} (score: {s['score']})")
    print("="*60 + "\n")

if __name__ == "__main__":
    vs = load_vectorstore()
    
    # Test queries — these mirror what a Santander risk analyst would actually ask
    test_queries = [
        "What are the capital requirements under Basel III?",
        "How does IFRS 9 define expected credit loss?",
        "What are the liquidity coverage ratio requirements?",
    ]
    
    for query in test_queries:
        print(f"\nQuerying: {query}")
        result = ask(query, vs)
        print_response(result)