from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import time
# import sys
# import os

#make sure src/ is importable
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import load_vectorstore
from src.agent import ask  ## run ask method; call agent and return response



# load vectorstore once at startup 
vectorstore = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore
    print("Loading vectorstore...")
    vectorstore = load_vectorstore()
    print("Vectorstore ready. API is live.")
    yield  # before/after shutdown line
    print("Shutting down.")



# App 
app = FastAPI(
    title="Financial Risk RAG Agent",
    description="A RAG-powered Q&A agent over regulatory and financial risk documents for Santander Mexico.",
    version="1.0.0",
    lifespan=lifespan
)



# Schemas 
class QueryRequest(BaseModel):
    question: str
    k: int = 5  # number of chunks to retrieve, default 5

class SourceItem(BaseModel):
    source_id: int
    document: str
    page: int | str
    score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceItem]
    confidence: float
    low_confidence: bool
    response_time_ms: float





# endpoints
@app.get("/")
def root():
    return {"status": "Financial Risk RAG Agent is running"}


@app.get("/health")  
def health_check():   # later docker/azure might ping this, add it now and fix later ?
    return {
        "status": "ok",
        "chunks_loaded": vectorstore._collection.count() if vectorstore else 0
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    if vectorstore is None:
        raise HTTPException(status_code=503, detail="Vectorstore not loaded yet.")
    
    start = time.time()
    result = ask(request.question, vectorstore, k=request.k)
    elapsed = round((time.time() - start) * 1000, 2)
    
    return QueryResponse(
        question=request.question,
        answer=result["answer"],
        sources=result["sources"],
        confidence=result["confidence"],
        low_confidence=result["low_confidence"],
        response_time_ms=elapsed
    )

''' STEPS TO REDEPLOY:
# 1. Make your code changes

# 2. Rebuild the image with a new version tag
docker build -t financial-risk-rag-agent .
docker tag financial-risk-rag-agent upoal/financial-risk-rag-agent:vX  # update version number

# 3. Push to Docker Hub
docker push upoal/financial-risk-rag-agent:vX

# 4. Go to Render → Settings → update image tag to :vX → Save
# Render automatically redeploys
'''