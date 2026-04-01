# Financial Risk RAG Agent

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions about financial risk regulations using real regulatory documents. Built with LangChain, ChromaDB, FastAPI, and Streamlit — deployed on Docker and Render.

**Live Demo:** [financial-risk-rag-frontend.onrender.com/](https://financial-risk-rag-frontend.onrender.com/)  
**Interactive API Docs:** [/docs](https://financial-risk-rag-agent-v1.onrender.com/docs)

---

## Overview

Large language models hallucinate. In financial services, a hallucinated answer about capital requirements or credit loss provisioning is a liability.

This system solves that by grounding every answer in retrieved excerpts from official regulatory documents. No answer is generated without a traceable source. If the documents don't contain sufficient information to answer a question confidently, the system says so explicitly rather than guessing. I've created this tool after I realize I was struggling to find answers of my own when I previously worked for another financial institution. I wanted a tool that could, at the very least, point me in the right direction and possibly even provide a richer, more insightful answer that I would have otherwise. This version has been rebuilt for deployment now that I have a better understanding of the pipeline and workflow. The knowledgebase can be changed to create custom embeddings for any domain, open to potentially scaling.

This design reflects two principles central to model risk management in banking: **auditability** and **refusal over hallucination**.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
|                        INGESTION                            |
                                                              
      PDFs → Text Chunks (500 chars) → OpenAI Embeddings        
|                   → ChromaDB (11,353 chunks)                |
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
|                        RAG PIPELINE                         |
                                                             
           Question → Embed → Cosine Similarity Search               
|           → Top-k Chunks → GPT-4o-mini → Cited Answer       |
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                       SERVICE LAYER                         │
                                                             
             FastAPI (/query, /health, /docs)                          
|            Docker Container → Docker Hub → Render           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                        FRONTEND                             │
                                                             
│       Streamlit UI → Confidence Indicator → Audit Trail     │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Features

**Audit Trail**
Every answer returns the source document name and page number for each retrieved chunk. In regulated environments, answers without traceable sources are operationally unusable. 

**Confidence-Based Refusal**
A cosine similarity threshold gates every response. If the top retrieved chunk falls below the confidence threshold, the system returns an explicit "insufficient information" response rather than generating a plausible-sounding but unreliable answer. 

**Three-Tier Confidence Display**
The frontend displays 'Confident', 'Semi-Confident', or 'Not Confident' rather than raw scores.

**Configurable Retrieval**
The number of source chunks retrieved (k) is user-configurable at query time to allow for a tradeoff between response depth and latency.

---

## Document Corpus

| Document | Domain |
|---|---|
| Basel III Full Framework | Capital adequacy, leverage ratio |
| Basel III Reforms Summary | Post-2017 reforms, standardized approaches |
| IFRS 9 Financial Instruments | Expected credit loss, classification |
| Banxico Circular 3/2012 | Mexican central bank credit risk provisions |
| IFA Consolidated Annual Financial Report 2025 | Financial risk disclosures |

Total: **1,501 pages → 11,353 chunks**

---

## Tech Stack

| Layer | Technology |
|---|---|
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector Database | ChromaDB |
| Orchestration | LangChain |
| LLM | GPT-4o-mini (temperature=0) |
| API | FastAPI + Pydantic |
| Frontend | Streamlit |
| Containerization | Docker |
| Registry | Docker Hub |
| Deployment | Render |

---

## API Reference

### `GET /health`
Returns service status and number of loaded chunks.

```json
{
  "status": "ok",
  "chunks_loaded": 11353
}
```

### `POST /query`

**Request:**
```json
{
  "question": "What are the capital requirements under Basel III?",
  "k": 5
}
```

**Response:**
```json
{
  "question": "What are the capital requirements under Basel III?",
  "answer": "The capital requirements under Basel III are...",
  "sources": [
    {
      "source_id": 1,
      "document": "data/raw/baselIII.pdf",
      "page": 12,
      "score": 0.81
    }
  ],
  "confidence": 0.81,
  "low_confidence": false,
  "response_time_ms": 1823.4
}
```

---

## Running Locally

**1. Clone and set up environment:**
```bash
git clone https://github.com/YOUR_USERNAME/financial-risk-rag-agent.git
cd financial-risk-rag-agent
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure environment variables:**
```bash
cp .env.example .env
# Add your OpenAI API key to .env
```

**3. Add PDFs to `data/raw/` and run ingestion:**
```bash
python src/ingest.py
```

**4. Start the API:**
```bash
uvicorn api.main:app --reload
```

**5. Start the frontend:**
```bash
streamlit run frontend/app.py
```

---

## Running With Docker

```bash
docker build -t financial-risk-rag-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key financial-risk-rag-agent
```

---

## Design Decisions

**Why RAG over fine-tuning?**
Regulatory documents change. Fine-tuned models encode knowledge statically, and updating them requires retraining. RAG separates the knowledge layer (ChromaDB) from the reasoning layer (GPT-4o-mini), so new documents can be ingested without touching the model.

**Why GPT-4o-mini at temperature=0?**
Cost and determinism. At fractions of a cent per query, the entire project runs for under $5 in API costs. Temperature=0 ensures consistent, reproducible answers, and decreases the probability of unwanted creativity. The same question should return the same answer.

**Why refuse below the confidence threshold?**
A system that occasionally says "I don't know" is more trustworthy in a regulated environment than one that always produces an answer. Refusal behavior is the most direct expression of model risk management in such a service.

**Why a separate frontend and backend service?**
Independent deployability. The API can be updated, versioned, and consumed by other clients without touching the UI. The frontend can be redesigned without redeploying the RAG pipeline.

**Why deploy on Render?**
Unfortunately my student subcription to Azure had regional policy restrictions which prevented access to Container Registry + Apps. I simply thought to switch over to Render as a means to simplify the troublshooting process and produce a live product, for free. This does come with the tradeoff that the application needs to "cold start" after a while of inactivity, which is not acceptable for production, however I took this liberty for my own personal use. 

---

## Evaluation

Sample queries and observed behavior: please try the sample queries provided above the chat input. These provide examples of answers with different confidence scores. 

---

## Limitations

- Render free tier spins down after 15 minutes of inactivity. First request after idle period takes 30–60 seconds
- Answers are limited to content in the loaded document corpus Questions outside this domain will trigger the refusal response
- No conversation memory. Each query is stateless

---

## Author

**Diego Ortega Dounce**  
BSc Computer Science (French Immersion) · University of Ottawa  
diego.ortega.dounce@gmail.com · linkedin.com/in/diego-ortega-dounce/
