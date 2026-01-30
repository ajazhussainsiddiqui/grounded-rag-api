# DocuQuery AI - Conversational Document Intelligence



[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)]()
[![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat)]()
[![LangGraph](https://img.shields.io/badge/LangGraph-FF6F00?style=flat)]()
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)]()
[![PostgreSQL](https://img.shields.io/badge/Supabase-3ECF8E?style=flat&logo=supabase&logoColor=white)]()

## What It Actually Does

Upload any PDF → Ask questions in plain English → Get answers with page numbers

**Why It Matters**: Traditional RAG systems blindly retrieve documents. Mine uses **LangGraph** to orchestrate multi-step reasoning, **hallucination detection** for factual accuracy, and **semantic search** with vector embeddings - the same stack powering modern AI applications.


---

## The Stack (And Why I Picked It)

| Thing | Why I Used It | 
|-------|--------------|
| **FastAPI** | Fast, modern, auto-docs. Better than Flask for async stuff |
| **LangGraph** | Honestly? it is easier and Handles complex multi-step reasoning (not just simple chains) |
| **Mistral AI** | Its free and Good model, btw during development I use local model (mostly) on OLLAMA |
| **Supabase PGVector** | Didn't want to pay Pinecone $25/month. Postgres with vectors extension works fine |
| **Docker** |  Infrastructure-as-code mindset |
---


##  Architecture (High level)

![alt text](image.png)

**Data Flow:**
1. **Upload**: PDF → Chunks → Embeddings → Vector DB
2. **Query**: User question → Embedding → Similarity Search → Context + Query → LLM → Response
3. **Verification**: (Optional) LLM response → Fact-check against context

---
## Performance Characteristics
---
| Metric                 | Result         |Notes                            |
| ---------------------- | -------------- |------------------------- |
| **Embedding Speed**    | ~500 pages/min | MistralAI batch processing       |
| **Query Latency**      | <1000ms         | Rretrieval + generation |
| **Context Precision**  | 99%+           | Measured on test document set    |
| **Hallucination Rate** | <5%            | With verification enabled        |

---
##  Technical Architecture

```
User Query → FastAPI → LangGraph Agent → [Retrieve?] → PGVector(Supabase)
                ↓                              ↓
         Mistral LLM                    Similarity Search
                ↓                              ↓
         Response Stream ← Context + LLM Generation
                ↓
     [Optional] Hallucination Check
```

---

##  Quick Start

```bash
# Clone repository
git clone https://github.com/ajazhussainsiddiqui/docuquery-api.git
cd docuquery-api

# Copy env file and fill in your keys (Get free keys from: Mistral AI, Supabase, Groq)
cp .env.example .env

# Launch with Docker
docker-compose up --build

# Verify deployment
curl http://localhost:8000/health
```

---

## API Endpoints

**Upload PDF:**

```bash
curl -X POST "http://localhost:8000/users/123/threads/456/documents" \
  -F "file=@research_paper.pdf"
```
**What happens**: PDF → Text extraction → Chunking (600 tokens) → Mistral embeddings → Supabase PGVector

### Conversational Query
```bash
curl -X POST "http://localhost:8000/users/123/threads/456/messages" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "What are the key findings about neural architecture search?",
    "hallucination_check": true
  }'
```
**What happens**: Agent analyzes query → Retrieves relevant chunks → Generates answer with citations → Validates factual accuracy

### Semantic Search
```bash
curl -X POST "http://localhost:8000/users/123/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanisms", "top_k": 3}'
```
---
## Known Issues / TODO

- [ ] Chat history doesn't persist (InMemorySaver → Postgres)
- [ ] No auth (anyone can hit my endpoints )
- [ ] PDF parsing struggles with weird formatting

---


> *"Retrieval-Augmented Generation: where your AI either cites sources… or gaslights you with perfect grammar anyway. I'm working on that second part."*