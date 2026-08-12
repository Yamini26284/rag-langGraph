"""
FastAPI server for rag-langgraph.

Endpoints:
  GET  /health  -> liveness check
  POST /ask     -> { "question": "..." } -> { answer, citations, trace }

Run: uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.graph import ask

app = FastAPI(title="rag-langgraph", version="0.1.0")


class AskRequest(BaseModel):
    question: str


class Citation(BaseModel):
    chunk_id: str
    source_file: str
    heading: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace: dict


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(request: AskRequest):
    question = request.question.strip()
    if not question:
        return JSONResponse(
            status_code=400,
            content={"error": "question must not be empty"},
        )

    try:
        result = ask(question)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "internal error processing question", "detail": str(e)},
        )