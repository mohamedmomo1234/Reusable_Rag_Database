"""
Optional FastAPI backend for the reusable RAG engine.

Streamlit does not require this file to be running.

Run:
uvicorn api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import APP_NAME, APP_DESCRIPTION
from graph import ask_rag_bot
from rate_limiter import allow_request


app = FastAPI(
    title=f"{APP_NAME} API",
    description=APP_DESCRIPTION,
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    session_id: str = "anonymous"
    history: list[dict] = []


@app.get("/health")
def health():
    return {"status": "ok", "app": APP_NAME}


@app.post("/chat")
def chat(request: ChatRequest):
    if not allow_request(request.session_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests.",
        )

    return ask_rag_bot(
        question=request.question,
        history=request.history,
    )
