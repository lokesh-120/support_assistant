"""
FastAPI wrapper around the LangGraph pipeline.

Run locally with:
    uvicorn main:app --host 0.0.0.0 --port 7860

MOCK_LLM is left at its default (unset / "1") for the graded baseline --
no signup, no API key, and no network call to any LLM provider are needed.
"""

from fastapi import FastAPI

from graph import GraphState, build_graph
from models import AskRequest, AskResponse

app = FastAPI(
    title="Zepto Support Assistant",
    description="LangGraph-orchestrated RAG support assistant for Zepto policies.",
)

# Build/compile the graph once at startup rather than per-request.
_compiled_graph = build_graph()


@app.get("/")
def root() -> dict:
    return {"status": "ok", "service": "Zepto Support Assistant"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    initial_state: GraphState = {
        "query": request.query,
        "intent": "",
        "retrieved_chunks": [],
        "answer": "",
        "sources": [],
        "confidence": 0.0,
    }
    final_state = _compiled_graph.invoke(initial_state)
    return AskResponse(
        answer=final_state["answer"],
        sources=final_state["sources"],
        confidence=final_state["confidence"],
    )
