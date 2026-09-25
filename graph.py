"""
LangGraph orchestration for the Zepto support assistant.

Every node's generation step branches on the MOCK_LLM environment variable:
  - MOCK_LLM unset or "1" (default, graded baseline): deterministic,
    rule-based / templated logic only. No LLM call is made anywhere.
  - MOCK_LLM="0" (optional, ungraded extension): calls a real LLM
    (see llm.py) for classification and generation.

The retrieval step in retrieve_and_answer always runs for real in both
modes, since local embeddings + ChromaDB require no API key or network call.
"""

import os
from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

from ingest import build_or_load_collection, retrieve_top_k
from prompts import build_prompt

POLICY_KEYWORDS = [
    "delivery",
    "return",
    "refund",
    "membership",
    "tracking",
    "cancel",
    "gift card",
    "support hours",
]


class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    confidence: float


_collection = None


def get_collection():
    global _collection
    if _collection is None:
        _collection = build_or_load_collection()
    return _collection


def is_mock_mode() -> bool:
    """MOCK_LLM unset or any value other than '0' -> mock (graded) mode."""
    return os.environ.get("MOCK_LLM", "1") != "0"


def _keyword_classify(query: str) -> str:
    lowered = query.lower()
    return "policy_question" if any(k in lowered for k in POLICY_KEYWORDS) else "general_question"


# ---------------------------------------------------------------------------
# Node 1: classify_intent
# ---------------------------------------------------------------------------
def classify_intent(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Graded baseline: pure keyword heuristic, no LLM call.
        intent = _keyword_classify(query)
    else:
        # Optional MOCK_LLM=0 extension: ask the LLM to classify instead.
        from llm import classify_intent_llm

        try:
            intent = classify_intent_llm(query)
        except Exception:
            intent = _keyword_classify(query)

    return {**state, "intent": intent}


def route_from_intent(state: GraphState) -> str:
    """Conditional edge target based on the classified intent."""
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"


# ---------------------------------------------------------------------------
# Node 2: retrieve_and_answer
# ---------------------------------------------------------------------------
def retrieve_and_answer(state: GraphState) -> GraphState:
    query = state["query"]

    # Retrieval always runs for real: local embeddings + ChromaDB need no
    # API key and no network call in either mode.
    collection = get_collection()
    retrieved = retrieve_top_k(collection, query, k=3)
    chunk_ids = [r["chunk_id"] for r in retrieved]

    if is_mock_mode():
        # Graded baseline: canned templated answer, no LLM call.
        top_chunk_text = retrieved[0]["text"] if retrieved else ""
        top_chunk_snippet = top_chunk_text[:200]
        answer = f"Based on the retrieved context: {top_chunk_snippet}"
        sources = chunk_ids
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: real LLM, grounded via the
        # structured prompt template, with schema validation + retries.
        from llm import grounded_answer_llm

        context = "\n\n".join(f"[{r['chunk_id']}] {r['text']}" for r in retrieved)
        prompt = build_prompt(query=query, context=context)
        parsed = grounded_answer_llm(prompt)

        if parsed is None:
            answer = "ERROR: the model could not produce a schema-valid response after retries."
            sources = chunk_ids
            confidence = 0.0
        else:
            answer = str(parsed.get("answer", ""))
            sources = parsed.get("sources") or chunk_ids
            try:
                confidence = float(parsed.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

    return {
        **state,
        "retrieved_chunks": retrieved,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Node 3: direct_answer
# ---------------------------------------------------------------------------
def direct_answer(state: GraphState) -> GraphState:
    query = state["query"]

    if is_mock_mode():
        # Graded baseline: fixed canned string, no LLM call.
        answer = "I can only answer questions about Zepto policies right now."
        confidence = 1.0
    else:
        # Optional MOCK_LLM=0 extension: direct LLM call, no retrieval.
        from llm import direct_answer_llm

        try:
            answer = direct_answer_llm(query)
            confidence = 0.7
        except Exception:
            answer = "I can only answer questions about Zepto policies right now."
            confidence = 1.0

    return {
        **state,
        "retrieved_chunks": [],
        "answer": answer,
        "sources": [],
        "confidence": confidence,
    }


def build_graph():
    """Compile the 3-node LangGraph StateGraph with its conditional edge."""
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("retrieve_and_answer", retrieve_and_answer)
    graph.add_node("direct_answer", direct_answer)

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        route_from_intent,
        {
            "retrieve_and_answer": "retrieve_and_answer",
            "direct_answer": "direct_answer",
        },
    )
    graph.add_edge("retrieve_and_answer", END)
    graph.add_edge("direct_answer", END)

    return graph.compile()
