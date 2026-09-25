"""
Runs two example queries directly through the compiled graph (bypassing
HTTP) and prints their raw JSON responses -- one that should trigger
retrieval, one that should not. Run with MOCK_LLM left at its default.

    python demo.py

Equivalent to what POST /ask would return for the same two queries once
the FastAPI server is running.
"""

import json

from graph import build_graph

EXAMPLE_QUERIES = [
    "What is your delivery fee for small orders?",  # -> policy_question
    "What is the capital of France?",  # -> general_question
]


def main() -> None:
    compiled_graph = build_graph()

    for query in EXAMPLE_QUERIES:
        initial_state = {
            "query": query,
            "intent": "",
            "retrieved_chunks": [],
            "answer": "",
            "sources": [],
            "confidence": 0.0,
        }
        final_state = compiled_graph.invoke(initial_state)
        response = {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "confidence": final_state["confidence"],
        }
        print(f"Query: {query}")
        print(f"Classified intent: {final_state['intent']}")
        print(json.dumps(response, indent=2))
        print()


if __name__ == "__main__":
    main()
