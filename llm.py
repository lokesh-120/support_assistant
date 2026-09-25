"""
Optional, ungraded extension. Only used when MOCK_LLM=0 is explicitly set.

Calls Groq's free-tier chat completion API (https://console.groq.com).
Requires the GROQ_API_KEY environment variable -- never hardcode the key
or commit it to the repository.
"""

import json
import os
from typing import Optional

import requests

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"


def call_groq_llm(prompt: str) -> str:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. This is required only for the optional "
            "MOCK_LLM=0 extension."
        )

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def classify_intent_llm(query: str) -> str:
    prompt = (
        "Classify the following customer question as exactly one word, "
        "either 'policy_question' or 'general_question'. A policy_question "
        "concerns Zepto's delivery, returns, membership, tracking, "
        "cancellation, damaged/missing items, gift cards, or support hours. "
        "Respond with only the single classification word, nothing else.\n\n"
        f"Question: {query}"
    )
    raw = call_groq_llm(prompt).strip().lower()
    return "policy_question" if "policy_question" in raw else "general_question"


def direct_answer_llm(query: str) -> str:
    prompt = (
        "You are Zepto's customer support assistant. Answer the following "
        "general question helpfully and briefly. Do not invent specific "
        "Zepto policy details you are not certain about.\n\n"
        f"Question: {query}"
    )
    return call_groq_llm(prompt).strip()


def _clean_json_text(raw: str) -> str:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def grounded_answer_llm(prompt: str, max_retries: int = 2) -> Optional[dict]:
    """
    Call the LLM with the structured prompt and validate the JSON output
    against the expected schema (answer/sources/confidence), retrying with
    a corrective instruction up to max_retries additional times on failure.
    """
    current_prompt = prompt
    last_error = None

    for _ in range(max_retries + 1):
        raw = call_groq_llm(current_prompt)
        cleaned = _clean_json_text(raw)
        try:
            parsed = json.loads(cleaned)
            if not all(k in parsed for k in ("answer", "sources", "confidence")):
                raise ValueError("Response JSON is missing required fields.")
            return parsed
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            current_prompt = (
                prompt
                + "\n\nYour previous response was not valid JSON matching the "
                "required schema (answer, sources, confidence). "
                f"Error: {e}. Respond again with ONLY a valid JSON object, "
                "no markdown fences, no extra text."
            )

    return None
