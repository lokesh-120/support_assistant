# Zepto Support Assistant (`/support_assistant`)

A small RAG service for Zepto's own delivery, returns, membership, and
support policies, built with local embeddings + ChromaDB + LangGraph +
FastAPI.

## Running it

```bash
cd support_assistant
pip install -r requirements.txt

# Graded baseline -- MOCK_LLM left at its default (unset == "1").
# No signup, no API key, no network call to any LLM provider.
uvicorn main:app --host 0.0.0.0 --port 7860
```

The first request (or `python ingest.py`) builds the ChromaDB collection by
embedding the 8 documents in `docs/` with `sentence-transformers`
(`all-MiniLM-L6-v2`); this only needs network access once, to download the
open-source model weights -- no account and no API key.

To reproduce the two example transcripts below directly (no HTTP needed):

```bash
python demo.py
```

Or with the server running:

```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "What is your delivery fee for small orders?"}'

curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

### Docker (required, graded baseline)

```bash
docker build -t zepto-support-assistant .
docker run -p 7860:7860 zepto-support-assistant
```

This builds and serves `POST /ask` locally with `MOCK_LLM=1` (the default
baked into the image); no push to any registry is required.

### Optional, ungraded extensions

- **Real LLM (MOCK_LLM=0):** set `MOCK_LLM=0` and `GROQ_API_KEY=<your key>`
  (free tier at console.groq.com, no card required) as environment
  variables. `classify_intent`, `retrieve_and_answer`, and `direct_answer`
  will then call the LLM instead of using the mock logic. This is entirely
  optional; the graded submission is evaluated with `MOCK_LLM` left at its
  default.
- **Hugging Face Spaces deployment:** the same `Dockerfile` can be pushed to
  a free-tier community CPU Space, storing `GROQ_API_KEY` as a Space secret
  (never hardcoded/committed). Not attempted for this submission -- the
  Dockerfile is only required to build and run locally.

## Architecture: ingestion -> embedding -> retrieval -> generation

**Ingestion** (`ingest.py::load_documents`): the 8 `docs/doc_0N.txt` files
are read from disk. `ingest.py::chunk_document` performs a simple
per-document chunking scheme -- each policy document is short enough to be
kept as a single chunk, with a fixed-size (400-char) fallback split for
anything longer.

**Embedding** (`ingest.py::build_or_load_collection`): each chunk is
embedded locally with `sentence-transformers`'s `all-MiniLM-L6-v2` model
(`get_embedding_model`) -- no API key, no network call at inference time.
The resulting vectors, chunk text, and `source_doc` metadata are stored in a
persistent ChromaDB collection named `zepto_policies` (configured with
`hnsw:space: cosine` so similarity search uses cosine distance), persisted
under `chroma_db/`.

**Retrieval** (`graph.py::retrieve_and_answer`, calling
`ingest.py::retrieve_top_k`): for queries classified as `policy_question`,
the query is embedded with the same local model and the top-3 most similar
chunks are pulled from the ChromaDB collection. This retrieval step always
runs for real in both `MOCK_LLM` states, since it needs no API key and no
network call.

**Generation** (`graph.py`'s three LangGraph nodes, compiled in
`build_graph`):
- `classify_intent` routes each query to `policy_question` or
  `general_question`.
- `retrieve_and_answer` (reached via the conditional edge for
  `policy_question`) produces the final answer from the retrieved chunks.
- `direct_answer` (reached for `general_question`) produces the final
  answer without retrieval.

A conditional edge out of `classify_intent`
(`graph.py::route_from_intent`) implements the graph-based intent router,
sending state to `retrieve_and_answer` or `direct_answer` based on the
classified intent. Both terminal nodes populate the shared `GraphState`
(`answer`, `sources`, `confidence`), which `main.py`'s `POST /ask` endpoint
validates against the `AskResponse` Pydantic model (`models.py`) before
returning it.

### Where `MOCK_LLM` branches things

`graph.py::is_mock_mode()` reads the `MOCK_LLM` environment variable
(`unset` or `"1"` -> mock; `"0"` -> real LLM) and every node's *generation*
step branches on it -- the routing logic itself (`route_from_intent`) does
not depend on `MOCK_LLM`.

| Node | Mock mode (default, graded) | MOCK_LLM=0 (optional extension) |
|---|---|---|
| `classify_intent` | Keyword heuristic (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`) | LLM call (`llm.py::classify_intent_llm`) |
| `retrieve_and_answer` | Retrieval runs for real either way; answer is the canned `f"Based on the retrieved context: {top_chunk_snippet}"` template; `sources`/`confidence` set deterministically in code (`confidence=1.0`) | Real LLM call using the structured prompt (`prompts.py`), with up to 2 retries on schema-invalid JSON (`llm.py::grounded_answer_llm`) |
| `direct_answer` | Fixed canned string, no retrieval, no LLM call | Direct LLM call, no retrieval (`llm.py::direct_answer_llm`) |

In mock mode nothing is ever passed to an LLM, so the Pydantic schema is
populated directly by code and cannot fail validation. In the optional
`MOCK_LLM=0` state, the same schema is instead filled from the LLM's raw
JSON output, validated against `AskResponse`, and retried (with a
corrective follow-up instruction) up to 2 additional times before returning
a clearly marked error response.

## Example call transcripts (MOCK_LLM left at its default)

**1. Query that triggers retrieval (`policy_question`)**

Request:
```json
{"query": "What is your delivery fee for small orders?"}
```

Response (`classify_intent` -> `policy_question` via the `delivery`
keyword; `retrieve_and_answer` retrieves `doc_01` as the top chunk):
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard del",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

**2. Query that does not trigger retrieval (`general_question`)**

Request:
```json
{"query": "What is the capital of France?"}
```

Response (`classify_intent` -> `general_question`, no policy keyword
present; routed straight to `direct_answer`, no retrieval, no LLM call):
```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

*(These transcripts were produced by running `python demo.py` / the
corresponding `curl` calls against the endpoint with `MOCK_LLM` left
unset. `sources` for query 1 lists the single top-retrieved chunk ID;
`retrieve_top_k` returns up to 3, and all returned chunk IDs are included
in `sources` when more than one chunk is retrieved.)*
