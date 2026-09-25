"""
Ingestion pipeline: load the 8 Zepto policy documents, chunk them, embed
each chunk locally with sentence-transformers (all-MiniLM-L6-v2), and store
the embeddings in a persistent ChromaDB collection.

No API key and no network call to any LLM provider are required for this
module -- embeddings run entirely on-device.
"""

import os
from typing import Dict, List

import chromadb
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "zepto_policies"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Lazily load and cache the local embedding model."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def load_documents(docs_dir: str = DOCS_DIR) -> List[Dict[str, str]]:
    """Load each .txt file in docs_dir as one document."""
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith(".txt"):
            continue
        doc_id = os.path.splitext(filename)[0]  # e.g. "doc_01"
        with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
            text = f.read().strip()
        documents.append({"id": doc_id, "text": text})
    return documents


def chunk_document(doc_id: str, text: str, chunk_size: int = 400) -> List[Dict[str, str]]:
    """
    Simple per-document chunking. Each policy document here is a single
    short paragraph, so it is kept as one chunk unless it exceeds
    chunk_size characters, in which case it falls back to fixed-size
    splitting.
    """
    if len(text) <= chunk_size:
        return [{"chunk_id": doc_id, "text": text, "source_doc": doc_id}]

    chunks = []
    for i in range(0, len(text), chunk_size):
        piece = text[i : i + chunk_size]
        chunks.append(
            {
                "chunk_id": f"{doc_id}_chunk{i // chunk_size}",
                "text": piece,
                "source_doc": doc_id,
            }
        )
    return chunks


def build_or_load_collection(persist_directory: str = CHROMA_DIR):
    """
    Build the ChromaDB collection on first run (embedding all 8 documents),
    or return the already-populated collection on subsequent runs.
    """
    client = chromadb.PersistentClient(path=persist_directory)
    # Explicitly configure cosine similarity (Chroma's default is L2).
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        return collection

    model = get_embedding_model()
    documents = load_documents()

    all_chunks: List[Dict[str, str]] = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc["id"], doc["text"]))

    ids = [c["chunk_id"] for c in all_chunks]
    texts = [c["text"] for c in all_chunks]
    metadatas = [{"source_doc": c["source_doc"]} for c in all_chunks]

    embeddings = model.encode(texts, convert_to_numpy=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )
    return collection


def retrieve_top_k(collection, query: str, k: int = 3) -> List[Dict[str, str]]:
    """Embed the query and retrieve the top-k most similar chunks (cosine similarity)."""
    model = get_embedding_model()
    query_embedding = model.encode([query], convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=k)

    retrieved = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    distances = results.get("distances", [[None] * len(ids)])[0]
    for chunk_id, text, distance in zip(ids, docs, distances):
        retrieved.append({"chunk_id": chunk_id, "text": text, "distance": distance})
    return retrieved


if __name__ == "__main__":
    # Standalone check: build the collection and run one sample query.
    col = build_or_load_collection()
    print(f"Collection '{COLLECTION_NAME}' has {col.count()} chunks.")
    sample = retrieve_top_k(col, "What is your delivery fee?", k=3)
    for r in sample:
        print(r["chunk_id"], "->", r["text"][:80])
