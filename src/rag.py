"""Playbook RAG.

The playbook is a markdown file split by H2 sections, where each section is
a 'position' on a particular clause type (e.g. limitation of liability,
governing law, indemnification). Sections are embedded with a local sentence
transformer and retrieved by semantic similarity to a clause being reviewed.

This is the 'retrieval-augmented generation architecture' from the job
description, scoped down to something runnable on a laptop with no external
dependencies.
"""

from __future__ import annotations
import re
import hashlib
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config import CHROMA_DIR, PLAYBOOK_DIR
from src.audit import log_event


_COLLECTION_NAME = "playbook"


def _client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _embedder():
    # all-MiniLM-L6-v2 is small (~80MB), CPU-friendly, and good enough for
    # semantic clause retrieval. Swap to a legal-domain model in production.
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


def _parse_playbook(md: str) -> list[dict[str, str]]:
    """Split a markdown playbook into sections keyed by H2 headings."""
    sections: list[dict[str, str]] = []
    current_title: str | None = None
    current_body: list[str] = []

    for line in md.splitlines():
        if line.startswith("## "):
            if current_title is not None:
                sections.append(
                    {"title": current_title, "body": "\n".join(current_body).strip()}
                )
            current_title = line[3:].strip()
            current_body = []
        elif current_title is not None:
            current_body.append(line)

    if current_title is not None:
        sections.append(
            {"title": current_title, "body": "\n".join(current_body).strip()}
        )

    return [s for s in sections if s["body"]]


def index_playbook(playbook_path: Path | None = None) -> int:
    """(Re)build the playbook index. Returns the number of sections indexed."""
    if playbook_path is None:
        playbook_path = PLAYBOOK_DIR / "playbook.md"

    md = playbook_path.read_text(encoding="utf-8")
    sections = _parse_playbook(md)

    client = _client()
    # Reset collection so re-indexing is idempotent
    try:
        client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_embedder(),
    )

    ids = [
        hashlib.md5(s["title"].encode()).hexdigest() for s in sections
    ]
    documents = [f"{s['title']}\n\n{s['body']}" for s in sections]
    metadatas = [{"title": s["title"]} for s in sections]

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(sections)


def retrieve_playbook_position(
    *, run_id: str, clause_text: str, clause_type: str, top_k: int = 2
) -> list[dict[str, Any]]:
    """Retrieve the top-k playbook positions most relevant to a clause."""
    client = _client()
    try:
        collection = client.get_collection(
            name=_COLLECTION_NAME,
            embedding_function=_embedder(),
        )
    except Exception:
        # Auto-build if missing
        index_playbook()
        collection = client.get_collection(
            name=_COLLECTION_NAME,
            embedding_function=_embedder(),
        )

    query = f"Clause type: {clause_type}\n\n{clause_text}"
    results = collection.query(query_texts=[query], n_results=top_k)

    matches = []
    for i in range(len(results["ids"][0])):
        matches.append(
            {
                "id": results["ids"][0][i],
                "title": results["metadatas"][0][i]["title"],
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else None,
            }
        )

    log_event(
        run_id=run_id,
        stage="playbook",
        event_type="retrieval",
        payload={
            "query": query,
            "clause_type": clause_type,
            "top_k": top_k,
            "matches": matches,
        },
    )
    return matches
