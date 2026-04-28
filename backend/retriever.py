"""
retriever.py — RAG Step 4
- Dynamic synonym extraction from document using LLM
- BGE-compatible query encoding (query: prefix)
- Cosine similarity search top-5 chunks
"""

import os
from ingest import get_embed_model, get_collection

TOP_K = 5

# Read Ollama URL from environment — falls back to localhost for local dev
OLLAMA_URL   = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = "llama3.2"

# ── Per-document synonym store ────────────────────────────────────
_doc_synonyms: dict[str, dict[str, list[str]]] = {}


def extract_synonyms_from_doc(doc_id: str, sample_text: str, mode: str, api_key: str):
    """
    Ask the LLM to extract domain-specific synonyms from the document.
    Stores result in _doc_synonyms[doc_id].
    Called once per document upload in background — never crashes the app.
    """
    _doc_synonyms[doc_id] = {}

    if not sample_text.strip():
        return

    prompt = f"""Read the following document excerpt and extract domain-specific terms and their synonyms or related expressions used in this document.

Return ONLY a JSON object like this (no explanation, no markdown):
{{
  "term1": ["synonym1", "synonym2"],
  "term2": ["synonym1"],
  "abbreviation": ["full form"]
}}

Focus on:
- Abbreviations and their full forms (e.g. "GDP" → "gross domestic product")
- Domain jargon and plain-language equivalents
- Key metrics referred to by multiple names
- Only include terms actually present in the document

Document excerpt:
{sample_text[:3000]}"""

    try:
        import json

        if mode == "openai" and api_key:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500,
            )
            raw = response.choices[0].message.content.strip()

        elif mode == "gemini" and api_key:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0, max_output_tokens=500)
            )
            raw = response.text.strip()

        elif mode == "local":
            import requests as rq
            # Uses OLLAMA_URL env var — works both locally and in Docker
            response = rq.post(
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0}},
                timeout=60
            )
            raw = response.json().get("response", "{}").strip()

        else:
            return

        raw = raw.replace("```json", "").replace("```", "").strip()
        synonyms = json.loads(raw)
        _doc_synonyms[doc_id] = {
            k.lower(): [s.lower() for s in v]
            for k, v in synonyms.items()
            if isinstance(v, list)
        }
        print(f"[Retriever] Extracted {len(_doc_synonyms[doc_id])} synonym groups for '{doc_id}'")

    except Exception as e:
        print(f"[Retriever] Synonym extraction skipped (non-critical): {e}")
        _doc_synonyms[doc_id] = {}


def expand_query(query: str, doc_id: str | None) -> str:
    """Expand query using synonyms extracted from the specific document."""
    if not doc_id or doc_id not in _doc_synonyms:
        return query

    synonyms = _doc_synonyms[doc_id]
    query_lower = query.lower()
    extra_terms = []

    for term, syns in synonyms.items():
        if term in query_lower:
            extra_terms.extend(syns)
        for syn in syns:
            if syn in query_lower:
                extra_terms.append(term)
                extra_terms.extend([s for s in syns if s != syn])

    unique_extras = list(dict.fromkeys(
        t for t in extra_terms if t not in query_lower
    ))

    if unique_extras:
        expanded = query + " " + " ".join(unique_extras[:5])
        print(f"[Retriever] Expanded: '{query}' → +{unique_extras[:5]}")
        return expanded

    return query


def retrieve(query: str, doc_id: str | None = None, top_k: int = TOP_K) -> list[dict]:
    model      = get_embed_model()
    collection = get_collection()

    expanded_query  = expand_query(query, doc_id)
    query_embedding = model.encode(
        [f"query: {expanded_query}"],
        normalize_embeddings=True
    ).tolist()

    where_filter = {"doc_id": doc_id} if doc_id else None

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text":        doc,
                "chunk_index": meta.get("chunk_index", 0) + 1,
                "filename":    meta.get("filename", "unknown"),
                "file_type":   meta.get("file_type", "unknown"),
                "score":       round(1 - dist, 4),
            })

    print(f"[Retriever] {len(chunks)} chunks retrieved.")
    return chunks


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Source {i} | {chunk['filename']} | Score: {chunk['score']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)