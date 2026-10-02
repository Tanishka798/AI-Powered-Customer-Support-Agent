# RAG tests

Tests for `src/rag/`. Retriever tests use a real Chroma store in a temporary directory with deterministic keyword embeddings (`tests/fakes.py`), so no model download is needed.

| File | What it checks |
|---|---|
| `test_document_loader.py` | Markdown policies load with filename-only `source` metadata, splitting keeps that metadata, and a missing or empty knowledge folder is rejected. |
| `test_knowledge_retriever.py` | Search fails before initialization; relevant questions return the right source; irrelevant or blank questions return nothing; `limit` is respected; results are unique and filename-only; re-initializing doesn't duplicate chunks; a failed initialization leaves the retriever not ready. |
