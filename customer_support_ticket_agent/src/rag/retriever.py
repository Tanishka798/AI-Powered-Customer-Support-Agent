from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError


class KnowledgeRetriever:
    """Persistent Chroma retrieval service for the support knowledge base.

    This is the single place that sanitizes and deduplicates retrieved
    chunks: results contain only non-empty content and filename-only sources.
    """

    # Chroma's similarity_search_with_score returns a distance: lower is more similar.
    RELEVANCE_THRESHOLD = 1.0

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    @staticmethod
    def _document_id(content: str, source: str) -> str:
        """Create a stable ID for a source chunk."""
        return hashlib.sha256(f"{source}\n{content}".encode("utf-8")).hexdigest()

    async def initialize(self) -> None:
        """Load, split, embed, and persist the knowledge base."""
        try:
            store = self._open_store()
            self._index_new_documents(store, self._prepare_documents())
        except Exception as exc:
            self._store = None
            raise RuntimeError(f"Failed to initialize support knowledge base: {exc}") from exc
        # Only expose the store after initialization succeeds.
        self._store = store

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        """Retrieve relevant, deduplicated knowledge chunks as plain dictionaries."""
        if not query or not query.strip():
            return []
        if self._store is None:
            raise ComponentNotReadyError("Knowledge retriever is not initialized")

        requested_limit = self.settings.rag_top_k if limit is None else limit
        if requested_limit <= 0:
            return []

        # Fetch extra candidates so irrelevant/duplicate chunks can be dropped.
        results = self._store.similarity_search_with_score(query.strip(), k=requested_limit * 2)
        return self._relevant_unique_chunks(results, requested_limit)

    def _prepare_documents(self) -> dict[str, Document]:
        """Split documents keyed by stable ID, with filename-only source metadata."""
        documents_by_id: dict[str, Document] = {}
        for document in split_support_documents(load_support_documents(self.documents_dir)):
            source = _filename(document.metadata.get("source"))
            content = document.page_content.strip()
            if source and content:
                document.metadata = {"source": source}
                documents_by_id[self._document_id(content, source)] = document
        return documents_by_id

    def _open_store(self) -> Chroma:
        vector_db_path = Path(self.settings.vector_db_path)
        vector_db_path.mkdir(parents=True, exist_ok=True)
        return Chroma(
            collection_name=self.settings.rag_collection,
            embedding_function=build_embeddings(self.settings),
            persist_directory=str(vector_db_path),
        )

    @staticmethod
    def _index_new_documents(store: Chroma, documents_by_id: dict[str, Document]) -> None:
        """Add only chunks not already stored, so restarts don't duplicate them."""
        if not documents_by_id:
            return
        existing_ids = set(store.get(ids=list(documents_by_id)).get("ids", []))
        new_ids = [doc_id for doc_id in documents_by_id if doc_id not in existing_ids]
        if new_ids:
            store.add_documents(documents=[documents_by_id[doc_id] for doc_id in new_ids], ids=new_ids)

    def _relevant_unique_chunks(
        self, results: list[tuple[Document, float]], limit: int
    ) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for document, distance in results:
            content = document.page_content.strip()
            source = _filename(document.metadata.get("source"))
            if distance > self.RELEVANCE_THRESHOLD or not content or not source or (content, source) in seen:
                continue
            seen.add((content, source))
            chunks.append({"content": content, "source": source})
            if len(chunks) >= limit:
                break
        return chunks


def _filename(source: object) -> str:
    """Only expose the filename, never an absolute filesystem path."""
    return Path(str(source or "").strip()).name
