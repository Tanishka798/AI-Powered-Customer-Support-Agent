import pytest

from src.rag import retriever as retriever_module
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError
from tests.fakes import KeywordEmbeddings, make_settings


@pytest.fixture
def retriever(tmp_path, knowledge_dir, monkeypatch) -> KnowledgeRetriever:
    monkeypatch.setattr(retriever_module, "build_embeddings", lambda settings: KeywordEmbeddings())
    return KnowledgeRetriever(make_settings(vector_db_path=str(tmp_path / "vector_db")), knowledge_dir)


@pytest.mark.asyncio
async def test_search_before_initialize_raises(retriever) -> None:
    with pytest.raises(ComponentNotReadyError):
        await retriever.search("How long does shipping take?")


@pytest.mark.asyncio
async def test_relevant_question_returns_matching_source(retriever) -> None:
    await retriever.initialize()

    results = await retriever.search("How long does shipping take?")

    assert results
    assert results[0]["source"] == "shipping.md"
    assert set(results[0]) == {"content", "source"}


@pytest.mark.asyncio
async def test_irrelevant_question_returns_nothing(retriever) -> None:
    await retriever.initialize()

    assert await retriever.search("Do you sell gift vouchers?") == []


@pytest.mark.asyncio
async def test_blank_query_returns_nothing(retriever) -> None:
    await retriever.initialize()

    assert await retriever.search("   ") == []


@pytest.mark.asyncio
async def test_explicit_limit_caps_results(retriever) -> None:
    await retriever.initialize()

    assert len(await retriever.search("refund for a card charge", limit=1)) == 1
    assert await retriever.search("refund for a card charge", limit=0) == []


@pytest.mark.asyncio
async def test_sources_are_unique_filenames_only(retriever) -> None:
    await retriever.initialize()

    results = await retriever.search("refund for a card charge or delivery")

    pairs = [(item["content"], item["source"]) for item in results]
    assert len(pairs) == len(set(pairs))
    assert all("/" not in item["source"] and item["source"].endswith(".md") for item in results)


@pytest.mark.asyncio
async def test_reinitialize_does_not_duplicate_chunks(retriever) -> None:
    await retriever.initialize()
    first_count = retriever._store._collection.count()

    await retriever.initialize()

    assert retriever._store._collection.count() == first_count


@pytest.mark.asyncio
async def test_failed_initialize_leaves_retriever_not_ready(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(retriever_module, "build_embeddings", lambda settings: KeywordEmbeddings())
    broken = KnowledgeRetriever(make_settings(vector_db_path=str(tmp_path / "db")), tmp_path / "missing")

    with pytest.raises(RuntimeError):
        await broken.initialize()
    with pytest.raises(ComponentNotReadyError):
        await broken.search("shipping")
