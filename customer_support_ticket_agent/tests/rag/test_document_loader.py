import pytest

from src.rag.document_loader import load_support_documents, split_support_documents


def test_loader_preserves_source_names(knowledge_dir) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


def test_loader_rejects_missing_directory(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_support_documents(tmp_path / "missing")


def test_loader_rejects_directory_without_markdown(tmp_path) -> None:
    (tmp_path / "notes.txt").write_text("not a policy", encoding="utf-8")
    with pytest.raises(ValueError):
        load_support_documents(tmp_path)
