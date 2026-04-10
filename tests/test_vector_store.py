from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.documents import Document


@pytest.fixture
def mock_embeddings():
    emb = MagicMock()
    emb.embed_documents.return_value = [[0.1, 0.2, 0.3, 0.4]] * 10
    emb.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return emb


@pytest.fixture
def sample_docs():
    return [
        Document(page_content="Revenue increased to 10 million dollars", metadata={"page": 1}),
        Document(page_content="Net profit was 2 million dollars", metadata={"page": 1}),
        Document(page_content="Total debt reduced to 5 million", metadata={"page": 2}),
        Document(page_content="Operating cash flow improved significantly", metadata={"page": 3}),
        Document(page_content="Earnings per share rose to 1.50 dollars", metadata={"page": 3}),
    ]


def test_build_and_get_retriever(tmp_path, mock_embeddings, sample_docs):
    """build_index persists data; get_retriever returns a working retriever."""
    with patch("src.core.vector_store.get_qwen_embeddings", return_value=mock_embeddings), \
         patch("src.core.vector_store.settings") as mock_settings:
        mock_settings.CHROMA_DIR = tmp_path
        mock_settings.RETRIEVER_K = 3

        from src.core.vector_store import VectorStore
        store = VectorStore(session_id="test-session-001")
        store.build_index(sample_docs)

        retriever = store.get_retriever(search_type="similarity", k=3)
        assert retriever is not None


def test_build_index_filters_short_chunks(tmp_path, mock_embeddings):
    """build_index raises VectorStoreError when all chunks are too short."""
    from src.exceptions.custom_exceptions import VectorStoreError

    short_docs = [Document(page_content="hi", metadata={})]
    with patch("src.core.vector_store.get_qwen_embeddings", return_value=mock_embeddings), \
         patch("src.core.vector_store.settings") as mock_settings:
        mock_settings.CHROMA_DIR = tmp_path
        mock_settings.RETRIEVER_K = 3

        from src.core.vector_store import VectorStore
        store = VectorStore(session_id="test-session-002")
        with pytest.raises(VectorStoreError, match="too short"):
            store.build_index(short_docs)


def test_load_index_missing_raises(tmp_path, mock_embeddings):
    """load_index raises VectorStoreError when no persisted index exists."""
    from src.exceptions.custom_exceptions import VectorStoreError

    with patch("src.core.vector_store.get_qwen_embeddings", return_value=mock_embeddings), \
         patch("src.core.vector_store.settings") as mock_settings:
        mock_settings.CHROMA_DIR = tmp_path
        mock_settings.RETRIEVER_K = 3

        from src.core.vector_store import VectorStore
        store = VectorStore(session_id="nonexistent-session")
        with pytest.raises(VectorStoreError, match="No ChromaDB index"):
            store.load_index()
