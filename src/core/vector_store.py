from __future__ import annotations
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import filter_complex_metadata

from src.core.embeddings import get_qwen_embeddings
from src.exceptions.custom_exceptions import VectorStoreError
from src.logger.custom_logger import logger
from src.settings.config import settings


class VectorStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.index_dir = settings.CHROMA_DIR / session_id
        self._embeddings = get_qwen_embeddings()
        self._store: Chroma | None = None

    def build_index(self, documents: list[Document]) -> None:
        if not documents:
            raise VectorStoreError("No documents provided for indexing")
        
        documents = filter_complex_metadata(documents)

        filtered = [d for d in documents if len(d.page_content.strip()) > 50]
        if not filtered:
            raise VectorStoreError(
                "All extracted chunks were too short to index",
                detail="The PDF may contain only images or headers. Try OCR pre-processing.",
            )
        if len(filtered) < len(documents):
            logger.info(
                f"Filtered {len(documents) - len(filtered)} short chunks; "
                f"indexing {len(filtered)}"
            )

        self.index_dir.mkdir(parents=True, exist_ok=True)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                self._store = Chroma.from_documents(
                    documents=filtered,
                    embedding=self._embeddings,
                    persist_directory=str(self.index_dir),
                    collection_name=self.session_id,
                )
                break
            except Exception as e:
                if attempt == max_attempts:
                    raise VectorStoreError(
                        "Failed to build ChromaDB index", detail=str(e)
                    ) from e
                wait = 2 ** attempt
                logger.warning(
                    f"Embedding attempt {attempt}/{max_attempts} failed: {e}. "
                    f"Retrying in {wait}s…"
                )
                time.sleep(wait)

        logger.info(
            f"Built ChromaDB index with {len(filtered)} chunks: {self.index_dir}"
        )

    def load_index(self) -> None:
        """Load a previously saved ChromaDB collection from disk."""
        if not self.index_dir.exists():
            raise VectorStoreError(
                f"No ChromaDB index found for session: {self.session_id}",
                detail=f"Expected path: {self.index_dir}",
            )

        try:
            self._store = Chroma(
                persist_directory=str(self.index_dir),
                embedding_function=self._embeddings,
                collection_name=self.session_id,
            )
            logger.info(f"Loaded ChromaDB index from {self.index_dir}")

        except Exception as e:
            raise VectorStoreError("Failed to load ChromaDB index", detail=str(e)) from e

    def get_retriever(
        self,
        search_type: str = "mmr",
        k: int | None = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
    ):
        if self._store is None:
            self.load_index()

        search_kwargs: dict = {"k": k or settings.RETRIEVER_K}

        if search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k
            search_kwargs["lambda_mult"] = lambda_mult

        return self._store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )
