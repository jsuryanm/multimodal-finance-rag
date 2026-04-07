from __future__ import annotations 
from pathlib import Path 

from langchain_community.embeddings import JinaEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.documents import Document

from src.settings.config import settings 
from src.logger.custom_logger import logger 
from src.exceptions.custom_exceptions import VectorStoreError

class VectorStore:
    """Manages FAISS index lifecycle: 
    build,save,load and retrieve
    One index per session"""

    def __init__(self,session_id: str):
        self.session_id = session_id
        self.index_dir = settings.FAISS_DIR / session_id
        self._embeddings = self._build_embeddings()
        self._store: FAISS | None = None 

    def _build_embeddings(self) -> JinaEmbeddings:
        """Instantiate Jina embeddings"""
        
        if not settings.JINA_API_KEY:
            raise VectorStoreError("JINA_API_KEY not set",
                                   detail="Add JINA API KEY to .env file")
        
        return JinaEmbeddings(jina_api_key=settings.JINA_API_KEY,
                              model_name=settings.EMBEDDING_MODEL)
    
    def build_index(self,documents: list[Document]) -> None:
        """Create FAISS index from docs and save to disk"""

        if not documents:
            raise VectorStoreError("No documents uploaded for indexing")
        
        try:
            self._store = FAISS.from_documents(documents=documents,
                                               embedding=self._embeddings)
            
            self._store.save_local(str(self.index_dir))
            logger.info(f"Built FAISS index with {len(documents)} chunks: {self.index_dir}")
        except Exception as e:
            raise VectorStoreError("Failed to build FAISS index",detail=str(e))
    
    def load_index(self) -> None:
        """Load a previously saved FAISS index from disk"""
        if not self.index_dir.exists():
            raise VectorStoreError(f"No FAISS index found for session: {self.session_id}",
                                   detail=f"Expected path: {self.index_dir}")
        
        try:
            self._store = FAISS.load_local(str(self.index_dir),
                                           self._embeddings,
                                           allow_dangerous_deserialization=True)
            
            logger.info(f"Loaded FAISS index from {self.index_dir}")
        
        except Exception as e:
            raise VectorStoreError("Failed to load FAISS index",detail=str(e))
        
    def get_retriever(self,
                      search_type: str = "mmr",
                      k: int | None = None,
                      fetch_k: int = 20,
                      lambda_mult: float = 0.5):
        """Return a LangChain retriever from the loaded FAISS store.
        fetch_k: Number of candidates for MMR reranking
        lambda_mult: Diversity vs relevance tradeoff (0=diverse, 1=relevant)
        """
        
        if self._store is None:
            self.load_index()

        search_kwargs = {"k":k or settings.RETRIEVER_K}

        if search_type == "mmr":
            search_kwargs["fetch_k"] = fetch_k
            search_kwargs["lambda_mult"] = lambda_mult
        
        return self._store.as_retriever(search_type=search_type,
                                        search_kwargs=search_kwargs)
    
if __name__ == "__main__":
    docs = [
        Document(page_content="Revenue increased to 10 million dollars"),
        Document(page_content="Net profit was 2 million dollars"),
        Document(page_content="Total debt reduced to 5 million"),
        Document(page_content="Operating cash flow improved")
    ]

    session = "experiment"
    try:
        vector_store = VectorStore(session)
        logger.info("Building index")
        
        vector_store.build_index(docs)
        logger.info("Loading retriever")

        retriever = vector_store.get_retriever()

        query = "What is the revenue?"
        logger.info(f"Running query:{query}")
        
        results = retriever.invoke(query)
        
        print("\nResults:")
        for r in results:
            logger.info(f"{r.page_content}")
        
    except Exception as e:
        logger.exception(f"Error:{str(e)}") 
