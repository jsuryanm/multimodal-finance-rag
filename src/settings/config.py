from pydantic_settings import SettingsConfigDict,BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env",
                                      env_file_encoding="utf-8",
                                      case_sensitive=False,
                                      extra="ignore")
    
    # API keys for llms and zina embeddings 
    OPENAI_API_KEY: str = Field(default="")
    GROQ_API_KEY: str = Field(default="")
    JINA_API_KEY: str = Field(default="")

    # LangSmith
    LANGSMITH_API_KEY: str = Field(default="")
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_PROJECT: str = "financial-doc-analyzer"

    # LLM settings 
    LLM_PROVIDER: str = "openai"
    OPENAI_LLM: str = "gpt-4o-mini"
    OPENAI_VISION_MODEL: str = "gpt-4o-mini"
    
    GROQ_LLM: str = "llama-3.3-70b-versatile"
    GROQ_VISION_MODEL: str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # tavily api 
    TAVILY_API_KEY: str = Field(default="")


    # Embeddings model 
    EMBEDDING_MODEL: str = "jina-embeddings-v3"

    # Maximum completion tokens
    MAX_TOKENS: int = 500

    # RAG dettings 
    CHUNK_SIZE: int = 1000 
    CHUNK_OVERLAP: int = 200 
    RETRIEVER_K: int = 5 

    # paths
    DATA_DIR: Path = Path("data")
    FAISS_DIR: Path = Path("faiss_index")
    SQLITE_MEMORY_DB: Path = Path("data/memory.db")
    SQLITE_CHECKPOINT: Path = Path("data/checkpoints.db")


    # logging
    LOG_LEVEL: str = "INFO"
    

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

@lru_cache(maxsize=1)
def get_llm():
    """Return text LLM based on provider settings"""
    settings = get_settings()

    if settings.LLM_PROVIDER == "groq": 
        llm = ChatGroq(model=settings.GROQ_LLM,
                       api_key=settings.GROQ_API_KEY,
                       max_tokens=settings.MAX_TOKENS)
        return llm
    
    else:
        llm = ChatOpenAI(model=settings.OPENAI_LLM,
                         api_key=settings.OPENAI_API_KEY,
                         max_completion_tokens=settings.MAX_TOKENS)
        return llm 

@lru_cache(maxsize=1)
def get_vision_llm():
    """Return a vision-capable LLM for chart/image analysis."""
    settings = get_settings()
    if settings.LLM_PROVIDER == "groq":
        vision_llm = ChatGroq(model=settings.GROQ_VISION_MODEL,
                              api_key=settings.GROQ_API_KEY)
        return vision_llm
    
    else:
        vision_llm = ChatOpenAI(model=settings.OPENAI_VISION_MODEL,
                                api_key=settings.OPENAI_API_KEY)
        return vision_llm

settings = get_settings()