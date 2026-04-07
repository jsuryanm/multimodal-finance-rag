from __future__ import annotations 

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph,START,END

from src.agents.state import FinanceAgentState
from src.core.vector_store import VectorStore
from src.settings.config import get_llm

class SummaryAgent:
    pass 