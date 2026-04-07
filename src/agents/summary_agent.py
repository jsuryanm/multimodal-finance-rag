from __future__ import annotations 

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langgraph.graph import StateGraph,START,END

from src.agents.state import FinanceAgentState
from src.prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT
from src.core.vector_store import VectorStore
from src.settings.config import get_llm

class SummaryAgent:
    
    def __init__(self):
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",SUMMARY_SYSTEM_PROMPT),
            ("messages","{messages}")
        ])

        self.app = self.build_graph()

    def retrieve(self,state: FinanceAgentState):
        question = state["messages"][-1].content 

        store = VectorStore(store)