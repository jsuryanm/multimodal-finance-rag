from __future__ import annotations 

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import FinanceAgentState
from src.models.schemas import ComparisionSummary
from src.core.vector_store import VectorStore
from src.prompts.comparision_prompt import COMPARE_SYSTEM_PROMPT
from src.exceptions.custom_exceptions import AgentError
from src.logger.custom_logger import logger 

class ComparisionAgent: