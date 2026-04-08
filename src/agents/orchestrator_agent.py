from __future__ import annotations
import sys

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph,START,END

from src.agents.state import FinanceAgentState
from src.agents.summary_agent import SummaryAgent
from src.agents.chart_agent import ChartAgent
from src.agents.comparision_agent import ComparsionAgent

from src.memory.checkpoint import get_checkpointer
from src.memory.long_term import get_long_term_memory
from src.models.schemas import RouterDecision
from src.settings.config import get_llm
from src.logger.custom_logger import logger 
from src.exceptions.custom_exceptions import OrchestratorError

class OrchestratorAgent:
    """
    Main orchestrator that routes user question to correct 
    Graph flow:
        START
          → load_memory    (load previous Q&A context from SQLite)
          → route          (LLM classifies intent)
          → [summary | chart | comparison | stock_price]
          → save_memory    (save Q&A to SQLite for next session)
          → END
    """
    KNOWN_TICKERS = {
        "dbs": "D05.SI",
        "ocbc": "O39.SI",
        "uob": "U11.SI",
        "singtel": "Z74.SI",
        "capitaland": "9CI.SI",
        "keppel": "BN4.SI",
        "grab": "GRAB",
        "sea": "SE",
    }

    def __init__(self):
        self.llm = get_llm()
        self.summary_agent = SummaryAgent()
        self.chart_agent = ChartAgent()
        self.comparision_agent = ComparsionAgent()
        self.long_term_memory = get_long_term_memory()

        self._mcp_client: MultiServerMCPClient | None = None 
        self._mcp_tools: dict = {}
        self._app = None # compiled LangGraph 
    
    async def _connect_mcp(self) -> None:
        """Start MCP server subprocess and load its tools"""
        self._mcp_client  = MultiServerMCPClient({
            "finance-tools":{
                "command":sys.executable,
                "args":["-m","src.mcp_server.server"],
                "transport":"stdio"
            }
        })

        await self._mcp_client.__aenter__() 
        # aenter starts subprocess and establishes stdio connection

        tools_list = self._mcp_client.get_tools()
        self._mcp_tools = {tool.name:tool for tool in tools_list}
        logger.info(f"MCP server connected, Tools loaded: {list(self._mcp_tools.keys())}")