from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from src.agents.state import FinanceAgentState
from src.settings.config import get_llm
from src.logger.custom_logger import logger


class StockAgent:
    """
    ReAct-style agent that uses MCP tools to fetch live stock data.
    MCP tools from langchain_mcp_adapters are already LangChain BaseTool
    instances with proper JSON schemas — pass them directly to create_agent.
    """

    def __init__(self, mcp_tools: dict):
        self.llm = get_llm()
        tools = list(mcp_tools.values())
        self.agent = create_agent(
            model=self.llm,
            tools=tools,
            system_prompt=(
                "You are a financial assistant.\n"
                "Use tools to fetch real-time financial data.\n"
                "If the user asks for a stock price, ALWAYS call get_stock_price.\n"
                "If the user asks for news, call search_financial_news.\n"
                "Do not hallucinate financial data.\n"
            ),
        )

    # src/agents/stock_agent.py

    async def run(self, state: FinanceAgentState) -> dict:
        question = state["question"]
        try:
            result = await self.agent.ainvoke({
                "messages": [{"role": "user", "content": question}]
            })
            # AIMessage is an object — use .content, not ["content"]
            last_message = result["messages"][-1]
            answer = last_message.content if hasattr(last_message, "content") else str(last_message)
            
            logger.info("StockAgent completed")
            return {
                "answer": answer,
                "structured_responses": {"raw": answer},
                "messages": [AIMessage(content=answer)],
                "route": "stock_price",
            }
        except Exception as e:
            logger.error(f"StockAgent failed: {e}")
            return {
                "answer": "Failed to retrieve stock data.",
                "structured_responses": {"error": str(e)},
                "messages": [AIMessage(content="Failed to retrieve stock data.")],
                "route": "stock_price",
            }
