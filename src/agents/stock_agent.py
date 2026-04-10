from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from src.agents.state import FinanceAgentState
from src.settings.config import get_llm
from src.logger.custom_logger import logger
from langchain_core.tools import Tool


def convert_mcp_tools(mcp_tools: dict) -> list[Tool]:
    tools = []

    for tool in mcp_tools.values():
        tools.append(
            Tool(
                name=tool.name,
                description=tool.description,
                func=lambda x, t=tool: t.invoke(x),      # sync fallback
                coroutine=lambda x, t=tool: t.ainvoke(x) # async
            )
        )

    return tools

class StockAgent:
    """
    ReAct-style agent using LangChain v1 create_agent.
    LLM decides when to call MCP tools.
    """

    def __init__(self, mcp_tools: dict):
        self.llm = get_llm()

        # Convert MCP → LangChain tools
        self.tools = convert_mcp_tools(mcp_tools)

        # Create agent
        self.agent = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=(
                "You are a financial assistant.\n"
                "Use tools to fetch real-time financial data.\n"
                "If user asks for stock price, ALWAYS call get_stock_price.\n"
                "If user asks for news, call search_financial_news.\n"
                "Do not hallucinate financial data.\n"
            ),
        )

    async def run(self, state: FinanceAgentState) -> dict:
        question = state["question"]

        try:
            result = await self.agent.ainvoke({
                "messages": [{"role": "user", "content": question}]
            })

            # Extract final response
            answer = result["messages"][-1]["content"]

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