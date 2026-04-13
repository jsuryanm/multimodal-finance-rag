from __future__ import annotations
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, START, END

from src.agents.state import FinanceAgentState
from src.agents.summary_agent import SummaryAgent
from src.agents.chart_agent import ChartAgent
from src.agents.comparision_agent import ComparsionAgent
from src.agents.stock_agent import StockAgent

from src.memory.checkpoint import get_checkpointer
from src.memory.long_term import get_long_term_memory

from src.models.schemas import RouterDecision

from src.settings.config import get_llm, settings
from src.logger.custom_logger import logger
from src.exceptions.custom_exceptions import OrchestratorError


class OrchestratorAgent:

    def __init__(self):
        self.llm = get_llm()
        self.summary_agent = SummaryAgent()
        self.chart_agent = ChartAgent()
        self.comparision_agent = ComparsionAgent()
        self.stock_agent = None
        self.long_term_memory = get_long_term_memory()
        self._mcp_client: MultiServerMCPClient | None = None
        self._app = None

    async def _connect_mcp(self) -> None:
        self._mcp_client = MultiServerMCPClient({
            "finance-tools": {
                "command": "python",
                "args": ["-m", "src.mcp_server.server"],
                "transport": "stdio",
            }
        })
        tools_list = await self._mcp_client.get_tools()
        mcp_tools = {tool.name: tool for tool in tools_list}
        self.stock_agent = StockAgent(mcp_tools)
        logger.info(f"MCP server connected, Tools loaded: {list(mcp_tools.keys())}")


    async def _load_memory_node(self, state: FinanceAgentState) -> dict:
        summary = await self.long_term_memory.get_memory(state["session_id"])
        if summary:
            logger.info(f"Loaded long-term memory for session {state['session_id']}")
        else:
            logger.info(f"No prior memory for session={state['session_id']}")
        return {"long_term_summary": summary or ""}

    async def _route_node(self, state: FinanceAgentState) -> dict:
        try:
            structured_llm = self.llm.with_structured_output(RouterDecision)
            system = (
                "You are a financial query classifier. Classify the user's LATEST message into "
                "exactly one route. Use conversation history only to resolve context.\n\n"
                "Routes (use EXACT spelling — these are code identifiers):\n"
                "- stock_price: user asks about current/live/recent stock price or share price\n"
                "- comparision: user explicitly wants to COMPARE two companies "
                "(keywords: compare, versus, vs, side by side, both companies)\n"
                "- chart: user asks about a chart, graph, table, figure, or a specific page\n"
                "- summary: all other financial questions about the report\n\n"
                "Priority: stock_price > comparision > chart > summary\n"
            )
            if state.get("long_term_summary"):
                system += f"\nPrevious conversation context:\n{state['long_term_summary']}"

            messages = [SystemMessage(content=system)] + list(state["messages"])
            decision: RouterDecision = await structured_llm.ainvoke(messages)
            logger.info(f"Route: '{decision.route}' | Reason: {decision.reasoning}")
            return {"route": decision.route}
        except Exception as e:
            raise OrchestratorError("Router LLM call failed", detail=str(e))

    async def _summary_node(self, state: FinanceAgentState) -> dict:
        try:
            result = await asyncio.wait_for(
                self.summary_agent.run(state), timeout=settings.LLM_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"SummaryAgent timed out after {settings.LLM_TIMEOUT}s")
        return {
            "answer": result.get("answer", ""),
            "documents": result.get("documents"),
            "structured_responses": result.get("structured_responses"),
            "messages": result.get("messages", []),
        }

    async def _chart_node(self, state: FinanceAgentState) -> dict:
        try:
            result = await asyncio.wait_for(
                self.chart_agent.run(state), timeout=settings.LLM_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"ChartAgent timed out after {settings.LLM_TIMEOUT}s")
        return {
            "answer": result.get("answer", ""),
            "image_b64": result.get("image_b64"),
            "structured_responses": result.get("structured_responses"),
            "messages": result.get("messages", []),
        }

    async def _comparision_node(self, state: FinanceAgentState) -> dict:
        try:
            result = await asyncio.wait_for(
                self.comparision_agent.run(state), timeout=settings.LLM_TIMEOUT
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"ComparsionAgent timed out after {settings.LLM_TIMEOUT}s")
        return {
            "answer": result.get("answer", ""),
            "structured_responses": result.get("structured_responses"),
            "messages": result.get("messages", []),
        }

    async def _stock_price_node(self, state: FinanceAgentState) -> dict:
        """Delegate to StockAgent."""
        try:
            result = await asyncio.wait_for(
                self.stock_agent.run(state), timeout=settings.LLM_TIMEOUT
            )
        except asyncio.TimeoutError:
            answer = f"StockAgent timed out after {settings.LLM_TIMEOUT}s"
            return {"answer": answer, "messages": [AIMessage(content=answer)]}
        return {
            "answer": result.get("answer", ""),
            "structured_responses": result.get("structured_responses"),
            "messages": result.get("messages", []),
        }

    async def _save_memory_node(self, state: FinanceAgentState) -> dict:
        question = state.get("question", "")
        answer = state.get("answer", "")
        if not answer:
            return {}
        existing = state.get("long_term_summary", "")
        updated = f"{existing}\n\nQ: {question}\nA: {answer[:400]}".strip()
        if len(updated) > 3000:
            updated = updated[-3000:]
        await self.long_term_memory.save_memory(state["session_id"], updated)
        logger.info(f"Saved long-term memory for session: {state['session_id']}")
        return {}


    def _decide_route(self, state: FinanceAgentState) -> str:
        """Apply override rules on top of the LLM's route decision."""
        route = state.get("route", "summary")

        if state.get("session_id_b"):
            logger.info("session_id_b detected — forcing comparision route")
            return "comparision"

        if route == "comparision" and not state.get("session_id_b"):
            logger.warning("'comparision' selected but session_id_b missing — falling back to summary")
            return "summary"

        if route not in {"summary", "chart", "comparision", "stock_price"}:
            logger.warning(f"Unknown route '{route}' — defaulting to summary")
            return "summary"

        return route


    async def build_graph(self) -> None:
        """Connect to MCP and compile the LangGraph StateGraph. Call once at startup."""
        await self._connect_mcp()
        await self.long_term_memory.setup()
        self._checkpointer_cm, self.checkpointer = await get_checkpointer()

        graph = StateGraph(FinanceAgentState)

        graph.add_node("load_memory", self._load_memory_node)
        graph.add_node("route", self._route_node)
        graph.add_node("summary", self._summary_node)
        graph.add_node("chart", self._chart_node)
        graph.add_node("comparision", self._comparision_node)
        graph.add_node("stock_price", self._stock_price_node)
        graph.add_node("save_memory", self._save_memory_node)

        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "route")
        graph.add_conditional_edges(
            "route",
            self._decide_route,
            {
                "summary": "summary",
                "chart": "chart",
                "comparision": "comparision",
                "stock_price": "stock_price",
            },
        )
        for agent_node in ["summary", "chart", "comparision", "stock_price"]:
            graph.add_edge(agent_node, "save_memory")
        graph.add_edge("save_memory", END)

        self._app = graph.compile(checkpointer=self.checkpointer)
        logger.info("OrchestratorAgent graph compiled successfully")

    async def run(
        self,
        question: str,
        session_id: str,
        session_id_b: str | None = None,
        page_number: int | None = None,
        thread_id: str | None = None,
    ) -> dict:
        if self._app is None:
            await self.build_graph()

        config = {"configurable": {"thread_id": thread_id or session_id}}
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "session_id": session_id,
            "session_id_b": session_id_b,
            "question": question,
            "page_number": page_number,
        }
        result = await self._app.ainvoke(initial_state, config=config)
        return {
            "answer": result.get("answer", ""),
            "route": result.get("route", ""),
            "session_id": session_id,
        }

    async def stream(
        self,
        question: str,
        session_id: str,
        session_id_b: str | None = None,
        page_number: int | None = None,
        thread_id: str | None = None,
    ):
        if self._app is None:
            await self.build_graph()

        config = {"configurable": {"thread_id": thread_id or session_id}}
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "session_id": session_id,
            "session_id_b": session_id_b,
            "question": question,
            "page_number": page_number,
        }
        try:
            result = await self._app.ainvoke(initial_state, config=config)
            route = result.get("route", "summary")
            answer = result.get("answer", "")
            yield f"[ROUTE:{route}]"
            yield answer
        except Exception as e:
            logger.error(f"Orchestrator stream failed: {e} | detail: {getattr(e, 'detail', '')}")
            yield f"[ERROR] {e}"



_orchestrator: OrchestratorAgent | None = None


async def get_orchestrator() -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
        await _orchestrator.build_graph()
    return _orchestrator
