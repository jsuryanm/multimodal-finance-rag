from __future__ import annotations
import asyncio
import json
import uuid

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph,START,END

from src.agents.state import FinanceAgentState
from src.agents.summary_agent import SummaryAgent
from src.agents.chart_agent import ChartAgent
from src.agents.comparision_agent import ComparsionAgent
from src.agents.stock_agent  import StockAgent

from src.memory.checkpoint import get_checkpointer
from src.memory.long_term import get_long_term_memory

from src.models.schemas import RouterDecision

from src.settings.config import get_llm, settings
from src.logger.custom_logger import logger 
from src.exceptions.custom_exceptions import OrchestratorError

_SECTION_ORDER: dict[str, int] = {
    "stock_price": 0,
    "summary": 1,
    "chart": 2,
    "comparision": 3,
}

_SECTION_LABEL: dict[str, str] = {
    "stock_price": "📈 Stock Price",
    "summary":     "📄 Summary",
    "chart":       "🖼 Chart Analysis",
    "comparision": "⚖️ Comparison",
}


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

    def __init__(self):
        self.llm = get_llm()
        self.summary_agent = SummaryAgent()
        self.chart_agent = ChartAgent()
        self.comparision_agent = ComparsionAgent()
        self.stock_agent = None

        self.long_term_memory = get_long_term_memory()


        self._mcp_client: MultiServerMCPClient | None = None 
        self._mcp_tools: dict = {}
        self._app = None # compiled LangGraph 
    
    async def _connect_mcp(self) -> None:
        """Start MCP server subprocess and load its tools"""
        self._mcp_client  = MultiServerMCPClient({
            "finance-tools":{
                "command":"python",
                "args":["-m","src.mcp_server.server"],
                "transport":"stdio"
            }
        })


        tools_list = await self._mcp_client.get_tools()
        self._mcp_tools = {tool.name:tool for tool in tools_list}
        self.stock_agent = StockAgent(self._mcp_tools)
        logger.info(f"MCP server connected, Tools loaded: {list(self._mcp_tools.keys())}")


    def _format_stock_response(self, result: dict) -> str:
        """
        Format the dict returned by the MCP get_stock_price tool
        into a readable multi-line string.
        """
        if "error" in result:
            return f"Could not fetch stock price: {result['error']}"

        currency = result.get("currency", "SGD")
        price = result.get("price", 0)

        change = result.get("change_percent")
        change_str = f" ({change:+.2f}%)" if change is not None else ""

        lines = [
            f"{result.get('company', result.get('ticker', ''))}",
            f"Price: {currency} {price:.2f}{change_str}",
        ]

        if result.get("market_cap"):
            cap = result["market_cap"]
            cap_str = f"{cap / 1e9:.1f}B" if cap >= 1e9 else f"{cap / 1e6:.1f}M"
            lines.append(f"Market Cap: {currency} {cap_str}")

        if result.get("pe_ratio"):
            lines.append(f"P/E Ratio: {result['pe_ratio']:.1f}x")

        if result.get("dividend_yield"):
            lines.append(f"Dividend Yield: {result['dividend_yield'] * 100:.2f}%")

        if result.get("week_52_high") and result.get("week_52_low"):
            lines.append(
                f"52-Week Range: {result['week_52_low']:.2f} – {result['week_52_high']:.2f}"
            )

        # Bug 4 fix: join list into a string before returning
        return "\n".join(lines)
    
    async def _load_memory_node(self,state: FinanceAgentState) -> dict:
        """
        Load the long-term memory summary from SQLite.
        Gives the agent context from previous sessions for follow-up questions.
        """
        session_id = state['session_id']
        summary = await self.long_term_memory.get_memory(session_id)

        if summary:
            logger.info(f"Loaded long term memory for session:{session_id}")
        else:
            logger.info(f"No prior memory for session={session_id}")

        return {"long_term_summary":summary or ""}
    
    async def _route_node(self, state: FinanceAgentState) -> dict:
        """
        Use the LLM with structured output to classify the user's question.
        """
        try:
            structured_llm = self.llm.with_structured_output(RouterDecision)
            system_content = (
                "You are a financial query classifier. Read the user's LATEST message and classify "
                "it into exactly one of the four routes below. Use conversation history only to "
                "resolve context — the classification must be based on the LATEST message.\n\n"
                "Routes (use EXACT spelling — these are code identifiers):\n"
                "- stock_price: user explicitly asks about current/live/recent stock PRICE, "
                "share price, or trading price of a company (e.g. 'What is DBS stock price?', "
                "'How is D05.SI trading?'). Do NOT use this route for questions about financial "
                "metrics, earnings, or fundamentals just because a company name is mentioned.\n"
                "- comparision: user explicitly wants to COMPARE two companies — keywords: compare, "
                "versus, vs, side by side, both companies\n"
                "- chart: user asks about a specific page, or mentions chart, graph, table, figure, "
                "or any visual element from the report\n"
                "- summary: all other financial questions about the report (metrics, risks, "
                "strategy, dividends, earnings, performance)\n\n"
                "Priority when multiple routes apply: stock_price > comparision > chart > summary\n"
            )

            if state.get("long_term_summary"):
                system_content += f"\nPrevious conversation context:\n{state['long_term_summary']}"

            messages = [SystemMessage(content=system_content)] + list(state["messages"])

            decision: RouterDecision = await structured_llm.ainvoke(messages)
            logger.info(f"Route: '{decision.route}' | Reason: {decision.reasoning}")
            return {"route": decision.route}

        except Exception as e:
            raise OrchestratorError("Router LLM call failed", detail=str(e))
    
    async def _summary_node(self, state: FinanceAgentState) -> dict:
        """Delegate to SummaryAgent and wrap result in partial_answers."""
        try:
            result = await asyncio.wait_for(
                self.summary_agent.run(state),
                timeout=settings.LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"SummaryAgent timed out after {settings.LLM_TIMEOUT}s")
        return {
            "partial_answers": [{"route": "summary", "text": result.get("answer", "")}],
            "active_routes": ["summary"],
            "documents": result.get("documents"),
            "structured_responses": result.get("structured_responses"),
            "messages": result.get("messages", []),
        }

    async def _chart_node(self, state: FinanceAgentState) -> dict:
        """Delegate to ChartAgent and wrap result in partial_answers."""
        try:
            result = await asyncio.wait_for(
                self.chart_agent.run(state),
                timeout=settings.LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"ChartAgent timed out after {settings.LLM_TIMEOUT}s")
        return {
            "partial_answers": [{"route": "chart", "text": result.get("answer", "")}],
            "active_routes": ["chart"],
            "image_b64": result.get("image_b64"),
            "structured_responses": result.get("structured_responses"),
            "messages": result.get("messages", []),
        }

    async def _comparision_node(self, state: FinanceAgentState) -> dict:
        """Delegate to ComparsionAgent and wrap result in partial_answers."""
        try:
            result = await asyncio.wait_for(
                self.comparision_agent.run(state),
                timeout=settings.LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"ComparsionAgent timed out after {settings.LLM_TIMEOUT}s")
        return {
            "partial_answers": [{"route": "comparision", "text": result.get("answer", "")}],
            "active_routes": ["comparision"],
            "docs_a": result.get("docs_a"),
            "docs_b": result.get("docs_b"),
            "structured_responses": result.get("structured_responses"),
            "messages": result.get("messages", []),
        }

    async def _stock_price_node(self, state: FinanceAgentState) -> dict:
        """
        Delegate stock price handling to StockAgent (LLM + tools).
        No ticker extraction here.
        """
        try:
            result = await asyncio.wait_for(
                self.stock_agent.run(state),
                timeout=settings.LLM_TIMEOUT,
            )

            return {
                "partial_answers": [
                    {"route": "stock_price", "text": result.get("answer", "")}
                ],
                "active_routes": ["stock_price"],
                "messages": result.get("messages", []),
                "structured_responses": result.get("structured_responses"),
            }

        except asyncio.TimeoutError:
            answer = f"StockAgent timed out after {settings.LLM_TIMEOUT}s"
            return {
                "partial_answers": [{"route": "stock_price", "text": answer}],
                "active_routes": ["stock_price"],
                "messages": [AIMessage(content=answer)],
            }    
        
    async def _save_memory_node(self,state: FinanceAgentState) -> dict:
        """
        Append the latest Q&A to long-term memory in SQLite.
        Caps at 3000 characters to prevent unbounded growth.
        """
        session_id = state['session_id']
        question = state.get("question","")
        answer = state.get("answer","")

        if not answer:
            return {}
        
        existing = state.get("long_term_summary","")
        new_entry = f"Q: {question}\nA:{answer[:400]}"
        updated = f"{existing}\n\n{new_entry}".strip()

        if len(updated) > 3000:
            updated = updated[-3000:]
        
        await self.long_term_memory.save_memory(session_id,updated)
        logger.info(f"Saved long term memory for session: {session_id}")
        return {}
    
    async def _merge_node(self, state: FinanceAgentState) -> dict:
        """
        Aggregate partial_answers from all completed agent branches into a
        single formatted answer.

        Single-route: copies the one answer through unchanged.
        Multi-route: sorts by _SECTION_ORDER (stock_price first, summary second)
        and joins sections with bold headers and --- dividers.
        """
        parts = sorted(
            state.get("partial_answers", []),
            key=lambda x: _SECTION_ORDER.get(x["route"], 99),
        )
        if not parts:
            return {"answer": ""}
        if len(parts) == 1:
            return {"answer": parts[0]["text"]}
        sections = [
            f"**{_SECTION_LABEL.get(p['route'], p['route'])}**\n\n{p['text']}"
            for p in parts
        ]
        merged = "\n\n---\n\n".join(sections)
        logger.info(f"Merged {len(parts)} agent responses: {[p['route'] for p in parts]}")
        return {"answer": merged}

    def _decide_route(self, state: FinanceAgentState) -> list[str]:
        """
        Read route from state and apply override rules. Returns a list so
        LangGraph can fan-out to multiple nodes when needed.

        Overrides (in priority order):
        1. If session_id_b is set (two PDFs uploaded), force comparision.
        2. If route is comparision but session_id_b is missing, fall back to summary.
        3. Unknown routes fall back to summary.
        4. If route is summary AND question contains an explicit ticker, fan out to
           summary + stock_price in parallel.
        """
        route = state.get("route", "summary")

        if state.get("session_id_b") and route in ("summary", "comparision"):
            logger.info("session_id_b detected — forcing comparision route")
            return ["comparision"]

        if route == "comparision" and not state.get("session_id_b"):
            logger.warning(
                "Route 'comparision' selected but session_id_b is missing — "
                "falling back to summary"
            )
            return ["summary"]

        valid_routes = {"summary", "chart", "comparision", "stock_price"}
        if route not in valid_routes:
            logger.warning(f"Unknown route '{route}', defaulting to 'summary'")
            return ["summary"]

        # Multi-intent fan-out: summary + stock_price
        if route == "summary" and self._has_explicit_ticker(
            state.get("question", "")
        ):
            logger.info("Multi-intent detected: fanning out to summary + stock_price")
            return ["summary", "stock_price"]

        return [route]
    
    async def build_graph(self) -> None:
        """
        1. Connect to MCP server and load tools
        2. Compile the LangGraph StateGraph with SQLite checkpointing

        Call once at application startup via get_orchestrator().
        """
        await self._connect_mcp()

        await self.long_term_memory.setup()

        self._checkpointer_cm,self.checkpointer = await get_checkpointer()
        graph = StateGraph(FinanceAgentState)

        graph.add_node("load_memory",self._load_memory_node)
        graph.add_node("route",self._route_node)
        graph.add_node("summary",self._summary_node)
        graph.add_node("chart",self._chart_node)
        graph.add_node("comparision",self._comparision_node)
        graph.add_node("stock_price",self._stock_price_node)
        graph.add_node("merge",self._merge_node)
        graph.add_node("save_memory",self._save_memory_node)

        graph.add_edge(START,"load_memory")
        graph.add_edge("load_memory","route")
        graph.add_conditional_edges("route",
                                    self._decide_route,
                                    {
                                        "summary":"summary",
                                        "chart":"chart",
                                        "comparision":"comparision",
                                        "stock_price":"stock_price"
                                    })

        # All agents converge on merge (LangGraph waits for all parallel branches)
        for agent_node in ["summary","chart","comparision","stock_price"]:
            graph.add_edge(agent_node,"merge")

        graph.add_edge("merge","save_memory")
        graph.add_edge("save_memory",END)
        self._app = graph.compile(checkpointer=self.checkpointer)
        logger.info("OrchestratorAgent graph compiled successfully")
    
    async def run(self,
                  question: str,
                  session_id: str,
                  session_id_b: str | None = None,
                  page_number: int | None = None,
                  thread_id: str | None = None):
        """Runs the orchestrator agent for question 
        Returns: dict with: answer,route,session_id"""

        if self._app is None:
            await self.build_graph()

        config = {"configurable":{"thread_id":thread_id or session_id}}

        initial_state = {"messages":[HumanMessage(content=question)],
                        "session_id":session_id,
                        "session_id_b":session_id_b,
                        "question":question,
                        "page_number":page_number}
        
        result = await self._app.ainvoke(initial_state,config=config)

        return {"answer":result.get("answer",""),
                "route":", ".join(result.get("active_routes",[""])),
                "session_id":session_id}
    
    async def stream(self,
                     question: str,
                     session_id: str,
                     session_id_b: str | None = None,
                     page_number: int | None = None,
                     thread_id: str | None = None):
        """
        Stream answer tokens using LangGraph's astream_events (v2).

        Filters out tokens from routing/memory nodes so only the
        actual agent response is streamed to the user.
        """

        if self._app is None:
            await self.build_graph()

        config = {"configurable": {"thread_id": thread_id or session_id}}
        initial_state = {
            "messages":    [HumanMessage(content=question)],
            "session_id":  session_id,
            "session_id_b": session_id_b,
            "question":    question,
            "page_number": page_number,
        }

        route_emitted = False
        stock_price_complete = False
        stock_price_text = ""
        summary_started = False

        async for event in self._app.astream_events(
            initial_state, config=config, version="v2"
        ):
            event_type = event.get("event")
            node = event.get("metadata", {}).get("langgraph_node", "")

            # Events are delivered in node-execution order by astream_events; returning here
            # stops processing before any downstream node events can arrive.
            # Surface node errors — skip routing/memory/merge nodes
            if event_type == "on_chain_error":
                if node and node not in ("route", "load_memory", "save_memory", "merge"):
                    error = event.get("data", {}).get("error")
                    error_msg = str(error) if error else "Unknown agent error"
                    logger.error(f"Node '{node}' raised: {error_msg}")
                    yield f"[ERROR] {error_msg}"
                    return

            # Emit route badge once from the route node's output
            if not route_emitted and event_type == "on_chain_end" and node == "route":
                output = event.get("data", {}).get("output", {})
                route = output.get("route", "summary") if isinstance(output, dict) else "summary"
                yield f"[ROUTE:{route}]"
                route_emitted = True
                continue

            # Stock price node completed — buffer for possible multi-intent prefix
            if event_type == "on_chain_end" and node == "stock_price":
                output = event.get("data", {}).get("output", {})
                partial = output.get("partial_answers", []) if isinstance(output, dict) else []
                stock_price_text = partial[0]["text"] if partial else ""
                stock_price_complete = True
                continue

            # LLM token stream from summary/chart/comparision
            if event_type == "on_chat_model_stream":
                if node in ("route", "load_memory", "save_memory", "merge"):
                    continue

                # Multi-intent: emit stock-price section before first summary token
                if node == "summary" and not summary_started and stock_price_complete:
                    label_sp = _SECTION_LABEL["stock_price"]
                    label_su = _SECTION_LABEL["summary"]
                    yield (
                        f"**{label_sp}**\n\n{stock_price_text}"
                        f"\n\n---\n\n**{label_su}**\n\n"
                    )
                    summary_started = True

                chunk = event["data"].get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
                continue

            # Single stock_price route: merge_node fires, no LLM tokens were yielded
            if event_type == "on_chain_end" and node == "merge":
                if stock_price_complete and not summary_started:
                    yield stock_price_text
                continue

_orchestrator: OrchestratorAgent | None = None

async def get_orchestrator() -> OrchestratorAgent:
    """
    Return the shared OrchestratorAgent instance.
    Builds the graph and connects to MCP on first call.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
        await _orchestrator.build_graph()
    return _orchestrator

# if __name__ == "__main__":
   

#     async def main():
#         try:
#             logger.info("Testing orchestrator agent")
#             # Create orchestrator
#             orchestrator = OrchestratorAgent()

#             # Build graph + connect MCP
#             await orchestrator.build_graph()

#             # Use an existing session that already has FAISS index
#             session_id = "test_session"   # change if needed

#             # Test questions (try different routes)
#             questions = [
#                 "What is the revenue?",                     # summary
#                 "Describe charts on page 5",                # chart
#                 "Compare revenue of both companies",        # comparison
#                 "What is DBS stock price?"                  # stock price
#             ]

#             for q in questions:
#                 print(f"\nQuestion: {q}")
#                 try:
#                     result = await orchestrator.run(
#                         question=q,
#                         session_id=session_id,
#                         session_id_b=None,
#                         page_number=5,
#                         thread_id=str(uuid.uuid4())  # fresh thread per question to avoid history bleed
#                     )

#                     print("Route:", result["route"])
#                     print("Answer:", result["answer"])
#                 except Exception as e:
#                     detail = getattr(e, "detail", None)
#                     print("FAILED:", str(e))
#                     if detail:
#                         print("Detail:", detail)
#                 print("-" * 50)

#         except Exception as e:
#             detail = getattr(e, "detail", None)
#             print("Test setup failed:", str(e))
#             if detail:
#                 print("Detail:", detail)

#     asyncio.run(main())