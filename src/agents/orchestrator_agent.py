from __future__ import annotations
import re
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

from src.memory.checkpoint import get_checkpointer
from src.memory.long_term import get_long_term_memory
from src.models.schemas import RouterDecision
from src.settings.config import get_llm, settings
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
                "command":"python",
                "args":["-m","src.mcp_server.server"],
                "transport":"stdio"
            }
        })


        tools_list = await self._mcp_client.get_tools()
        self._mcp_tools = {tool.name:tool for tool in tools_list}

        logger.info(f"MCP server connected, Tools loaded: {list(self._mcp_tools.keys())}")

    def _extract_ticker(self, question: str) -> str:
        """
        Extract a stock ticker from the user's question.
        Priority:
        1. Explicit SGX ticker with .SI suffix (e.g., D05.SI)
        2. Known company name mapping (e.g., "dbs" → D05.SI)
        3. All-caps 2–5 letter word (e.g., AAPL)
        4. Default fallback to DBS (D05.SI)
        """
        # Bug 3 fix: use group(0) — no capture group in regex
        si_match = re.search(r"\b[A-Z0-9]{2,4}\.SI\b", question, re.IGNORECASE)
        if si_match:
            return si_match.group(0).upper()

        question_lower = question.lower()
        for company_name, ticker in self.KNOWN_TICKERS.items():
            if company_name in question_lower:
                return ticker

        NOT_TICKERS = {
            "THE", "AND", "FOR", "ARE", "DID", "CAN", "NOT", "GET",
            "ITS", "WAS", "HAS", "HAD", "WHAT", "WHEN", "HOW",
            "WHO", "WHY", "FROM", "WITH", "THAT", "THIS", "WILL",
            "BEEN", "THEIR", "THEY", "THAN", "MORE", "YEAR", "WHAT",
            "GIVE", "TELL", "SHOW", "DOES", "INTO", "OVER", "ALSO",
            "BOTH", "EACH", "MUCH", "SOME", "SUCH", "THEN", "THERE",
        }
        upper_matches = re.findall(r"\b[A-Z]{2,5}\b", question)
        for word in upper_matches:
            if word not in NOT_TICKERS:
                return word

        logger.warning(f"Could not extract ticker from '{question}'. Defaulting to D05.SI")
        return "D05.SI"

    def _has_explicit_ticker(self, question: str) -> bool:
        """Return True if the question contains an explicit ticker or known company name.

        Uses the same three-pass logic as _extract_ticker but returns a bool so that
        DBS (which maps to D05.SI via KNOWN_TICKERS) and the no-match fallback
        (also D05.SI) can be distinguished.
        """
        if re.search(r"\b[A-Z0-9]{2,4}\.SI\b", question, re.IGNORECASE):
            return True
        question_lower = question.lower()
        if any(name in question_lower for name in self.KNOWN_TICKERS):
            return True
        NOT_TICKERS = {
            "THE", "AND", "FOR", "ARE", "DID", "CAN", "NOT", "GET",
            "ITS", "WAS", "HAS", "HAD", "WHAT", "WHEN", "HOW",
            "WHO", "WHY", "FROM", "WITH", "THAT", "THIS", "WILL",
            "BEEN", "THEIR", "THEY", "THAN", "MORE", "YEAR", "WHAT",
            "GIVE", "TELL", "SHOW", "DOES", "INTO", "OVER", "ALSO",
            "BOTH", "EACH", "MUCH", "SOME", "SUCH", "THEN", "THERE",
        }
        return any(
            word not in NOT_TICKERS
            for word in re.findall(r"\b[A-Z]{2,5}\b", question)
        )

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
                "- stock_price: user asks about current/live stock price, share price, or mentions "
                "a ticker symbol (e.g. D05.SI, OCBC, DBS, GRAB)\n"
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
    
    async def _summary_node(self,state: FinanceAgentState) -> dict:
        """Delegate to SummaryAgent"""
        try:
            return await asyncio.wait_for(
                self.summary_agent.run(state),
                timeout=settings.LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"SummaryAgent timed out after {settings.LLM_TIMEOUT}s")

    async def _chart_node(self,state: FinanceAgentState) -> dict:
        """Delegate to ChartAgent"""
        try:
            return await asyncio.wait_for(
                self.chart_agent.run(state),
                timeout=settings.LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"ChartAgent timed out after {settings.LLM_TIMEOUT}s")

    async def _comparision_node(self,state: FinanceAgentState) -> dict:
        """Delegate to ComparsionAgent"""
        try:
            return await asyncio.wait_for(
                self.comparision_agent.run(state),
                timeout=settings.LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise OrchestratorError(f"ComparsionAgent timed out after {settings.LLM_TIMEOUT}s")
    
    async def _stock_price_node(self,state: FinanceAgentState) -> dict:
        """
        Fetch live stock price by calling the MCP server's get_stock_price tool.

        Flow:
        1. Extract ticker from user's question
        2. Call MCP tool via .ainvoke() — this goes to the MCP server subprocess
        3. Format the result into a readable string
        """
        question = state.get("question") or state["messages"][-1].content

        ticker = self._extract_ticker(question)
        logger.info(f"Stock price lookup for ticker: {ticker}")

        stock_tool = self._mcp_tools.get("get_stock_price")

        if not stock_tool:
            logger.error("get_stock_price MCP tool not found")
            answer = "Stock price tool is unavailable. Check MCP server"
            return {"answer":answer,
                    "route":"stock_price",
                    "messages":[AIMessage(content=answer)]}
        
        try:
            raw = await asyncio.wait_for(
                stock_tool.ainvoke({"ticker": ticker}),
                timeout=settings.LLM_TIMEOUT,
            )
        except asyncio.TimeoutError:
            answer = f"Stock price lookup timed out after {settings.LLM_TIMEOUT}s"
            return {"answer": answer, "route": "stock_price", "messages": [AIMessage(content=answer)]}

        # langchain_mcp_adapters returns a list of TextContent objects;
        # extract the JSON text and parse it into a dict
        if isinstance(raw, list):
            try:
                text = raw[0].text if hasattr(raw[0], "text") else raw[0].get("text", "{}")
                result = json.loads(text)
            except Exception as parse_err:
                result = {"error": f"Could not parse MCP response: {parse_err}"}
        elif isinstance(raw, str):
            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                result = {"error": raw}
        else:
            result = raw  # already a dict

        answer = self._format_stock_response(result)
        logger.info(f"Stock price fetched for {ticker}")

        return {"answer":answer,
                "route":"stock_price",
                "messages":[AIMessage(content=answer)]}
    
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
        
        for agent_node in ["summary","chart","comparision","stock_price"]:
            graph.add_edge(agent_node,"save_memory")
        
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
                "route":result.get("route",""),
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
        
        config = {"configurable":{"thread_id":thread_id or session_id}}
        initial_state = {"messages":[HumanMessage(content=question)],
                        "session_id":session_id,
                        "session_id_b":session_id_b,
                        "question":question,
                        "page_number":page_number}
        
        route_emitted = False
        async for event in self._app.astream_events(
            initial_state, config=config, version="v2"):

            event_type = event.get("event")
            node = event.get("metadata", {}).get("langgraph_node", "")

            # Surface node errors as [ERROR] tokens so the stream never silently empties
            if event_type == "on_chain_error":
                if node and node not in ("route", "load_memory", "save_memory"):
                    error = event.get("data", {}).get("error")
                    error_msg = str(error) if error else "Unknown agent error"
                    logger.error(f"Node '{node}' raised: {error_msg}")
                    yield f"[ERROR] {error_msg}"
                    return

            # Capture route from the route node's chain-end event and emit first
            if not route_emitted and event_type == "on_chain_end":
                if node == "route":
                    output = event.get("data", {}).get("output", {})
                    route = output.get("route", "summary") if isinstance(output, dict) else "summary"
                    yield f"[ROUTE:{route}]"
                    route_emitted = True
                    continue

            if event_type != "on_chat_model_stream":
                continue

            if node in ("route", "load_memory", "save_memory"):
                continue

            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content

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