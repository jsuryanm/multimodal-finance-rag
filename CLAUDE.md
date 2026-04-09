# multimodal-finance-rag — Claude Code Guide

## Project Overview

Agentic RAG system for analyzing SGX (Singapore Exchange) annual reports. Users upload PDF annual reports; the system answers financial questions, generates chart analysis from page images, compares two companies side-by-side, and fetches live stock prices.

**Stack:** LangGraph (orchestration), LangChain, FAISS (vector store), Jina embeddings, FastAPI (backend), MCP (stock price / news tools), SQLite (memory + checkpointing), OpenAI/Groq LLMs.

---

## Architecture

```
User Question
    │
    ▼
OrchestratorAgent (LangGraph StateGraph)
    ├── load_memory   → load prior Q&A context from SQLite
    ├── route         → LLM classifies intent → one of 4 routes
    │
    ├── summary       → SummaryAgent   (FAISS RAG → structured JSON answer)
    ├── chart         → ChartAgent     (page image → vision LLM → chart analysis)
    ├── comparision   → ComparsionAgent (dual FAISS RAG → side-by-side table)
    └── stock_price   → MCP tool call  (yfinance via stdio MCP server)
    │
    └── save_memory   → append Q&A to SQLite long-term memory
```

**Important:** The route name for comparison is spelled `"comparision"` throughout the codebase (graph nodes, valid_routes, schema, state). Do NOT change to correct spelling without updating all occurrences consistently.

---

## Key Files

| Path | Purpose |
|------|---------|
| `src/agents/orchestrator_agent.py` | Main LangGraph graph, routing logic, MCP connection |
| `src/agents/state.py` | `FinanceAgentState` TypedDict (shared state across all nodes) |
| `src/agents/summary_agent.py` | RAG pipeline: retrieve → generate structured financial summary |
| `src/agents/chart_agent.py` | Vision LLM pipeline: load page image → analyze charts/tables |
| `src/agents/comparision_agent.py` | Dual-document RAG for company comparison |
| `src/core/pdf_processor.py` | PDF → text chunks + page PNG images |
| `src/core/vector_store.py` | FAISS index build/load/retrieve (one index per session) |
| `src/memory/long_term.py` | SQLite-backed long-term memory (per session Q&A summaries) |
| `src/memory/checkpoint.py` | LangGraph SQLite checkpointer (full graph state) |
| `src/mcp_server/server.py` | FastMCP server: `get_stock_price` + `search_financial_news` |
| `src/models/schemas.py` | Pydantic schemas: `RouterDecision`, `FinancialSummary`, `ChartAnalysis`, `ComparisionSummary` |
| `src/settings/config.py` | Pydantic settings from `.env`; `get_llm()`, `get_vision_llm()` |
| `backend/app.py` | FastAPI app (in progress — currently empty) |

---

## Session / Data Model

- Each uploaded PDF gets a `session_id` (UUID).
- Files stored at `data/<session_id>/`.
- Page images stored at `data/<session_id>/page_images/page_N.png`.
- FAISS index stored at `faiss_index/<session_id>/`.
- Long-term memory stored in `data/memory.db` (SQLite).
- LangGraph checkpoints stored in `data/checkpoints.db` (SQLite).

---

## Running

```bash
# Install deps
uv sync

# Run MCP server in dev mode (for testing tools)
uv run mcp dev src/mcp_server/server.py

# Test orchestrator standalone
uv run python -m src.agents.orchestrator_agent

# Test PDF processor standalone
uv run python -m src.core.pdf_processor
```

---

## Environment Variables (`.env`)

```
OPENAI_API_KEY=
GROQ_API_KEY=
JINA_API_KEY=          # Required for embeddings
LANGSMITH_API_KEY=
TAVILY_API_KEY=        # Optional — for search_financial_news MCP tool

LLM_PROVIDER=openai    # or "groq"
```

---

## LLM Provider

Set `LLM_PROVIDER` in `.env`:
- `openai` → uses `gpt-4o-mini` for text, `gpt-4o-mini` for vision
- `groq` → uses `llama-3.3-70b-versatile` for text, `meta-llama/llama-4-scout-17b-16e-instruct` for vision

`get_llm()` and `get_vision_llm()` are cached with `@lru_cache`. Changing provider requires restarting the process.

---

## Known Conventions & Pitfalls

1. **"comparision" is the canonical spelling** in route names, graph node names, schema fields, and state comments. The system message in `_route_node` must also use "comparision" — if it uses "comparison", the LLM may return the correctly-spelled string which won't match any valid route and will silently fall back to "summary".

2. **Memory saving is handled by the orchestrator's `_save_memory_node`** for all routes. `SummaryAgent` has its own `save_memory_node()` method but it is NOT called from `SummaryAgent.run()` (the orchestrator handles it). Do not re-add the call — it would cause a double-save where the orchestrator overwrites the agent's LLM-compressed summary.

3. **`SummaryAgent.run()` still calls `load_memory_node()`** internally (slightly redundant since the orchestrator already loaded it), but this is harmless and provides the agent with fresh memory for standalone use.

4. **State mutation:** Agent `run()` methods call `state.update(...)` to thread intermediate results through their internal pipeline. This is fine because LangGraph passes state as a plain dict.

5. **MCP client:** `MultiServerMCPClient` is instantiated without `async with`. The tools returned by `get_tools()` maintain their connection. Do not wrap in `async with` unless the library version requires it.

6. **`MAX_TOKENS=500`** is intentionally low. Increase in `.env` if answers are truncated.
