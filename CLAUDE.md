# multimodal-finance-rag — Claude Code Guide

## Project Overview

Agentic RAG system for analyzing SGX (Singapore Exchange) annual reports. Users upload PDF annual reports; the system answers financial questions, generates chart analysis from page images, compares two companies side-by-side, and fetches live stock prices.

**Stack:** LangGraph (orchestration), LangChain, ChromaDB (vector store), Qwen3-VL-Embedding-2B (local GPU embeddings), FastAPI (backend), MCP (stock price / news tools), SQLite (memory + checkpointing), OpenAI/Groq LLMs.

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
    ├── summary       → SummaryAgent    (ChromaDB RAG → structured JSON answer)
    ├── chart         → ChartAgent      (page image → vision LLM → chart analysis)
    ├── comparision   → ComparsionAgent (dual ChromaDB RAG → side-by-side table)
    └── stock_price   → StockAgent      (LangChain ReAct agent → MCP tools → yfinance / Tavily)
    │
    └── save_memory   → append Q&A to SQLite long-term memory
```

**Important:** The route name for comparison is spelled `"comparision"` throughout the codebase (graph nodes, valid_routes, schema, state). Do NOT change to correct spelling without updating all occurrences consistently.

---

## Key Files

| Path | Purpose |
|------|---------|
| `src/agents/orchestrator_agent.py` | Main LangGraph graph, routing logic, MCP connection, streaming |
| `src/agents/state.py` | `FinanceAgentState` (extends `MessagesState`) — shared state across all nodes |
| `src/agents/summary_agent.py` | RAG pipeline: retrieve → generate structured financial summary |
| `src/agents/chart_agent.py` | Vision LLM pipeline: load page image → analyze charts/tables |
| `src/agents/comparision_agent.py` | Dual-document RAG for company comparison |
| `src/agents/stock_agent.py` | LangChain v1 `create_agent` ReAct loop; wraps MCP tools as LangChain `Tool`s |
| `src/core/pdf_processor.py` | PDF → markdown text chunks (tables preserved inline via `pymupdf4llm`) + page PNG images at 3× scale + chart-page detection with captions |
| `src/core/embeddings.py` | `QwenVLEmbeddings` + `get_qwen_embeddings()` singleton (pre-loaded at startup) |
| `src/core/vector_store.py` | ChromaDB collection build/load/retrieve (one collection per session) |
| `src/memory/long_term.py` | SQLite-backed long-term memory (per session Q&A summaries) |
| `src/memory/checkpoint.py` | LangGraph SQLite checkpointer (full graph state) |
| `src/mcp_server/server.py` | FastMCP server: `get_stock_price` + `search_financial_news` |
| `src/models/schemas.py` | Pydantic schemas: `RouterDecision`, `FinancialSummary`, `ChartAnalysis`, `ComparisionSummary` |
| `src/prompts/` | System prompts for each agent |
| `src/exceptions/custom_exceptions.py` | `FinDocBaseException` hierarchy: `AgentError`, `OrchestratorError`, `VectorStoreError`, `PDFProcessingError` |
| `src/logger/custom_logger.py` | Shared `logger` (writes to `app.log`) |
| `src/settings/config.py` | Pydantic settings from `.env`; cached `get_llm()` / `get_vision_llm()` |
| `backend/app.py` | FastAPI: `lifespan` warm-up, `/health`, `/upload` (PDF→ChromaDB index), `/chat/stream` (SSE), exception handler |
| `backend/models/api_schemas.py` | `UploadResponse`, `ChatRequest` request/response models |
| `frontend/` | Frontend client (served separately from the FastAPI backend) |

---

## Session / Data Model

- Each uploaded PDF gets a `session_id` (UUID).
- Files stored at `data/<session_id>/`.
- Page images stored at `data/<session_id>/page_images/page_N.png`.
- ChromaDB index stored at `chroma_index/<session_id>/`.
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

# Run FastAPI backend
uv run uvicorn backend.app:app --reload
```

---

## Environment Variables (`.env`)

```
OPENAI_API_KEY=
GROQ_API_KEY=
LANGSMITH_API_KEY=
TAVILY_API_KEY=        # Optional — for search_financial_news MCP tool

LLM_PROVIDER=openai    # or "groq"

# Embeddings — local model, no API key required
EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_DEVICE=auto  # auto → CUDA → MPS → CPU
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

6. **`MAX_TOKENS=4000`** — default in `src/settings/config.py`. Comparison responses need ≥2000 for full JSON output; 4000 gives headroom. Override in `.env` if needed.

7. **Embedding model cold start:** FastAPI lifespan pre-loads Qwen3-VL via `asyncio.to_thread` (~30s on CPU, faster on GPU). The first upload will not block if the lifespan warm-up has completed.

8. **`StockAgent` is a ReAct agent, not a direct MCP call.** It uses `langchain.agents.create_agent` with MCP tools wrapped as LangChain `Tool`s (see `convert_mcp_tools` in `src/agents/stock_agent.py`). The LLM decides whether to call `get_stock_price` or `search_financial_news`. Do not bypass it with a direct tool call — the ReAct loop handles ticker extraction and error recovery.

9. **`orchestrator.stream()` is not token-by-token.** All four agents output structured JSON, so the orchestrator runs each agent to completion via `ainvoke`, then yields a `[ROUTE:<name>]` badge followed by the full answer string. Do not rewire to `astream_events` without a plan for suppressing raw JSON tokens in the UI.

10. **`_decide_route` overrides the LLM.** If `session_id_b` is set, the route is forced to `"comparision"` regardless of what the router LLM picks. If the LLM picks `"comparision"` but `session_id_b` is missing, it falls back to `"summary"`. Tests that assert routing behavior must account for this.

11. **Tables live in the RAG index, not the chart agent.** `PDFProcessor.extract_documents()` uses `pymupdf4llm.to_markdown(page_chunks=True)` which preserves tables as markdown pipe rows inside the chunked text. This means the summary agent can answer numeric table questions directly via ChromaDB retrieval — you should not route table lookups through the chart agent. The chart agent is for genuinely visual content (bar charts, line graphs, diagrams) that the vision LLM must see.

12. **`chart_pages.json` schema.** Written by `PDFProcessor.detect_chart_pages()`. Format: `[{"page": int (1-indexed), "tables": int, "images": int, "graphics": int, "caption": str}]`. The caption is extracted from the first markdown heading / bold line on the page and is what `ChartAgent._find_best_chart_page()` ranks against the user question. Old sessions uploaded before this change have a flat `[int, int, ...]` schema and will break the chart agent — re-upload them.

13. **Prompt strings in `src/prompts/` use single braces, not `{{` / `}}`.** These prompts are consumed as raw Python f-strings (e.g. `f"{system_prompt}\n\n..."` in [src/agents/chart_agent.py:309](src/agents/chart_agent.py#L309)), NOT via `ChatPromptTemplate.format()`. So literal `{` and `}` in a schema example must stay single. If you escape them as `{{` `}}`, the escapes pass through verbatim to the LLM, the model mirrors them in its output, and `JsonOutputParser` fails with `Invalid json output: {{ ... }}`. This exact bug hit `ChartAgent`: `CHART_BASE_PROMPT` previously contained a `{{`-wrapped schema example plus the line `Response MUST start with {{ and end with }}`, so the vision LLM faithfully wrapped its JSON in `{{` `}}`. If you ever switch a prompt to an actual `ChatPromptTemplate`, re-escape the braces at the same time.
