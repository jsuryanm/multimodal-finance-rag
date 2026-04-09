# Multimodal Finance RAG

An agentic RAG system for analyzing SGX annual reports. Upload a PDF, ask financial questions, compare companies side-by-side, analyze charts from specific pages, or fetch live stock prices — all from a single interface.

Built with LangGraph, LangChain, FAISS, and FastAPI. Supports OpenAI and Groq as LLM providers.

---

## Features

- **Financial Q&A** — RAG over uploaded annual reports using FAISS vector search and Jina embeddings
- **Chart & Table Analysis** — Sends page images to a vision LLM to describe charts, graphs, and tables
- **Company Comparison** — Retrieves from two separate indexes and generates a structured side-by-side analysis
- **Live Stock Prices** — Fetches real-time prices and metrics from Yahoo Finance via an MCP tool server
- **Conversational Memory** — Persists Q&A summaries in SQLite so context carries across sessions
- **Streaming** — Streams answer tokens to the frontend via LangGraph's `astream_events`

---

## Architecture

```
User Question
    │
    ▼
OrchestratorAgent  (LangGraph StateGraph)
    ├── load_memory     load prior Q&A context from SQLite
    ├── route           LLM classifies intent → one of 4 routes
    │
    ├── summary     ──► SummaryAgent        FAISS RAG → structured JSON answer
    ├── chart       ──► ChartAgent          page image → vision LLM → chart analysis
    ├── comparision ──► ComparsionAgent     dual FAISS RAG → side-by-side table
    └── stock_price ──► MCP Tool            yfinance → live price + metrics
    │
    └── save_memory     append Q&A to SQLite
```

Each uploaded PDF gets its own `session_id`. The orchestrator routes to the right agent, injects long-term memory for context, and checkpoints full graph state in SQLite so requests can resume after failure.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| LLM (text) | OpenAI `gpt-4o-mini` or Groq `llama-3.3-70b-versatile` |
| LLM (vision) | OpenAI `gpt-4o-mini` or Groq `llama-4-scout-17b` |
| Embeddings | [Jina Embeddings v3](https://jina.ai/embeddings) |
| Vector Store | FAISS (one index per session) |
| PDF Processing | PyMuPDF + PyPDF |
| Tool Server | [FastMCP](https://github.com/jlowin/fastmcp) over stdio |
| Memory | SQLite via `aiosqlite` |
| Backend API | FastAPI + Uvicorn |

---

## Project Structure

```
multimodal-finance-rag/
├── src/
│   ├── agents/
│   │   ├── orchestrator_agent.py   # LangGraph graph, routing, MCP connection
│   │   ├── summary_agent.py        # RAG pipeline for financial Q&A
│   │   ├── chart_agent.py          # Vision LLM pipeline for charts/tables
│   │   ├── comparision_agent.py    # Dual-document comparison agent
│   │   └── state.py                # FinanceAgentState (shared TypedDict)
│   ├── core/
│   │   ├── pdf_processor.py        # PDF → text chunks + page PNG images
│   │   └── vector_store.py         # FAISS index build / load / retrieve
│   ├── memory/
│   │   ├── long_term.py            # SQLite-backed per-session Q&A summaries
│   │   └── checkpoint.py           # LangGraph SQLite checkpointer
│   ├── mcp_server/
│   │   └── server.py               # FastMCP tools: get_stock_price, search_financial_news
│   ├── models/schemas.py           # Pydantic schemas for structured LLM outputs
│   ├── prompts/                    # System prompts for each agent
│   └── settings/config.py          # Pydantic settings, get_llm(), get_vision_llm()
├── backend/
│   └── app.py                      # FastAPI application (in progress)
├── data/                           # Runtime: uploaded PDFs, page images, SQLite DBs
├── faiss_index/                    # Runtime: FAISS indexes (one dir per session)
├── agents.md                       # Agent reference documentation
├── skills.md                       # MCP tool reference documentation
└── CLAUDE.md                       # Claude Code guide for this repo
```

---

## Quickstart

### 1. Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/) package manager

### 2. Clone and install

```bash
git clone https://github.com/jsuryanm/multimodal-finance-rag.git
cd multimodal-finance-rag
uv sync
```

### 3. Configure environment

```bash
cp .env.example .env   # or create .env manually
```

```env
# Required
OPENAI_API_KEY=sk-...
JINA_API_KEY=...

# Optional — switch to Groq instead of OpenAI
GROQ_API_KEY=...
LLM_PROVIDER=groq

# Optional — for web news search tool
TAVILY_API_KEY=...

# Optional — LangSmith tracing
LANGSMITH_API_KEY=...
```

### 4. Run the MCP server (dev mode)

```bash
uv run mcp dev src/mcp_server/server.py
```

Opens a browser UI to test `get_stock_price` and `search_financial_news` interactively.

### 5. Test the orchestrator

```bash
uv run python -m src.agents.orchestrator_agent
```

Runs four test questions (summary, chart, comparison, stock price) against `test_session`. The first three require a PDF to be uploaded and indexed; the stock price question works immediately.

Expected output for the stock price question:
```
Question: What is DBS stock price?
Route: stock_price
Answer: DBS Group Holdings Ltd
Price: SGD 57.20 (-0.17%)
Market Cap: SGD 162.3B
```

---

## Supported Stock Tickers

The orchestrator auto-extracts tickers from natural language. Built-in mappings for SGX-listed stocks:

| Company | Ticker |
|---------|--------|
| DBS Bank | `D05.SI` |
| OCBC | `O39.SI` |
| UOB | `U11.SI` |
| Singtel | `Z74.SI` |
| CapitaLand | `9CI.SI` |
| Keppel | `BN4.SI` |
| Grab | `GRAB` |
| Sea Limited | `SE` |

US-listed tickers (e.g. `AAPL`, `TSLA`) are also supported.

---

## Example Questions

| Question | Route |
|----------|-------|
| "What was DBS's net profit in 2024?" | `summary` |
| "Describe the chart on page 12" | `chart` |
| "Compare DBS and OCBC's revenue growth" | `comparision` |
| "What is the current OCBC stock price?" | `stock_price` |
| "What are the key risks mentioned in the report?" | `summary` |
| "Explain the table on page 45" | `chart` |

---

## Documentation

| File | Contents |
|------|---------|
| [`agents.md`](agents.md) | Detailed reference for all four agents — methods, inputs, outputs, schemas |
| [`skills.md`](skills.md) | MCP tool reference — `get_stock_price`, `search_financial_news`, adding new tools |
| [`CLAUDE.md`](CLAUDE.md) | Claude Code guide — architecture, conventions, known pitfalls |

---

## Development Notes

- **LLM provider** is set via `LLM_PROVIDER=openai` or `LLM_PROVIDER=groq` in `.env`. Changing it requires a process restart (`get_llm()` uses `@lru_cache`).
- **Route names** use the spelling `comparision` (not `comparison`) throughout the codebase. This is consistent and intentional — changing it in one place breaks routing.
- **Memory** is per-session. Long-term memory is stored in `data/memory.db`; graph checkpoints in `data/checkpoints.db`.
- **Test the MCP server** independently with `uv run mcp dev src/mcp_server/server.py` before running the full orchestrator.
