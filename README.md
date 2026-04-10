# Multimodal Finance RAG

An agentic RAG system for analyzing SGX annual reports. Upload a PDF, ask financial questions, compare companies side-by-side, analyze charts from specific pages, or fetch live stock prices — all from a single interface.

Built with LangGraph, LangChain, ChromaDB, and FastAPI. Supports OpenAI and Groq as LLM providers and runs a local Qwen3-VL embedding model on CUDA / MPS / CPU.

---

## Features

- **Financial Q&A** — RAG over uploaded annual reports using ChromaDB and local Qwen3-VL-Embedding-2B
- **Chart & Table Analysis** — Sends page images to a vision LLM to describe charts, graphs, and tables
- **Company Comparison** — Retrieves from two separate ChromaDB collections and generates a structured side-by-side analysis
- **Live Stock Prices** — `StockAgent` (LangChain ReAct) calls MCP tools (`get_stock_price`, `search_financial_news`) backed by yfinance and Tavily
- **Conversational Memory** — Long-term per-session Q&A summaries in SQLite + full LangGraph checkpoints for resume
- **Streaming** — `/chat/stream` (SSE) emits a `[ROUTE:<name>]` badge followed by the completed structured answer

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
    ├── summary     ──► SummaryAgent        ChromaDB RAG → structured JSON answer
    ├── chart       ──► ChartAgent          page image → vision LLM → chart analysis
    ├── comparision ──► ComparsionAgent     dual ChromaDB RAG → side-by-side table
    └── stock_price ──► StockAgent          ReAct loop → MCP tools → yfinance / Tavily
    │
    └── save_memory     append Q&A to SQLite
```

Each uploaded PDF gets its own `session_id`. The orchestrator routes to the right agent, injects long-term memory for context, and checkpoints full graph state in SQLite so requests can resume after failure.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Agent framework | LangChain v1 (`create_agent` for the ReAct stock agent) |
| LLM (text) | OpenAI `gpt-4o-mini` or Groq `llama-3.3-70b-versatile` |
| LLM (vision) | OpenAI `gpt-4o-mini` or Groq `llama-4-scout-17b` |
| Embeddings | [Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen) (local, CUDA / MPS / CPU) |
| Vector Store | ChromaDB (one collection per session) |
| PDF Processing | PyMuPDF + PyPDF |
| Tool Server | [FastMCP](https://github.com/jlowin/fastmcp) over stdio |
| Memory | SQLite via `aiosqlite` (long-term summaries + LangGraph checkpoints) |
| Backend API | FastAPI + Uvicorn (SSE streaming) |

---

## Project Structure

```
multimodal-finance-rag/
├── src/
│   ├── agents/
│   │   ├── orchestrator_agent.py   # LangGraph graph, routing, MCP connection, streaming
│   │   ├── summary_agent.py        # RAG pipeline for financial Q&A
│   │   ├── chart_agent.py          # Vision LLM pipeline for charts/tables
│   │   ├── comparision_agent.py    # Dual-document comparison agent
│   │   ├── stock_agent.py          # LangChain ReAct agent over MCP tools
│   │   └── state.py                # FinanceAgentState (extends MessagesState)
│   ├── core/
│   │   ├── pdf_processor.py        # PDF → text chunks + page PNG images + chart detection
│   │   ├── embeddings.py           # Qwen3-VL local embedding model singleton
│   │   └── vector_store.py         # ChromaDB collection build / load / retrieve
│   ├── memory/
│   │   ├── long_term.py            # SQLite-backed per-session Q&A summaries
│   │   └── checkpoint.py           # LangGraph SQLite checkpointer
│   ├── mcp_server/
│   │   └── server.py               # FastMCP tools: get_stock_price, search_financial_news
│   ├── models/schemas.py           # Pydantic schemas for structured LLM outputs
│   ├── prompts/                    # System prompts for each agent
│   ├── exceptions/                 # FinDocBaseException hierarchy
│   ├── logger/                     # Shared logger (writes to app.log)
│   └── settings/config.py          # Pydantic settings, get_llm(), get_vision_llm()
├── backend/
│   ├── app.py                      # FastAPI: lifespan warm-up, /upload, /chat/stream
│   └── models/api_schemas.py       # UploadResponse, ChatRequest
├── frontend/                       # Frontend client
├── tests/                          # Pytest suite
├── evals/                          # Evaluation harnesses (DeepEval)
├── docs/                           # Additional project docs
├── data/                           # Runtime: uploaded PDFs, page images, SQLite DBs
├── chroma_index/                   # Runtime: ChromaDB collections (one dir per session)
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
# Required (at least one LLM provider)
OPENAI_API_KEY=sk-...

# Optional — switch to Groq instead of OpenAI
GROQ_API_KEY=...
LLM_PROVIDER=openai        # or "groq"

# Embeddings — local Qwen3-VL model, no API key required
EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_DEVICE=auto      # auto → CUDA → MPS → CPU

# Optional — for search_financial_news MCP tool
TAVILY_API_KEY=...

# Optional — LangSmith tracing
LANGSMITH_API_KEY=...
```

> The Qwen3-VL embedding model (~5 GB) is downloaded from Hugging Face on first run and loaded into memory at FastAPI startup (`lifespan`). Expect a ~30 s cold start on CPU; much faster on CUDA / MPS.

### 4. Run the MCP server (dev mode)

```bash
uv run mcp dev src/mcp_server/server.py
```

Opens a browser UI to test `get_stock_price` and `search_financial_news` interactively.

### 5. Test the orchestrator

```bash
uv run python -m src.agents.orchestrator_agent
```

Runs test questions (summary, chart, comparison, stock price) against `test_session`. The first three require a PDF to be uploaded and indexed first; the stock price question works immediately via the MCP server subprocess.

### 6. Run the FastAPI backend

```bash
uv run uvicorn backend.app:app --reload
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `GET`  | `/health`       | Liveness probe |
| `POST` | `/upload`       | Upload a PDF — returns `session_id`, page count, chunk count, detected chart pages |
| `POST` | `/chat/stream`  | SSE stream of `[ROUTE:<name>]` badge + final answer |

---

## Stock Tickers

The `StockAgent` is a ReAct loop — the LLM decides when to call `get_stock_price` and which ticker to pass. SGX tickers use the `.SI` suffix (e.g. `D05.SI` for DBS, `O39.SI` for OCBC, `U11.SI` for UOB). US-listed tickers (e.g. `AAPL`, `TSLA`) are also supported.

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
- **Routing overrides:** if `session_id_b` is set on a request, the orchestrator forces the `comparision` route regardless of the LLM's decision. If the LLM picks `comparision` without a second session, it falls back to `summary`.
- **Streaming is not token-by-token.** Agents return structured JSON, so `/chat/stream` runs each agent to completion, then emits the route badge and final answer string. See the `stream()` docstring in `src/agents/orchestrator_agent.py`.
- **Memory** is per-session. Long-term memory is stored in `data/memory.db`; LangGraph checkpoints in `data/checkpoints.db`.
- **Test the MCP server** independently with `uv run mcp dev src/mcp_server/server.py` before running the full orchestrator.
