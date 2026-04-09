# Agents

This project uses a LangGraph `StateGraph` with four specialist agents coordinated by a central orchestrator.

---

## OrchestratorAgent

**File:** `src/agents/orchestrator_agent.py`

The top-level controller. Builds and runs the LangGraph graph, routes user questions to the correct specialist agent, and manages short- and long-term memory.

### Graph Flow

```
START
  └─ load_memory    Load prior Q&A summary from SQLite into state
  └─ route          LLM classifies question intent → one of 4 routes
       ├─ summary       → SummaryAgent
       ├─ chart         → ChartAgent
       ├─ comparision   → ComparsionAgent
       └─ stock_price   → MCP tool call (yfinance)
  └─ save_memory    Append Q&A to SQLite long-term memory
END
```

### Key Methods

| Method | Description |
|--------|-------------|
| `build_graph()` | Connects MCP server, sets up SQLite checkpointer, compiles the StateGraph. Call once at startup. |
| `run(question, session_id, ...)` | Invokes the graph and returns `{answer, route, session_id}`. |
| `stream(question, session_id, ...)` | Async generator that yields answer tokens via `astream_events`. Filters out routing/memory node tokens. |
| `_route_node(state)` | Calls LLM with structured output (`RouterDecision`) to classify intent. Priority rule: `stock_price` and `comparision` take precedence over `summary`. |
| `_decide_route(state)` | Conditional edge function. Forces `comparision` if `session_id_b` is present. Defaults unknown routes to `summary`. |
| `_extract_ticker(question)` | Extracts stock ticker from text. Priority: explicit `.SI` suffix → known company name → all-caps word → fallback `D05.SI`. |
| `_format_stock_response(result)` | Formats the MCP `get_stock_price` dict into a readable multi-line string. |

### Routing

The route key is always spelled `"comparision"` (consistent throughout the codebase). Valid values: `summary`, `chart`, `comparision`, `stock_price`.

### Singleton

```python
orchestrator = await get_orchestrator()  # shared instance, builds graph on first call
```

---

## SummaryAgent

**File:** `src/agents/summary_agent.py`

Answers general financial questions using RAG (Retrieval-Augmented Generation) over a FAISS vector index built from the uploaded PDF.

### Flow

```
load_memory_node  →  retrieve_node  →  generate_node
```

### Key Methods

| Method | Description |
|--------|-------------|
| `retrieve_node(state)` | Loads FAISS index for `session_id`, runs MMR retrieval (`k=5`), returns `{documents}`. |
| `generate_node(state)` | Builds a RAG chain (`prompt → LLM → JsonOutputParser`), generates a `FinancialSummary` structured response. Falls back to raw string if parsing fails. |
| `load_memory_node(state)` | Loads long-term memory from SQLite for context injection. |
| `save_memory_node(state)` | Uses LLM to compress conversation into a summary and saves to SQLite. *Not called from `run()` — orchestrator handles saving.* |
| `run(state)` | Runs the full pipeline sequentially. Returns `{answer, documents, structured_responses, messages, route}`. |
| `stream(state)` | Async generator that streams raw LLM tokens (no JSON parsing). |

### Output Schema

`FinancialSummary` — see `src/models/schemas.py`:
- `revenue`, `net_profit`, `operating_profit`, `total_assets`, `total_liabilities`
- `eps`, `roe`, `dividend`, `yoy_growth`, `key_risks`
- `summary` — 2–3 sentence analyst-style narrative (used as the main `answer`)

### Requirements

- FAISS index must exist at `faiss_index/<session_id>/`
- Raises `VectorStoreError` → wrapped as `AgentError` if index is missing

---

## ChartAgent

**File:** `src/agents/chart_agent.py`

Analyzes charts, tables, and figures from a specific PDF page using a vision-capable LLM.

### Flow

```
load_image_node  →  analyze_image_node
```

### Key Methods

| Method | Description |
|--------|-------------|
| `load_image_node(state)` | Reads `data/<session_id>/page_images/page_N.png`, converts to base64. |
| `analyze_image_node(state)` | Sends a multimodal message (text prompt + base64 image) to the vision LLM, parses the response as `ChartAnalysis`. |
| `run(state)` | Runs both nodes sequentially. Returns `{answer, structured_responses, messages, image_b64, route}`. |
| `stream(state)` | Async generator that streams raw vision LLM tokens. |

### Output Schema

`ChartAnalysis` — see `src/models/schemas.py`:
- `visual_type` — bar chart, line chart, pie chart, table, etc.
- `title`, `time_period`
- `key_values`, `trend`, `key_insight`
- `explanation` — used as the main `answer`

### Requirements

- Page images must exist at `data/<session_id>/page_images/page_N.png` (1-indexed)
- Requires a vision-capable LLM (`get_vision_llm()`)
- `page_number` must be set in state; defaults to `1` if not provided

---

## ComparsionAgent

**File:** `src/agents/comparision_agent.py`

Generates a structured side-by-side comparison of two companies by retrieving from two separate FAISS indexes in parallel.

### Flow

```
retrieve_both_nodes  →  compare_nodes
```

### Key Methods

| Method | Description |
|--------|-------------|
| `retrieve_both_nodes(state)` | Runs MMR retrieval against both `session_id` (Company A) and `session_id_b` (Company B) FAISS indexes in parallel via `asyncio.gather`. Returns `{docs_a, docs_b}`. |
| `compare_nodes(state)` | Builds a comparison prompt with both contexts, calls LLM with `JsonOutputParser`, parses result as `ComparisionSummary`. Uses `final_verdict` as the main answer. |
| `run(state)` | Runs both nodes sequentially. Returns `{answer, structured_responses, messages, route}`. |
| `stream(state)` | Async generator that streams raw LLM tokens for the comparison. |

### Output Schema

`ComparisionSummary` — see `src/models/schemas.py`:
- `company_a_name`, `company_b_name`
- `rows` — list of `ComparisionRow` (metric, company_a, company_b, insight)
- `revenue_comparision`, `profit_comparison`, `debt_comparison`, `growth_comparison`
- `final_verdict` — used as the main `answer`

### Requirements

- `session_id_b` must be set in state (raises `AgentError` if missing)
- Both FAISS indexes must exist (`session_id` and `session_id_b`)

---

## Shared State

All agents read from and write to `FinanceAgentState` (defined in `src/agents/state.py`), which extends LangGraph's `MessagesState`.

| Field | Type | Used by |
|-------|------|---------|
| `session_id` | `str` | All agents |
| `session_id_b` | `Optional[str]` | ComparsionAgent, Orchestrator routing |
| `question` | `Optional[str]` | All agents |
| `route` | `Optional[str]` | Orchestrator |
| `documents` | `Optional[list[Document]]` | SummaryAgent |
| `docs_a`, `docs_b` | `Optional[list[Document]]` | ComparsionAgent |
| `page_number` | `Optional[int]` | ChartAgent |
| `image_b64` | `Optional[str]` | ChartAgent |
| `answer` | `Optional[str]` | All agents (output) |
| `structured_responses` | `Optional[dict]` | All agents (output) |
| `long_term_summary` | `Optional[str]` | Orchestrator memory nodes |
| `summaries` | `Annotated[list[str], add]` | Accumulates across turns |
