# Frontend & Backend Design
**Date:** 2026-04-09  
**Status:** Approved

---

## Overview

Add a Streamlit frontend and FastAPI backend to the existing multimodal-finance-rag agent pipeline. Streamlit communicates with FastAPI exclusively via HTTP — no shared Python imports between the two layers.

---

## Architecture

```
Streamlit  (frontend/app.py)   port 8501
    │  HTTP  (requests library)
    ▼
FastAPI    (backend/app.py)    port 8000
    │  Python import
    ▼
OrchestratorAgent  (src/agents/orchestrator_agent.py)
    ├── SummaryAgent      FAISS RAG → structured financial answer
    ├── ChartAgent        Vision LLM → chart/table analysis
    ├── ComparsionAgent   Dual FAISS → side-by-side comparison
    └── MCP tools         yfinance → live stock price
```

FastAPI owns the orchestrator lifecycle. Streamlit is a pure HTTP client.

---

## FastAPI Backend  (`backend/app.py`)

### Startup

Use FastAPI `lifespan` context manager. On startup, call `await get_orchestrator()` to warm up the MCP subprocess and compile the LangGraph graph before the first request arrives.

### Endpoints

#### `GET /health`
Returns `{"status": "ok"}`. Used by Streamlit to verify the backend is reachable on load.

#### `POST /upload`
Upload a PDF and build its FAISS index.

**Request:** `multipart/form-data`
- `file` — PDF bytes (`UploadFile`)
- `session_id` — optional string query param. If omitted, a new UUID is generated.

**Processing:**
1. Instantiate `PDFProcessor(session_id)`
2. `save_pdf(filename, content)` — write bytes to `data/<session_id>/`
3. `extract_documents(pdf_path)` — load + chunk text
4. `VectorStore(session_id).build_index(chunks)` — build FAISS index
5. `extract_page_images(pdf_path)` — render page PNGs to `data/<session_id>/page_images/`

**Response:** `200 OK`
```json
{
  "session_id": "3f8a…c21b",
  "filename": "DBS_2024.pdf",
  "pages": 111,
  "chunks": 992
}
```

**Errors:**
- `422` — not a PDF, or PDF has no extractable text
- `500` — FAISS build failure or JINA API key missing

#### `POST /chat/stream`
Stream an answer for a user question. The orchestrator classifies intent and routes internally.

**Request:** `application/json`
```json
{
  "session_id": "uuid-a",
  "session_id_b": "uuid-b or null",
  "question": "What is the revenue?",
  "page_number": 1
}
```

**Processing:**
1. Validate `session_id` is present
2. Call `orchestrator.stream(question, session_id, session_id_b, page_number)`
3. Wrap the async generator in `StreamingResponse` with `media_type="text/event-stream"`
4. Each yielded chunk sent as: `data: <token>\n\n`
5. On completion, send: `data: [DONE]\n\n`

**Response:** `text/event-stream`

The stream sends events in this order:
1. Route event (always first): `data: [ROUTE:summary]\n\n`
2. Content chunks: `data: <token>\n\n` (one per yield from orchestrator)
3. Done sentinel: `data: [DONE]\n\n`

```
data: [ROUTE:summary]

data: DBS Group reported

data:  a net profit of

data:  SGD 10.3 billion

data: [DONE]
```

Streamlit reads the first event to extract the route, then collects remaining content chunks into the answer string.

**Errors:** If the orchestrator raises a `FinDocBaseException`, stream a single error event before closing:
```
data: [ERROR] No FAISS index found for session: uuid-a
```

### Error Handling

A global FastAPI exception handler catches all `FinDocBaseException` subclasses and maps them to HTTP responses:

| Exception | Status | Hint |
|-----------|--------|------|
| `VectorStoreError` | 422 | Upload a PDF first |
| `PDFProcessingError` | 422 | Check the PDF is not image-only |
| `AgentError` | 500 | Internal agent failure |
| `OrchestratorError` | 500 | Routing or LLM failure |

Response shape: `{"detail": "...", "hint": "..."}`

### File layout

```
backend/
├── app.py          # All FastAPI code (~120 lines)
└── models/
    └── schemas.py  # Already exists — UploadResponse, ChatRequest, ChatResponse
```

---

## Streamlit Frontend  (`frontend/app.py`)

### Layout

**Sidebar (left, fixed):**
- App title: `📊 Finance RAG`
- **Company A** — `st.file_uploader`, calls `POST /upload` on change, stores `session_id` in `st.session_state`
- Session ID display (truncated, e.g. `3f8a…c21b`)
- **Company B** — identical uploader, labeled "optional — enables comparison", stores `session_id_b`
- **Chart page** — `st.number_input(min=1)`, stored in `st.session_state.page_number`
- **Clear Chat** button — resets `st.session_state.messages`

**Main area:**
- Chat history rendered from `st.session_state.messages`
- Each message: `{"role": "user"|"assistant", "content": str, "route": str|None}`
- User messages: right-aligned blue bubble
- Assistant messages: left-aligned dark bubble with small route badge above it
- Route badge colours: `summary`=blue, `chart`=purple, `comparision`=orange, `stock_price`=green
- Input bar at bottom: `st.chat_input("Ask about the annual report…")`

### Streaming Flow

On question submit:
1. Append `{"role": "user", "content": question}` to `st.session_state.messages`
2. Display a spinner: `"Routing to agent…"`
3. Call `POST /chat/stream` with `stream=True` via `requests.post(..., stream=True)`
4. Define a generator that iterates `response.iter_lines()`, strips `data: ` prefix, stops on `[DONE]`, yields on `[ERROR]` with error text
5. Use `st.write_stream(generator)` inside an `st.chat_message("assistant")` block — Streamlit handles the incremental rendering
6. Capture the final concatenated content and route from the last SSE event
7. Append `{"role": "assistant", "content": full_answer, "route": route}` to session state

### State

All mutable state lives in `st.session_state`:

| Key | Type | Description |
|-----|------|-------------|
| `session_id` | `str\|None` | Company A session |
| `session_id_b` | `str\|None` | Company B session |
| `uploaded_filename_a` | `str\|None` | Tracks which file was uploaded for A (prevents re-upload on rerun) |
| `uploaded_filename_b` | `str\|None` | Tracks which file was uploaded for B |
| `messages` | `list[dict]` | Full chat history |
| `page_number` | `int` | Page for chart queries, default 1 |
| `api_url` | `str` | FastAPI base URL, default `http://localhost:8000` |

### Error Display

- If `POST /upload` fails: `st.error("Upload failed: <detail>")` in sidebar
- If `/chat/stream` returns `[ERROR]`: display error in red inside the assistant message bubble
- If FastAPI is unreachable on load: `st.error("Backend not reachable at http://localhost:8000 — is it running?")`

### File layout

```
frontend/
├── app.py          # All Streamlit code (~150 lines)
└── __init__.py     # Already exists (empty)
```

---

## Running the Full Stack

```bash
# Terminal 1 — FastAPI backend
uv run uvicorn backend.app:app --reload --port 8000

# Terminal 2 — Streamlit frontend
uv run streamlit run frontend/app.py
```

Both processes must be running. Streamlit checks `/health` on startup and shows an error if the backend is unreachable.

---

## Out of Scope

- Authentication / user accounts
- Persistent chat history across browser sessions
- File management UI (delete uploaded PDFs)
- Multi-user concurrency (single-user local tool)
