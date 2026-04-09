# Frontend & Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI backend and Streamlit frontend that expose the existing LangGraph orchestrator as a streaming HTTP API with a chat UI.

**Architecture:** FastAPI owns the orchestrator lifecycle (warms up on startup). Streamlit calls FastAPI exclusively via HTTP using `requests`. All routing is done by the orchestrator internally — Streamlit sends every question to one endpoint (`POST /chat/stream`) and receives an SSE stream whose first event is `[ROUTE:xxx]` followed by content tokens.

**Tech Stack:** FastAPI, Uvicorn, Streamlit, requests, pytest, starlette TestClient

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `backend/app.py` | Create | FastAPI app — lifespan, /health, /upload, /chat/stream, error handlers |
| `backend/models/schemas.py` | Modify | Update `UploadResponse` and `ChatRequest` to match spec |
| `src/agents/orchestrator_agent.py` | Modify | `stream()` emits `[ROUTE:xxx]` as first token |
| `frontend/app.py` | Create | Streamlit UI — sidebar, upload, chat, SSE streaming |
| `tests/test_backend.py` | Create | pytest tests for all three backend endpoints |

---

## Task 1: Update Schemas + Backend Skeleton

**Files:**
- Modify: `backend/models/schemas.py`
- Create: `backend/app.py`
- Create: `tests/test_backend.py`

- [ ] **Step 1: Update `backend/models/schemas.py`**

Replace the file contents with:

```python
from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    session_id: str
    filename: str
    pages: int
    chunks: int

class ChatRequest(BaseModel):
    session_id: str
    question: str
    session_id_b: Optional[str] = None
    page_number: Optional[int] = 1
```

- [ ] **Step 2: Write the failing health test**

Create `tests/test_backend.py`:

```python
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    mock_orchestrator = MagicMock()
    mock_orchestrator.stream = AsyncMock()

    with patch("backend.app.get_orchestrator", new_callable=AsyncMock, return_value=mock_orchestrator):
        from backend.app import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
```

- [ ] **Step 3: Run test to confirm it fails**

```bash
uv run pytest tests/test_backend.py::test_health -v
```

Expected: `ModuleNotFoundError` or `ImportError` (backend/app.py is empty).

- [ ] **Step 4: Create `backend/app.py` with lifespan and health**

```python
from __future__ import annotations

import uuid
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from backend.models.schemas import UploadResponse, ChatRequest
from src.agents.orchestrator_agent import get_orchestrator
from src.core.pdf_processor import PDFProcessor
from src.core.vector_store import VectorStore
from src.exceptions.custom_exceptions import (
    FinDocBaseException, VectorStoreError, PDFProcessingError,
    AgentError, OrchestratorError,
)
from src.logger.custom_logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up orchestrator…")
    await get_orchestrator()
    logger.info("Orchestrator ready.")
    yield


app = FastAPI(title="Finance RAG API", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 5: Run health test to confirm it passes**

```bash
uv run pytest tests/test_backend.py::test_health -v
```

Expected: `PASSED`.

- [ ] **Step 6: Commit**

```bash
git add backend/app.py backend/models/schemas.py tests/test_backend.py
git commit -m "feat: backend skeleton — lifespan, health endpoint, updated schemas"
```

---

## Task 2: Backend Upload Endpoint

**Files:**
- Modify: `backend/app.py`
- Modify: `tests/test_backend.py`

- [ ] **Step 1: Write failing upload tests**

Append to `tests/test_backend.py`:

```python
import io


def test_upload_missing_file(client):
    response = client.post("/upload")
    assert response.status_code == 422


def test_upload_not_pdf(client):
    data = io.BytesIO(b"not a pdf")
    response = client.post(
        "/upload",
        files={"file": ("report.txt", data, "text/plain")},
    )
    assert response.status_code == 422
    assert "PDF" in response.json()["detail"]


def test_upload_pdf(client):
    mock_processor = MagicMock()
    mock_processor.session_id = "test-session-123"
    mock_processor.save_pdf.return_value = MagicMock()
    mock_processor.extract_documents.return_value = [MagicMock()] * 10
    mock_processor.extract_page_images.return_value = [MagicMock()] * 5

    mock_store = MagicMock()
    mock_store.build_index.return_value = None

    with patch("backend.app.PDFProcessor", return_value=mock_processor), \
         patch("backend.app.VectorStore", return_value=mock_store):
        pdf_bytes = b"%PDF-1.4 fake pdf content"
        response = client.post(
            "/upload",
            files={"file": ("DBS_2024.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["session_id"], str) and len(body["session_id"]) > 0
    assert body["filename"] == "DBS_2024.pdf"
    assert body["chunks"] == 10
    assert body["pages"] == 5
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_backend.py::test_upload_missing_file tests/test_backend.py::test_upload_not_pdf tests/test_backend.py::test_upload_pdf -v
```

Expected: all FAIL (endpoint not implemented yet).

- [ ] **Step 3: Add upload endpoint to `backend/app.py`**

Add after the `health` function:

```python
@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")

    content = await file.read()
    session_id = str(uuid.uuid4())

    processor = PDFProcessor(session_id=session_id)
    pdf_path = await asyncio.to_thread(processor.save_pdf, file.filename, content)
    chunks = await asyncio.to_thread(processor.extract_documents, pdf_path)
    images = await asyncio.to_thread(processor.extract_page_images, pdf_path)

    store = VectorStore(session_id=session_id)
    await asyncio.to_thread(store.build_index, chunks)

    logger.info(f"Uploaded {file.filename} → session {session_id} ({len(chunks)} chunks, {len(images)} pages)")

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        pages=len(images),
        chunks=len(chunks),
    )
```

- [ ] **Step 4: Run upload tests to confirm they pass**

```bash
uv run pytest tests/test_backend.py::test_upload_missing_file tests/test_backend.py::test_upload_not_pdf tests/test_backend.py::test_upload_pdf -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/app.py tests/test_backend.py
git commit -m "feat: backend upload endpoint with PDF validation and FAISS indexing"
```

---

## Task 3: Emit Route from Orchestrator Stream

**Files:**
- Modify: `src/agents/orchestrator_agent.py` (stream method, lines 355–390)

The `stream()` method currently yields only content tokens. The spec requires that the first yielded value is `[ROUTE:xxx]` so the frontend can display the route badge.

- [ ] **Step 1: Replace the `stream()` body in `src/agents/orchestrator_agent.py`**

Find the block:
```python
        async for event in self._app.astream_events(
            initial_state,config=config,version="v2"):
            
            if event.get("event") != "on_chat_model_stream":
                continue
                
            node = event.get("metadata", {}).get("langgraph_node", "")
            if node in ("route", "load_memory", "save_memory"):
                continue

            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content
```

Replace with:

```python
        route_emitted = False
        async for event in self._app.astream_events(
            initial_state, config=config, version="v2"):

            # Capture route from the route node's chain-end event and emit first
            if not route_emitted and event.get("event") == "on_chain_end":
                node = event.get("metadata", {}).get("langgraph_node", "")
                if node == "route":
                    output = event.get("data", {}).get("output", {})
                    route = output.get("route", "summary") if isinstance(output, dict) else "summary"
                    yield f"[ROUTE:{route}]"
                    route_emitted = True
                    continue

            if event.get("event") != "on_chat_model_stream":
                continue

            node = event.get("metadata", {}).get("langgraph_node", "")
            if node in ("route", "load_memory", "save_memory"):
                continue

            chunk = event["data"].get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield chunk.content
```

- [ ] **Step 2: Manually verify the change looks correct**

```bash
grep -n "route_emitted" src/agents/orchestrator_agent.py
```

Expected: two lines — the `route_emitted = False` declaration and the `if not route_emitted` check.

- [ ] **Step 3: Commit**

```bash
git add src/agents/orchestrator_agent.py
git commit -m "feat: orchestrator stream emits [ROUTE:xxx] as first token"
```

---

## Task 4: Backend Chat/Stream Endpoint + Error Handling

**Files:**
- Modify: `backend/app.py`
- Modify: `tests/test_backend.py`

- [ ] **Step 1: Write failing streaming tests**

Append to `tests/test_backend.py`:

```python
def test_chat_stream_missing_session(client):
    response = client.post("/chat/stream", json={"question": "What is revenue?"})
    assert response.status_code == 422


def test_chat_stream_returns_sse(client):
    async def mock_stream(*args, **kwargs):
        yield "[ROUTE:summary]"
        yield "Revenue was SGD 14.3B"

    mock_orchestrator = MagicMock()
    mock_orchestrator.stream = mock_stream

    with patch("backend.app.get_orchestrator", new_callable=AsyncMock, return_value=mock_orchestrator):
        with client.stream(
            "POST", "/chat/stream",
            json={"session_id": "abc", "question": "What is revenue?"},
        ) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            lines = [line for line in response.iter_lines() if line]

    assert "data: [ROUTE:summary]" in lines
    assert "data: Revenue was SGD 14.3B" in lines
    assert "data: [DONE]" in lines


def test_chat_stream_error_becomes_sse_error(client):
    from src.exceptions.custom_exceptions import VectorStoreError

    async def mock_stream_error(*args, **kwargs):
        raise VectorStoreError("No FAISS index found for session: abc")
        yield  # make it a generator

    mock_orchestrator = MagicMock()
    mock_orchestrator.stream = mock_stream_error

    with patch("backend.app.get_orchestrator", new_callable=AsyncMock, return_value=mock_orchestrator):
        with client.stream(
            "POST", "/chat/stream",
            json={"session_id": "abc", "question": "What is revenue?"},
        ) as response:
            lines = [line for line in response.iter_lines() if line]

    assert any("[ERROR]" in line for line in lines)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_backend.py::test_chat_stream_missing_session tests/test_backend.py::test_chat_stream_returns_sse tests/test_backend.py::test_chat_stream_error_becomes_sse_error -v
```

Expected: all FAIL.

- [ ] **Step 3: Add chat/stream endpoint and error handler to `backend/app.py`**

Add after the upload endpoint:

```python
@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    orchestrator = await get_orchestrator()

    async def generate():
        try:
            async for chunk in orchestrator.stream(
                question=request.question,
                session_id=request.session_id,
                session_id_b=request.session_id_b,
                page_number=request.page_number,
            ):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except FinDocBaseException as e:
            yield f"data: [ERROR] {e}\n\n"
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e}")
            yield f"data: [ERROR] Internal server error\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.exception_handler(FinDocBaseException)
async def fin_doc_exception_handler(request, exc: FinDocBaseException):
    from fastapi.responses import JSONResponse
    status = 422 if isinstance(exc, (VectorStoreError, PDFProcessingError)) else 500
    hints = {
        VectorStoreError: "Upload a PDF first before asking questions.",
        PDFProcessingError: "Check the PDF is not image-only (needs extractable text).",
        AgentError: "An agent failed internally. Check app.log for details.",
        OrchestratorError: "The router failed. Check your LLM API key.",
    }
    return JSONResponse(
        status_code=status,
        content={
            "detail": str(exc),
            "hint": hints.get(type(exc), "Check app.log for details."),
        },
    )
```

- [ ] **Step 4: Run all streaming tests**

```bash
uv run pytest tests/test_backend.py -v
```

Expected: all PASSED.

- [ ] **Step 5: Commit**

```bash
git add backend/app.py tests/test_backend.py
git commit -m "feat: backend chat/stream SSE endpoint with error handling"
```

---

## Task 5: Streamlit Skeleton + Sidebar

**Files:**
- Create: `frontend/app.py`

- [ ] **Step 1: Create `frontend/app.py` with skeleton and sidebar**

```python
from __future__ import annotations

import requests
import streamlit as st

API_URL = "http://localhost:8000"

ROUTE_BADGES = {
    "summary":     ("📄", "summary",    "#1f6feb"),
    "chart":       ("🖼",  "chart",      "#8957e5"),
    "comparision": ("⚖️", "comparision","#f0883e"),
    "stock_price": ("📈", "stock_price","#3fb950"),
}

# ── session state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "session_id": None,
        "session_id_b": None,
        "uploaded_filename_a": None,
        "uploaded_filename_b": None,
        "messages": [],
        "page_number": 1,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── backend health check ──────────────────────────────────────────────────────
def _check_backend() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


# ── sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar():
    with st.sidebar:
        st.title("📊 Finance RAG")
        st.divider()

        # Company A upload
        st.markdown("**Company A**")
        file_a = st.file_uploader(
            "Upload annual report PDF",
            type="pdf",
            key="uploader_a",
            label_visibility="collapsed",
        )
        if file_a and file_a.name != st.session_state.uploaded_filename_a:
            _upload_file(file_a, slot="a")

        if st.session_state.session_id:
            sid = st.session_state.session_id
            st.caption(f"Session: `{sid[:8]}…{sid[-4:]}`")

        st.divider()

        # Company B upload
        st.markdown("**Company B** *(optional — enables comparison)*")
        file_b = st.file_uploader(
            "Upload second PDF",
            type="pdf",
            key="uploader_b",
            label_visibility="collapsed",
        )
        if file_b and file_b.name != st.session_state.uploaded_filename_b:
            _upload_file(file_b, slot="b")

        if st.session_state.session_id_b:
            sid = st.session_state.session_id_b
            st.caption(f"Session: `{sid[:8]}…{sid[-4:]}`")

        st.divider()

        # Chart page number
        st.markdown("**Chart page**")
        st.session_state.page_number = st.number_input(
            "Page number for chart analysis",
            min_value=1,
            value=st.session_state.page_number,
            label_visibility="collapsed",
        )

        st.divider()

        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ── upload helper ─────────────────────────────────────────────────────────────
def _upload_file(file, slot: str):
    with st.sidebar:
        with st.spinner(f"Indexing {file.name}…"):
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files={"file": (file.name, file.getvalue(), "application/pdf")},
                    timeout=300,
                )
                response.raise_for_status()
                data = response.json()
                if slot == "a":
                    st.session_state.session_id = data["session_id"]
                    st.session_state.uploaded_filename_a = file.name
                else:
                    st.session_state.session_id_b = data["session_id"]
                    st.session_state.uploaded_filename_b = file.name
                st.success(f"✓ {file.name} — {data['chunks']} chunks, {data['pages']} pages")
            except requests.exceptions.HTTPError as e:
                detail = e.response.json().get("detail", str(e))
                st.error(f"Upload failed: {detail}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot reach backend. Is it running?")


# ── chat history ──────────────────────────────────────────────────────────────
def _render_chat():
    for msg in st.session_state.messages:
        role = msg["role"]
        with st.chat_message(role):
            if role == "assistant" and msg.get("route"):
                route = msg["route"]
                emoji, label, color = ROUTE_BADGES.get(route, ("🤖", route, "#8b949e"))
                st.markdown(
                    f'<span style="font-size:11px;background:#21262d;'
                    f'border:1px solid #30363d;border-radius:10px;'
                    f'padding:1px 8px;color:{color};">{emoji} {label}</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(msg["content"])


# ── streaming chat ────────────────────────────────────────────────────────────
def _stream_answer(question: str):
    payload = {
        "session_id": st.session_state.session_id,
        "session_id_b": st.session_state.session_id_b,
        "question": question,
        "page_number": st.session_state.page_number,
    }

    route = "summary"

    def token_generator():
        nonlocal route
        with requests.post(
            f"{API_URL}/chat/stream", json=payload, stream=True, timeout=120
        ) as response:
            for raw_line in response.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                if data.startswith("[ERROR]"):
                    yield f"⚠️ {data[7:].strip()}"
                    break
                if data.startswith("[ROUTE:"):
                    route = data[7:-1]
                    continue
                yield data

    with st.chat_message("assistant"):
        full_response = st.write_stream(token_generator())

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response or "",
        "route": route,
    })


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="Finance RAG", page_icon="📊", layout="wide")
    _init_state()

    if not _check_backend():
        st.error(
            "⚠️ Backend not reachable at `http://localhost:8000` — "
            "run `uv run uvicorn backend.app:app --reload --port 8000` first."
        )
        return

    _render_sidebar()
    _render_chat()

    if not st.session_state.session_id:
        st.info("Upload a PDF in the sidebar to start asking questions.")
        return

    if question := st.chat_input("Ask about the annual report…"):
        st.session_state.messages.append({"role": "user", "content": question, "route": None})
        with st.chat_message("user"):
            st.markdown(question)
        try:
            _stream_answer(question)
        except requests.exceptions.ConnectionError:
            st.error("Lost connection to backend.")
        st.rerun()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file was created**

```bash
python -c "import ast; ast.parse(open('frontend/app.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
git add frontend/app.py
git commit -m "feat: Streamlit frontend — sidebar, upload, chat UI, SSE streaming"
```

---

## Task 6: End-to-End Smoke Test

- [ ] **Step 1: Start the backend**

```bash
uv run uvicorn backend.app:app --reload --port 8000
```

Watch for: `Orchestrator ready.` in the logs. This confirms MCP connected and graph compiled.

- [ ] **Step 2: Start the frontend in a second terminal**

```bash
uv run streamlit run frontend/app.py
```

Open `http://localhost:8501`.

- [ ] **Step 3: Verify health check passes**

In the browser, the app should load without the "Backend not reachable" error.

- [ ] **Step 4: Test stock price (no PDF needed)**

In the chat input, type: `What is DBS stock price?`

Expected:
- Route badge: `📈 stock_price` (green)
- Answer streams in: `DBS Group Holdings Ltd\nPrice: SGD …`

- [ ] **Step 5: Upload a PDF and test summary**

Upload any annual report PDF in the Company A sidebar slot.
Wait for the "✓ filename — N chunks, N pages" confirmation.
Ask: `What is the net profit?`

Expected:
- Route badge: `📄 summary` (blue)
- Answer streams from the RAG pipeline

- [ ] **Step 6: Run the full test suite**

```bash
uv run pytest tests/ -v
```

Expected: all PASSED.

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "feat: complete frontend/backend — FastAPI SSE + Streamlit chat UI"
git push origin main
```
