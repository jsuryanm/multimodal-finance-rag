import io
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
