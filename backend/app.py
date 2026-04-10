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
