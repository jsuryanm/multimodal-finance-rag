from __future__ import annotations

import uuid
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from backend.models.api_schemas import UploadResponse, ChatRequest
from src.agents.orchestrator_agent import get_orchestrator
from src.core.embeddings import get_qwen_embeddings
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
    # Load the embedding model now so the first upload doesn't block mid-request.
    # asyncio.to_thread keeps the event loop responsive during the ~30s load time.
    logger.info("Loading embedding model…")
    await asyncio.to_thread(get_qwen_embeddings)
    logger.info("Embedding model ready.")
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

    try:
        processor = PDFProcessor(session_id=session_id)
        pdf_path = await asyncio.to_thread(processor.save_pdf, file.filename, content)
        chunks = await asyncio.to_thread(processor.extract_documents, pdf_path)
        images = await asyncio.to_thread(processor.extract_page_images, pdf_path)
        chart_pages = await asyncio.to_thread(processor.detect_chart_pages, pdf_path)

        store = VectorStore(session_id=session_id)
        await asyncio.to_thread(store.build_index, chunks)

    except FinDocBaseException:
        raise  # let the registered exception handler format it
    except Exception as e:
        logger.exception(f"Upload failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    logger.info(
        f"Uploaded {file.filename} → session {session_id} "
        f"({len(chunks)} chunks, {len(images)} pages, {len(chart_pages)} chart pages)"
    )

    return UploadResponse(
        session_id=session_id,
        filename=file.filename,
        pages=len(images),
        chunks=len(chunks),
        chart_pages=chart_pages,
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
