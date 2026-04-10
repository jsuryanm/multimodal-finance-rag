from __future__ import annotations
import base64
import json
import uuid
from pathlib import Path

import fitz

# Suppress non-fatal MuPDF warnings (colour-space errors, structure-tree noise).
# These are cosmetic — they don't affect extraction quality.
fitz.TOOLS.mupdf_display_errors(False)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata

from src.settings.config import settings
from src.exceptions.custom_exceptions import PDFProcessingError
from src.logger.custom_logger import logger


class PDFProcessor:
    """
    Handles everything related to turning a raw PDF into usable data.

    Responsibilities:
    1. Save the uploaded PDF to disk under a session folder
    2. Extract text → split into LangChain Document chunks
    3. Render each page as a PNG image (used by the chart agent)
    4. Detect which pages likely contain charts or tables
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_dir = settings.DATA_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Save ──────────────────────────────────────────────────────────

    def save_pdf(self, filename: str, content: bytes) -> Path:
        """Write the uploaded PDF bytes to the session folder and return the path."""
        dst = self.session_dir / filename
        dst.write_bytes(content)
        logger.info(f"Saved PDF: {dst}")
        return dst

    # ── Step 2: Extract text ──────────────────────────────────────────────────

    def extract_documents(self, pdf_path: Path) -> list[Document]:
        """
        Load the PDF and split it into text chunks for vector indexing.

        Uses UnstructuredPDFLoader with strategy="fast" (pdfminer backend) which
        gives semantically coherent units — paragraphs, titles, tables — before
        chunking. This avoids arbitrary mid-sentence splits.

        Metadata note: Unstructured uses 1-indexed 'page_number'. We add a
        0-indexed 'page' alias so the rest of the codebase stays consistent.
        """
        try:
            loader = UnstructuredPDFLoader(
                str(pdf_path),
                mode="elements",
                strategy="fast",  # pdfminer only — no OCR or detectron2 needed
            )
            docs = loader.load()
        except Exception as e:
            raise PDFProcessingError(f"Cannot load PDF: {pdf_path.name}", detail=str(e))

        # Unstructured adds rich nested metadata (coordinates, layout dicts) that
        # ChromaDB cannot store. Strip anything that isn't a primitive type.
        docs = filter_complex_metadata(docs)

        # Add 0-indexed 'page' for backward compatibility with retrieval code
        for doc in docs:
            if "page_number" in doc.metadata and "page" not in doc.metadata:
                doc.metadata["page"] = doc.metadata["page_number"] - 1

        # Drop empty elements (headers, footers, blank lines)
        docs = [doc for doc in docs if doc.page_content.strip()]

        if not docs:
            raise PDFProcessingError(
                f"No text extracted from {pdf_path.name}",
                detail="The PDF may be image-only. Consider OCR pre-processing.",
            )

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        chunks = splitter.split_documents(docs)

        # Drop chunks that are too short to be meaningful (page numbers, headers)
        chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

        logger.info(
            f"Extracted {len(chunks)} chunks from {pdf_path.name} "
            f"({len(docs)} elements, "
            f"{len({d.metadata.get('page_number') for d in docs})} pages)"
        )
        return chunks

    # ── Step 3: Render page images ────────────────────────────────────────────

    def extract_page_images(self, pdf_path: Path) -> list[Path]:
        """
        Render every PDF page as a PNG at 2x resolution.

        These images are used by the chart agent to send pages to a vision LLM.
        Stored at: data/<session_id>/page_images/page_N.png  (1-indexed)
        """
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise PDFProcessingError(
                f"Cannot open PDF for image extraction: {pdf_path.name}", detail=str(e)
            )

        images_dir = self.session_dir / "page_images"
        images_dir.mkdir(exist_ok=True)
        image_paths: list[Path] = []

        for page_num in range(len(doc)):
            # Matrix(2, 2) = 2x scale → ~144 DPI, clear enough for chart reading
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
            img_path = images_dir / f"page_{page_num + 1}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)

        doc.close()
        logger.info(f"Rendered {len(image_paths)} page images from {pdf_path.name}")
        return image_paths

    # ── Step 4: Detect chart pages ────────────────────────────────────────────

    def detect_chart_pages(self, pdf_path: Path) -> list[int]:
        """
        Scan each page and return 1-indexed page numbers that likely contain
        charts, graphs, or tables.

        Heuristics:
        - Page contains embedded raster images → likely a chart screenshot or photo
        - Page has more than 5 vector drawing operations → likely table borders or chart axes

        Results are saved to chart_pages.json so the chart agent can look up
        relevant pages without re-opening the PDF on every request.
        """
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise PDFProcessingError(
                f"Cannot open PDF for chart detection: {pdf_path.name}", detail=str(e)
            )

        chart_pages: list[int] = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            has_images = len(page.get_images()) > 0
            has_drawings = len(page.get_drawings()) > 5  # table lines, chart axes
            if has_images or has_drawings:
                chart_pages.append(page_num + 1)  # store as 1-indexed

        doc.close()

        chart_pages_path = self.session_dir / "chart_pages.json"
        chart_pages_path.write_text(json.dumps(chart_pages))
        logger.info(
            f"Detected {len(chart_pages)} chart pages in {pdf_path.name} "
            f"(saved to {chart_pages_path})"
        )
        return chart_pages

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def image_to_base64(image_path: Path) -> str:
        """Read a PNG file and return its base64 string for vision LLM APIs."""
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")