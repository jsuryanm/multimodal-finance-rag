from __future__ import annotations
import base64
import json
import re
import uuid
from pathlib import Path

import fitz
import pymupdf4llm

# Suppress non-fatal MuPDF warnings (colour-space errors, structure-tree noise).
# These are cosmetic — they don't affect extraction quality.
fitz.TOOLS.mupdf_display_errors(False)

# Stick with the classic pymupdf_rag backend rather than the newer layout engine.
# It's more stable for financial PDFs and has well-documented output structure
# (each page_chunk dict contains: metadata, toc_items, tables, images, graphics, text, words).
pymupdf4llm.use_layout(False)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.settings.config import settings
from src.exceptions.custom_exceptions import PDFProcessingError
from src.logger.custom_logger import logger


class PDFProcessor:
    """
    Handles everything related to turning a raw PDF into usable data.

    Responsibilities:
    1. Save the uploaded PDF to disk under a session folder
    2. Extract text as markdown (tables preserved inline) → LangChain Document chunks
    3. Render each page as a PNG image (used by the chart agent)
    4. Detect which pages contain charts or tables and label them with captions
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_dir = settings.DATA_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        # Cached page_chunks output from pymupdf4llm — populated by extract_documents()
        # and reused by detect_chart_pages() to avoid re-parsing the PDF.
        self._page_chunks: list[dict] | None = None

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
        Extract per-page markdown (tables preserved inline) and split into chunks.

        Uses pymupdf4llm.to_markdown(page_chunks=True) which returns a list of
        page dicts: {metadata, toc_items, tables, images, graphics, text, words}.
        The text field is markdown — headings become '#', tables become pipe-
        separated rows — so table values survive chunking and can be retrieved
        from the vector store by the summary agent.
        """
        try:
            page_chunks = pymupdf4llm.to_markdown(
                str(pdf_path),
                page_chunks=True,
                show_progress=False,
            )
        except Exception as e:
            raise PDFProcessingError(f"Cannot load PDF: {pdf_path.name}", detail=str(e))

        # Cache for detect_chart_pages() so we don't re-parse the PDF
        self._page_chunks = page_chunks

        # Build one LangChain Document per page. Keep metadata primitive so
        # ChromaDB can store it — no nested dicts or coordinate arrays.
        docs: list[Document] = []
        for chunk in page_chunks:
            text = chunk.get("text", "").strip()
            if not text:
                continue
            page_0 = chunk["metadata"].get("page", 0)  # pymupdf4llm uses 0-indexed
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_path),
                        "page": page_0,              # 0-indexed — matches existing retrieval code
                        "page_number": page_0 + 1,   # 1-indexed — matches page_images/page_N.png
                        "has_tables": len(chunk.get("tables", [])) > 0,
                    },
                )
            )

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

        # Drop chunks too short to be meaningful (page numbers, headers, footers)
        chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

        logger.info(
            f"Extracted {len(chunks)} chunks from {pdf_path.name} "
            f"({len(docs)} pages, {sum(1 for d in docs if d.metadata['has_tables'])} with tables)"
        )
        return chunks

    # ── Step 3: Render page images ────────────────────────────────────────────

    def extract_page_images(self, pdf_path: Path) -> list[Path]:
        """
        Render every PDF page as a PNG at 3x resolution (~216 DPI).

        Higher DPI than before so small chart numbers and table values survive
        for the vision LLM. Stored at:
            data/<session_id>/page_images/page_N.png  (1-indexed)
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
            # Matrix(3, 3) = 3x scale → ~216 DPI. Costs ~2x disk vs 2x but makes
            # small numbers in charts/tables legible to the vision LLM.
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(3, 3), colorspace=fitz.csRGB)
            img_path = images_dir / f"page_{page_num + 1}.png"
            pix.save(str(img_path))
            image_paths.append(img_path)

        doc.close()
        logger.info(f"Rendered {len(image_paths)} page images from {pdf_path.name}")
        return image_paths

    # ── Step 4: Detect chart pages ────────────────────────────────────────────

    def detect_chart_pages(self, pdf_path: Path) -> list[dict]:
        """
        Identify pages that contain charts, graphs, or tables, and label each
        one with a caption drawn from the nearest markdown heading or bold line.

        Signals:
        - `page_chunk['tables']` — real tables detected by pymupdf4llm (not a
          drawing-ops heuristic). Any page with ≥1 table counts.
        - `page_chunk['images']` — embedded raster images. Filtered to only
          include "meaningful" ones (i.e. pages that also have non-trivial text
          or graphics), which filters out decorative cover-page images.
        - `page_chunk['graphics']` — vector drawings. ≥10 suggests a chart.

        Output shape (saved to chart_pages.json, 1-indexed page numbers):
            [
                {
                    "page": 5,
                    "tables": 2,
                    "images": 1,
                    "graphics": 14,
                    "caption": "Revenue by segment FY2024"
                },
                ...
            ]
        """
        # Reuse cached chunks from extract_documents() if available,
        # otherwise re-parse (e.g. when called standalone in tests).
        if self._page_chunks is not None:
            page_chunks = self._page_chunks
        else:
            try:
                page_chunks = pymupdf4llm.to_markdown(
                    str(pdf_path),
                    page_chunks=True,
                    show_progress=False,
                )
            except Exception as e:
                raise PDFProcessingError(
                    f"Cannot open PDF for chart detection: {pdf_path.name}",
                    detail=str(e),
                )

        chart_pages: list[dict] = []
        for chunk in page_chunks:
            tables = chunk.get("tables") or []
            images = chunk.get("images") or []
            graphics = chunk.get("graphics") or []
            text = chunk.get("text", "") or ""

            has_table = len(tables) > 0
            has_meaningful_image = len(images) > 0 and len(text.strip()) > 100
            has_complex_graphics = len(graphics) >= 10  # likely chart axes/bars

            if not (has_table or has_meaningful_image or has_complex_graphics):
                continue

            page_1 = chunk["metadata"].get("page", 0) + 1  # store 1-indexed
            caption = self._extract_caption(text, page_1)

            chart_pages.append(
                {
                    "page": page_1,
                    "tables": len(tables),
                    "images": len(images),
                    "graphics": len(graphics),
                    "caption": caption,
                }
            )

        chart_pages_path = self.session_dir / "chart_pages.json"
        chart_pages_path.write_text(json.dumps(chart_pages, indent=2))
        logger.info(
            f"Detected {len(chart_pages)} chart pages in {pdf_path.name} "
            f"(saved to {chart_pages_path})"
        )
        return chart_pages

    # ── Caption helper ────────────────────────────────────────────────────────

    @staticmethod
    def _extract_caption(markdown_text: str, page_number: int) -> str:
        """
        Pick a human-readable caption for a chart page from its markdown text.

        Priority:
        1. First markdown heading (`# ...`, `## ...`) on the page.
        2. First **bold** line.
        3. First non-empty line that looks like a title (<80 chars, no pipes).
        4. Generic fallback: "Page {N}".
        """
        lines = [ln.strip() for ln in markdown_text.splitlines() if ln.strip()]

        for ln in lines:
            m = re.match(r"^#{1,6}\s+(.+?)\s*#*$", ln)
            if m:
                return m.group(1).strip()[:120]

        for ln in lines:
            m = re.match(r"^\*\*(.+?)\*\*\s*$", ln)
            if m:
                return m.group(1).strip()[:120]

        for ln in lines:
            if "|" in ln or ln.startswith("-") or ln.startswith("```"):
                continue
            if 3 < len(ln) < 80:
                return ln[:120]

        return f"Page {page_number}"

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def image_to_base64(image_path: Path) -> str:
        """Read a PNG file and return its base64 string for vision LLM APIs."""
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")


if __name__ == "__main__":
    import sys

    pdf_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if not pdf_arg:
        # Pick the first PDF we can find under data/ for a smoke test
        for p in sorted(settings.DATA_DIR.glob("*/*.pdf")):
            pdf_arg = str(p)
            break
    if not pdf_arg:
        print("Usage: python -m src.core.pdf_processor <pdf_path>")
        sys.exit(1)

    proc = PDFProcessor()
    path = Path(pdf_arg)
    chunks = proc.extract_documents(path)
    chart_pages = proc.detect_chart_pages(path)
    print(f"chunks: {len(chunks)}")
    print(f"chart pages: {len(chart_pages)}")
    for cp in chart_pages[:5]:
        print(f"  p{cp['page']:>3}  tables={cp['tables']}  images={cp['images']}  graphics={cp['graphics']}  -> {cp['caption']}")
    # Show one chunk that contains a markdown table to verify table preservation
    for c in chunks:
        if "|" in c.page_content and "---" in c.page_content:
            print("\n--- sample chunk with table ---")
            print(c.page_content[:600])
            break
