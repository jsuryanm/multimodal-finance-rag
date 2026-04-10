from __future__ import annotations

import base64
import json
import uuid
from pathlib import Path

import fitz
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import EasyOcrOptions, PdfPipelineOptions, TableFormerMode
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from langchain_core.documents import Document

from src.exceptions.custom_exceptions import PDFProcessingError
from src.logger.custom_logger import logger
from src.settings.config import settings

fitz.TOOLS.mupdf_display_errors(False)


class PDFProcessor:
    """Extracts text, page images, and chart metadata from annual report PDFs."""

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_dir = settings.DATA_DIR / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._docling_doc = None  # cached after first parse, reused by detect_chart_pages
        self._converter = self._build_converter()

    def _build_converter(self) -> DocumentConverter:
        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        opts.do_ocr = getattr(settings, "PDF_OCR_ENABLED", False)
        if opts.do_ocr:
            opts.ocr_options = EasyOcrOptions(lang=["en"])
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    def save_pdf(self, filename: str, content: bytes) -> Path:
        dst = self.session_dir / filename
        dst.write_bytes(content)
        logger.info(f"Saved PDF: {dst}")
        return dst

    def extract_documents(self, pdf_path: Path) -> list[Document]:
        """Parse PDF with Docling and return structure-aware LangChain Document chunks."""
        try:
            result = self._converter.convert(str(pdf_path))
        except Exception as e:
            raise PDFProcessingError(f"Docling failed to parse: {pdf_path.name}", detail=str(e))

        doc = result.document
        self._docling_doc = doc  # cache for detect_chart_pages

        chunker = HybridChunker(
            tokenizer="sentence-transformers/all-MiniLM-L6-v2",
            max_tokens=settings.CHUNK_SIZE,
            merge_peers=True,
        )

        try:
            raw_chunks = list(chunker.chunk(dl_doc=doc))
        except Exception as e:
            raise PDFProcessingError(f"Chunking failed for {pdf_path.name}", detail=str(e))

        if not raw_chunks:
            raise PDFProcessingError(
                f"No content extracted from {pdf_path.name}",
                detail="PDF may be image-only. Enable OCR via PDF_OCR_ENABLED=true.",
            )

        # Pages that have real tables — used to set has_tables metadata on chunks
        table_pages = {t.prov[0].page_no for t in doc.tables if t.prov}

        lc_docs = []
        for chunk in raw_chunks:
            text = chunker.serialize(chunk=chunk)
            if not text or len(text.strip()) < 50:
                continue

            page_1 = self._chunk_page(chunk)
            headings = " > ".join(chunk.meta.headings) if chunk.meta and chunk.meta.headings else ""

            lc_docs.append(Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_1 - 1,       # 0-indexed for ChromaDB
                    "page_number": page_1,     # 1-indexed for page_images filenames
                    "has_tables": page_1 in table_pages or "|" in text,
                    "headings": headings,
                },
            ))

        if not lc_docs:
            raise PDFProcessingError(f"All chunks were filtered for {pdf_path.name}")

        logger.info(f"Extracted {len(lc_docs)} chunks from {pdf_path.name}")
        return lc_docs

    def extract_page_images(self, pdf_path: Path) -> list[Path]:
        """Render each page as a 3x-resolution PNG for the vision LLM."""
        try:
            pdf = fitz.open(str(pdf_path))
        except Exception as e:
            raise PDFProcessingError(f"fitz cannot open: {pdf_path.name}", detail=str(e))

        images_dir = self.session_dir / "page_images"
        images_dir.mkdir(exist_ok=True)
        paths = []

        for i in range(len(pdf)):
            pix = pdf[i].get_pixmap(matrix=fitz.Matrix(3, 3), colorspace=fitz.csRGB)
            p = images_dir / f"page_{i + 1}.png"
            pix.save(str(p))
            paths.append(p)

        pdf.close()
        logger.info(f"Rendered {len(paths)} page images from {pdf_path.name}")
        return paths

    def detect_chart_pages(self, pdf_path: Path) -> list[dict]:
        """Identify pages with tables, raster figures, or vector charts and write chart_pages.json.

        Uses two complementary sources:
        - Docling: detects real data tables and embedded raster images (doc.tables, doc.pictures)
        - fitz:    detects vector-drawn charts (bar/line/pie charts exported from Excel or design
                   tools as PDF vector paths). Docling does NOT see these — they are not images.
        """
        doc = self._docling_doc or self._parse(pdf_path)
        page_data: dict[int, dict] = {}

        def _ensure(page_no: int):
            if page_no not in page_data:
                page_data[page_no] = {"page": page_no, "tables": 0, "figures": 0,
                                       "vector_charts": 0, "caption": ""}

        # Pass 1 — Docling: real tables and raster figures
        for elements, key in [(doc.tables, "tables"), (doc.pictures, "figures")]:
            for element in elements:
                if not element.prov:
                    continue
                page_no = element.prov[0].page_no
                _ensure(page_no)
                page_data[page_no][key] += 1
                if not page_data[page_no]["caption"]:
                    page_data[page_no]["caption"] = self._caption(element, doc)

        # Pass 2 — fitz: vector-drawn charts (bar charts, line graphs, pie charts)
        # Annual reports from SGX companies use vector charts almost exclusively —
        # they are produced by Excel/PowerPoint and exported as PDF vector paths,
        # not as embedded images. Docling's doc.pictures misses all of these.
        # get_drawings() returns every vector path on a page; a high count reliably
        # signals a chart. Threshold of 20 paths filters out decorative borders/lines.
        for page_no, path_count in self._vector_chart_pages(pdf_path).items():
            _ensure(page_no)
            page_data[page_no]["vector_charts"] = path_count

        chart_pages = sorted(page_data.values(), key=lambda x: x["page"])
        chart_pages = self._backfill_captions(chart_pages, doc)

        out = self.session_dir / "chart_pages.json"
        out.write_text(json.dumps(chart_pages, indent=2))

        n_vector = sum(1 for p in chart_pages if p["vector_charts"] > 0)
        logger.info(
            f"Detected {len(chart_pages)} chart pages in {pdf_path.name} "
            f"({sum(p['tables'] for p in chart_pages)} tables, "
            f"{sum(p['figures'] for p in chart_pages)} raster figures, "
            f"{n_vector} pages with vector charts)"
        )
        return chart_pages

    @staticmethod
    def _vector_chart_pages(pdf_path: Path, threshold: int = 20) -> dict[int, int]:
        """Return {1-indexed page_no: path_count} for pages that likely contain vector charts.

        fitz.page.get_drawings() returns every vector drawing operation on a page.
        Pages with >= threshold paths are flagged as likely chart pages.
        Decorative borders and dividers rarely exceed 5-10 paths; real charts
        (axes, bars, grid lines, data points) typically produce 20-200+ paths.
        """
        vector_pages: dict[int, int] = {}
        try:
            pdf = fitz.open(str(pdf_path))
            for i, page in enumerate(pdf):
                drawings = page.get_drawings()
                if len(drawings) >= threshold:
                    vector_pages[i + 1] = len(drawings)  # 1-indexed
            pdf.close()
        except Exception as e:
            # Non-fatal — vector detection is additive. Docling results still apply.
            logger.warning(f"Vector chart detection failed for {pdf_path.name}: {e}")
        return vector_pages

    def _parse(self, pdf_path: Path):
        """Parse PDF and cache the DoclingDocument. Used when called standalone."""
        try:
            result = self._converter.convert(str(pdf_path))
            self._docling_doc = result.document
            return self._docling_doc
        except Exception as e:
            raise PDFProcessingError(f"Docling parse failed: {pdf_path.name}", detail=str(e))

    @staticmethod
    def _chunk_page(chunk) -> int:
        """Return 1-indexed page number from a HybridChunker chunk, defaulting to 1."""
        try:
            items = chunk.meta.doc_items if chunk.meta else []
            if items and items[0].prov:
                return items[0].prov[0].page_no
        except (AttributeError, IndexError):
            pass
        return 1

    @staticmethod
    def _caption(element, doc) -> str:
        """Return the Docling-linked caption for a table or figure, or empty string."""
        try:
            text = element.caption_text(doc) if hasattr(element, "caption_text") else ""
            return (text or "").strip()[:120]
        except Exception:
            return ""

    @staticmethod
    def _backfill_captions(chart_pages: list[dict], doc) -> list[dict]:
        """Fill missing captions with the nearest section heading above the page."""
        page_headings: dict[int, str] = {}
        try:
            for item in doc.texts:
                if item.prov and "SECTION_HEADER" in str(getattr(item, "label", "")):
                    p = item.prov[0].page_no
                    page_headings.setdefault(p, item.text.strip()[:120])
        except Exception:
            pass

        last = ""
        for page in chart_pages:
            last = page_headings.get(page["page"], last)
            if not page["caption"]:
                page["caption"] = last or f"Page {page['page']}"
        return chart_pages

    @staticmethod
    def image_to_base64(image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")


# if __name__ == "__main__":
#     import sys
#     pdf_arg = sys.argv[1] if len(sys.argv) > 1 else next(
#         (str(p) for p in sorted(settings.DATA_DIR.glob("*/*.pdf"))), None
#     )
#     if not pdf_arg:
#         print("Usage: python -m src.core.pdf_processor <pdf_path>")
#         sys.exit(1)

#     proc = PDFProcessor()
#     path = Path(pdf_arg)
#     chunks = proc.extract_documents(path)
#     chart_pages = proc.detect_chart_pages(path)

#     print(f"Chunks: {len(chunks)} | Table chunks: {sum(1 for c in chunks if c.metadata['has_tables'])}")
#     print(f"Chart pages: {len(chart_pages)}")
#     for cp in chart_pages[:5]:
#         print(f"  p{cp['page']:>3}  tables={cp['tables']}  figures={cp['figures']}  -> {cp['caption']}")