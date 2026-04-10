from __future__ import annotations
import base64
import json
import uuid
from pathlib import Path

import fitz

# Suppress non-fatal MuPDF C-library messages (structure-tree errors,
# colour-space warnings). They are non-fatal and clutter logs.
# Warnings remain accessible via fitz.TOOLS.mupdf_warnings() if needed.
fitz.TOOLS.mupdf_display_errors(False)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredPDFLoader

from src.settings.config import settings 
from src.exceptions.custom_exceptions import PDFProcessingError
from src.logger.custom_logger import logger

class PDFProcessor:
    """
    1.Saves uploaded file to disk under a session folder 
    2. Extract text -> split into langchain documents
    3. Render each page as PNG image for chart agent"""

    def __init__(self,session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_dir = settings.DATA_DIR / self.session_id 
        self.session_dir.mkdir(parents=True,exist_ok=True)

    def save_pdf(self,filename: str,content: bytes) -> Path:
        """Saves pdf bytes to the session folder and returns the path"""
        dst = self.session_dir / filename
        dst.write_bytes(content)
        # write_byte method writes the raw binary data directly to file
        # it takes pdf file bytes from uploaded file and saves to disk 
        logger.info(f"Saved PDF: {dst}") 
        return dst 
    
    def extract_documents(self, pdf_path: Path) -> list[Document]:
        """Load PDF and split into chunks using Unstructured (pdfminer backend).

        Uses strategy="fast" which routes to pdfminer.six for pure-Python text
        extraction — no PyMuPDF, no structure-tree errors.  mode="elements" gives
        semantically coherent units (paragraphs, titles, tables) before splitting.

        Metadata normalisation: Unstructured sets 1-indexed 'page_number'; we add
        a 0-indexed 'page' alias so the rest of the codebase (chart auto-detect,
        FAISS retrieval) stays unchanged.
        """
        try:
            loader = UnstructuredPDFLoader(
                str(pdf_path),
                mode="elements",
                strategy="fast",   # pdfminer.six only — no detectron2/tesseract
            )
            docs = loader.load()
        except Exception as e:
            raise PDFProcessingError(f"Cannot load pdf: {pdf_path.name}", detail=str(e))

        # Normalise metadata: unstructured uses 1-indexed 'page_number';
        # add 0-indexed 'page' for backward compat with retrieval code.
        for doc in docs:
            if "page_number" in doc.metadata and "page" not in doc.metadata:
                doc.metadata["page"] = doc.metadata["page_number"] - 1

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

        # Strip low-quality chunks (headers, footers, very short lines)
        chunks = [c for c in chunks if len(c.page_content.strip()) > 50]

        logger.info(
            f"Extracted {len(chunks)} chunks from {pdf_path.name} "
            f"({len(docs)} elements across {len({d.metadata.get('page_number') for d in docs})} pages)"
        )
        return chunks
    
    def extract_page_images(self,pdf_path: Path) -> list[Path]:
        """Render every page as a PNG image. Returns list of image paths.
        Used by the chart agent to send page images to a vision LLM."""

        try:
            doc = fitz.open(str(pdf_path))
        
        except Exception as e:
            raise PDFProcessingError(f"Cannot open PDF for image extraction: {pdf_path.name}",
                                     detail=str(e))

        images_dir = self.session_dir / "page_images"
        images_dir.mkdir(exist_ok=True)

        image_paths: list[Path] = []

        for page_num in range(len(doc)):
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
            img_path = images_dir / f"page_{page_num + 1}.png" 
            pix.save(str(img_path))
            image_paths.append(img_path)

        doc.close()
        logger.info(f"Rendered {len(image_paths)} page images from {pdf_path.name}")        
        return image_paths
    
    
    def detect_chart_pages(self, pdf_path: Path) -> list[int]:
        """Scan every page with PyMuPDF and return 1-indexed page numbers that are
        likely to contain charts, graphs, or tables.

        Heuristics (either condition triggers a hit):
        - Page contains embedded raster images (photos, chart screenshots)
        - Page has more than 5 vector drawing operations (table grid lines, bar/line chart paths)

        Results are saved to data/<session_id>/chart_pages.json for fast lookup
        at query time without re-opening the PDF.
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
            # vector drawings: table borders, chart axes, bar fills, etc.
            has_drawings = len(page.get_drawings()) > 5
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

    @staticmethod
    def image_to_base64(image_path: Path) -> str:
        """Read a PNG file and return its base64 string for LLM vision APIs."""
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")
    

if __name__ == "__main__":
    try:
        sample_pdf = Path(__file__).parent / Path("sample.pdf")

        if not sample_pdf.exists():
            raise FileNotFoundError("File not found place a pdf document")
        
        pdf_processor = PDFProcessor()

        pdf_bytes = sample_pdf.read_bytes()

        saved_path = pdf_processor.save_pdf(filename="sample.pdf",
                                            content=pdf_bytes)
        
        docs = pdf_processor.extract_documents(saved_path)
        images = pdf_processor.extract_page_images(saved_path)

        logger.info(f"Extracted chunks: {len(docs)}") 
        logger.info(f"Extracted images: {len(images)}")
        

    except Exception as e:
        logger.exception("PDFProcessor failed")