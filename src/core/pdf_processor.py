from __future__ import annotations 
import base64
import uuid
from pathlib import Path

import fitz 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document 
from langchain_community.document_loaders import PyPDFLoader

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
    
    def extract_documents(self,pdf_path: Path) -> list[Document]:
        """Load document and split into chunks"""
        try:
            loader = PyPDFLoader(str(pdf_path))
            docs = loader.load()

        except Exception as e:
            raise PDFProcessingError(f"Cannot load pdf: {pdf_path.name}",detail=str(e))
        
        # filter out empty pages 
        docs = [doc for doc in docs if doc.page_content.strip()]

        if not docs:
            raise PDFProcessingError(f"No text extracted from {pdf_path.name}",
                                     detail="The pdf maybe image-only. Try OCR")
        
        splitter = RecursiveCharacterTextSplitter(separators=["\n\n","\n",". "," ",""],
                                                  chunk_size=settings.CHUNK_SIZE,
                                                  chunk_overlap=settings.CHUNK_OVERLAP)
        
        chunks = splitter.split_documents(docs)

        logger.info(f"Extracted {len(chunks)} chunks from {pdf_path.name} ({len(docs)} pages with text)")
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
            pix = doc[page_num].get_pixmap(matrix=fitz.Matrix(2,2))
            img_path = images_dir / f"page_{page_num + 1}.png" 
            pix.save(str(img_path))
            image_paths.append(img_path)

        doc.close()
        logger.info(f"Rendered {len(image_paths)} page images from {pdf_path.name}")        
        return image_paths
    
    @staticmethod
    def image_to_base64(image_path: Path) -> str:
        """Read a PNG file and return its base64 string for LLM vision APIs."""
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")
    

# if __name__ == "__main__":
#     try:
#         sample_pdf = Path(__file__).parent / Path("sample.pdf")

#         if not sample_pdf.exists():
#             raise FileNotFoundError("File not found place a pdf document")
        
#         pdf_processor = PDFProcessor()

#         pdf_bytes = sample_pdf.read_bytes()

#         saved_path = pdf_processor.save_pdf(filename="sample.pdf",
#                                             content=pdf_bytes)
        
#         docs = pdf_processor.extract_documents(saved_path)
#         images = pdf_processor.extract_page_images(saved_path)

#         logger.info(f"Extracted chunks: {len(docs)}") 
#         logger.info(f"Extracted images: {len(images)}")
        

#     except Exception as e:
#         logger.exception("PDFProcessor failed")