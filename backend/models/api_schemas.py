from pydantic import BaseModel
from typing import Optional

class ChartPageInfo(BaseModel):
    page: int
    tables: int
    figures: int
    vector_charts: int
    caption: str


class UploadResponse(BaseModel):
    session_id: str
    filename: str
    pages: int
    chunks: int
    chart_pages: list[ChartPageInfo]  # 1-indexed page numbers detected as containing charts/tables


class ChatRequest(BaseModel):
    session_id: str
    question: str
    session_id_b: Optional[str] = None
    page_number: Optional[int] = 1
