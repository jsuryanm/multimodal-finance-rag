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
