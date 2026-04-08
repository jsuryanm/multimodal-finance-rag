from pydantic import Field,BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    session_id: str
    indexed: bool 
    message: Optional[str] = None 

class ChatRequest(BaseModel):
    session_id: str 
    question: str 

class ChatResponse(BaseModel):
    answer: str 
    route: Optional[str] = None 
    structured_data: Optional[dict] = None

class CompareRequest(BaseModel):
    session_id_a: str
    session_id_b: str
    question: str


class StockPriceRequest(BaseModel):
    ticker: str  # e.g. "D05.SI" for DBS


class StockPriceResponse(BaseModel):
    ticker: str
    price: Optional[float] = None
    currency: Optional[str] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    market_cap: Optional[str] = None
    error: Optional[str] = None
