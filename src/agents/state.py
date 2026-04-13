from __future__ import annotations

from typing import Optional, Annotated
from operator import add
from langgraph.graph import MessagesState
from langchain_core.documents import Document


class FinanceAgentState(MessagesState):

    session_id: str
    session_id_b: Optional[str]

    question: Optional[str] = None

    route: Optional[str] = None

    documents: Optional[list[Document]] = None
    docs_a: Optional[list[Document]] = None
    docs_b: Optional[list[Document]] = None

    page_number: Optional[int] = None
    image_b64: Optional[str] = None

    answer: Optional[str] = None
    structured_responses: Optional[dict] = None

    long_term_summary: Optional[str] = None

    summaries: Annotated[list[str], add] = []