from __future__ import annotations

from typing import Optional, Annotated
from operator import add
from langgraph.graph import MessagesState
from langchain_core.documents import Document


class FinanceAgentState(MessagesState):

    # session tracking
    session_id: str
    session_id_b: Optional[str]

    question: Optional[str] = None

    # Router decision
    route: Optional[str] = None

    # Retrieved documents
    documents: Optional[list[Document]] = None
    docs_a: Optional[list[Document]] = None
    docs_b: Optional[list[Document]] = None

    # chart
    page_number: Optional[int] = None
    image_b64: Optional[str] = None

    # responses
    answer: Optional[str] = None
    structured_responses: Optional[dict] = None

    # long term memory
    long_term_summary: Optional[str] = None

    # Fan-out collectors — reducer allows parallel agent branches to write without conflict
    # Each item in partial_answers: {"route": str, "text": str}
    partial_answers: Annotated[list[dict], add] = []
    active_routes: Annotated[list[str], add] = []

    # conversation summaries
    summaries: Annotated[list[str], add] = []