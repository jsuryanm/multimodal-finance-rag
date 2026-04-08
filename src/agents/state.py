from __future__ import annotations

from typing import Optional,Annotated
# annotated attaches extra metadata to typehint 
from operator import add 
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage 
from langchain_core.documents import Document 


class FinanceAgentState(MessagesState):

    # session tracking
    session_id: str # refers to company a report session 
    session_id_b: str | None # refers to company b report session (comparision only)

    question: Optional[str] = None

    # Router decision (Orchestrator Decision)
    # summary, chart, comparision, stock_price
    route: Optional[str] = None  

    # Retrieved documents
    documents: Optional[list[Document]] = None # For summary agent
    docs_a: Optional[list[Document]] = None # comparision agent (company A)
    docs_b: Optional[list[Document]] = None # comparision agent (company B)

    # chart
    page_number: Optional[int] = None 
    image_b64: Optional[str] = None

    # responses
    answer: Optional[str] = None 
    # structured output
    structured_responses: Optional[dict] = None 

    # long term memory (injected from postgresql at the start of each run)
    long_term_summary: Optional[str] = None 

    # conversation summaries
    summaries: Annotated[list[str],add] = []