from __future__ import annotations

from typing import Optional,Annotated
# annotated attaches extra metadata to typehint 
from operator import add 
from langgraph.graph import MessagesState
from langchain_core.messages import AnyMessage 
from langchain_core.documents import Document 


class FinanceAgentState(MessagesState):

    # session tracking
    session_id: str # refers to company a's annual report 
    session_id_b: str | None # refers to company b's annual report 
    # these are for the comparision agent

    question: Optional[str] = None

    # Router decision
    route: Optional[str] = None  

    # Retrieved documents
    documents: Optional[list[Document]] = None
    docs_a: Optional[list[Document]] = None 
    docs_b: Optional[list[Document]] = None

    # chart
    page_number: Optional[int] = None 
    image_b64: Optional[int] = None

    # responses
    answer: Optional[str] = None 
    
    # structured output
    structured_responses: Optional[dict] = None 

    # conversation summaries
    summaries: Annotated[list[str],add] = []