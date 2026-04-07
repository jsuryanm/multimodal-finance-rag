from __future__ import annotations

from typing import TypedDict,Annotated
from operator import add 

from langchain_core.messages import AnyMessage 
from langchain_core.documents import Document 


class FinanceAgentState(TypedDict):

    # conversation memory
    messages: Annotated[list[AnyMessage],add]

    session_id: str # refers to company a's annual report 
    session_id_b: str | None # refers to company b's annual report 
    # these are for the comparision agent

    # Router decision
    route: str 

    # Retrieved documents
    documents: list[Document]
    docs_a: list[Document]
    docs_b: list[Document]

    # chart
    page_number: int 
    image_b64: str 
    answer: str