from __future__ import annotations 
import asyncio

from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.agents.state import FinanceAgentState
from src.settings.config import get_llm
from src.models.schemas import ComparisionSummary
from src.core.vector_store import VectorStore
from src.prompts.comparision_prompt import COMPARE_SYSTEM_PROMPT
from src.exceptions.custom_exceptions import AgentError
from src.logger.custom_logger import logger 

class ComparsionAgent:
    def __init__(self):
        self.llm = get_llm()

    def _format_docs(self,docs: list) -> str:
        return "\n\n---\n\n".join(
            f"[Page {doc.metadata.get('page','?')}]\n{doc.page_content}"
            for doc in docs
        )
    
    async def retrieve_both_nodes(self,state: FinanceAgentState) -> dict:
        """Retrieve relevant chunks from both companies ChromaDB indexes in parallel"""
        question = state["question"]
        session_id_a = state["session_id"]
        session_id_b = state["session_id_b"]

        if not session_id_b:
            raise AgentError("Comparision requires 2 uploaded documents",
                             detail="session_id_b is missing. Upload a second document first.")
        
        store_a = VectorStore(session_id=session_id_a)
        store_b = VectorStore(session_id=session_id_b)

        retriever_a = store_a.get_retriever(search_type="mmr",k=5)
        retriever_b = store_b.get_retriever(search_type="mmr",k=5)

        docs_a,docs_b = await asyncio.gather(retriever_a.ainvoke(question),
                                             retriever_b.ainvoke(question))
        
        logger.info(f"Retrieved {len(docs_a)} chunks from Company A document")
        logger.info(f"Retrieved {len(docs_b)} chunks from Company B document")

        return {"docs_a":docs_a,"docs_b":docs_b}
    
    async def compare_nodes(self,state: FinanceAgentState) -> dict:
        """Generates a structured side by side comparision table"""
        question = state["question"]
        docs_a = state.get('docs_a',[])
        docs_b = state.get('docs_b',[])

        context_a = self._format_docs(docs_a)
        context_b = self._format_docs(docs_b)

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",COMPARE_SYSTEM_PROMPT
            ),
            (
                "human",(
                    "COMPANY A ANNUAL REPORT CONTEXT:\n{context_a}\n\n"
                    "COMPANY B ANNUAL REPORT CONTEXT:\n{context_b}\n\n"
                    "USER QUESTION: {question}\n\n"
                    "Respond with a JSON object matching ComparisionSummary schema."
                    "CRITICAL: Do not include markdown code blocks, backticks (```), "
                    "or any preamble/post-amble text. The response must begin with {{ and end with }}."
                )
            )
        ]) 

        chain =  prompt | self.llm | JsonOutputParser()

        result  = await chain.ainvoke({
            "context_a":context_a,
            "context_b":context_b,
            "question":question
        })
        
        try:
            comparision_obj = ComparisionSummary(**result)
            answer = comparision_obj.final_verdict
            structured = comparision_obj.model_dump()

        except Exception as e:
            logger.warning(f"Could not parse ComparisionSummary: {e}")
            answer = str(result)
            structured = {"raw":answer}

        ai_message = AIMessage(content=answer)
        logger.info("Comparision analysis completed")
        return {"answer":answer,
                "structured_responses":structured,
                "messages":[ai_message]}
    
    async def run(self,state: FinanceAgentState):
        try:
            retrieve_update = await self.retrieve_both_nodes(state)
            state.update(retrieve_update)

            compare_update = await self.compare_nodes(state)
            state.update(compare_update)

            logger.info(f"ComparisionAgent completed for session {state['session_id']}")
            
            return {"answer":state.get("answer"),
                    "structured_responses":state.get("structured_responses"),
                    "messages":state.get("messages"),
                    "route":"comparision"}
        
        except Exception as e: 
            logger.error(f"ComparsionAgent failed: {e} | detail: {getattr(e, 'detail', '')}")
            raise AgentError("ComparsionAgent execution failed", detail=str(e))

        
    async def stream(self,state: FinanceAgentState):
        try:
            retrieve_update = await self.retrieve_both_nodes(state)
            state.update(retrieve_update)

            docs_a = state.get("docs_a",[])
            docs_b = state.get("docs_b",[])

            question = state["question"]

            context_a = self._format_docs(docs_a)
            context_b = self._format_docs(docs_b)

            prompt = ChatPromptTemplate.from_messages([
                ("system",COMPARE_SYSTEM_PROMPT),
                (
                    "human",
                    (
                        "COMPANY A ANNUAL REPORT:\n{context_a}\n\n"
                        "COMPANY B ANNUAL REPORT:\n{context_b}\n\n"
                        "QUESTION: {question}"
                    )
                )
            ])

            chain  = prompt | self.llm 

            async for chunk in chain.astream({
                "context_a":context_a,
                "context_b":context_b,
                "question":question
                                              }):
                if chunk.content:
                    yield chunk.content
            
        except Exception as e:
            raise AgentError("ComparisionAgent streaming failed",detail=str(e))