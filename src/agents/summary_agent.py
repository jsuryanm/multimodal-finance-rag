from __future__ import annotations

from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph,START,END

from src.agents.state import FinanceAgentState
from src.models.schemas import FinancialSummary
from src.core.vector_store import VectorStore
from src.memory.checkpoint import get_checkpointer
from src.exceptions.custom_exceptions import AgentError
from src.prompts.summary_prompt import SUMMARY_SYSTEM_PROMPT
from src.memory.long_term import get_long_term_memory
from src.settings.config import get_llm
from src.logger.custom_logger import logger 

class SummaryAgent:
    def __init__(self):
        self.llm = get_llm()
        self.memory_store = get_long_term_memory()

    def format_docs(self,docs):
        return "\n\n---\n\n".join(f"[Page {doc.metadata.get('page','?')}]\n{doc.page_content}"
                                  for doc in docs)
    
    def build_rag_chain(self,retriever):
        prompt = ChatPromptTemplate.from_messages([
            ("system",SUMMARY_SYSTEM_PROMPT),
            ("human",(
                "LONG-TERM MEMORY (previous conversations):\n{memory}\n\n"
                "RETRIEVED CONTEXT:\n{context}\n\n"
                "QUESTION: {question}\n\n"
                "Respond with a JSON object matching FinancialSummary schema."
                "CRITICAL: Do not include markdown code blocks, backticks (```), or any preamble/post-amble text. The response must begin with { and end with }."
            ))
        ])

        chain = (
            {
                "context":retriever | self.format_docs,
                "question":RunnablePassthrough(),
                "memory":RunnablePassthrough()
            }
            | prompt 
            | self.llm
            | JsonOutputParser()
        )

        return chain 
    
    async def load_memory_node(self,state: FinanceAgentState) -> dict:
        """Load the long term memory summary from PostgreSQL
        This tell agent what was discussed in previous sessions."""
        summary = await self.memory_store.get_memory(state["session_id"])
        logger.info(f"Loaded long-term memory for state {state["session_id"]}")
        return {"long_term_summary":summary or "No previous conversation history"}

    async def retrieve_node(self,state: FinanceAgentState) -> dict:
        """Use FAISS vector store to retrieve relevant chunks for the question.
        Uses MMR (Maximal Marginal Relevance) to reduce redundancy in results.
        """
        question = state["question"]
        session_id = state["session_id"]

        store = VectorStore(session_id=session_id)
        retriever = store.get_retriever(search_type="mmr",k=5)

        docs = await retriever.ainvoke(question)
        logger.info(f"Retrieved {len(docs)} chunks for session {session_id}")
        return {"documents":docs}
    
    async def generate_node(self,state: FinanceAgentState) -> dict:
        """
        Generate a structured financial answer using the retrieved documents.
        Uses the LLM with structured output (FinancialSummary schema).
        """
        question = state["question"]
        documents = state.get("documents",[])
        memory = state.get("long_term_summary","")
        session_id = state["session_id"]

        store = VectorStore(session_id=session_id)
        retriever = store.get_retriever(search_type="mmr",k=5)

        rag_chain = self.build_rag_chain(retriever)
        
        result = await rag_chain.ainvoke({"question":question,
                                          "memory":memory})
        
        try:
            summary_obj = FinancialSummary(**result)
            answer = summary_obj.summary
            structured = summary_obj.model_dump()
        except AgentError:
            answer = str(result)
            structured = {"raw":answer}

        ai_message = AIMessage(content=answer)
        logger.info(f"Generated answer for session {session_id}")
        return {"answer":answer,
                "structured_response":structured,
                "messages":[ai_message]}
    

    async def save_memory_node(self,state: FinanceAgentState) -> dict:
        """Update the long term memory with summary of conversation"""
        session_id = state["session_id"]
        question = state.get("question","")
        answer = state.get("answer","")
        existing_memory = state.get("long_term_summary","")

        summary_prompt = (
            f"Previous memory: {existing_memory}\n\n"
            f"New exchange:\nUser asked: {question}\nAgent answered: {answer[:500]}...\n\n"
            "Write a brief 2-3 sentence summary of what has been discussed. "
            "Focus on financial topics, companies, and metrics mentioned."
        )

        response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
        new_summary = response.content
        
        await self.memory_store.save_memory(session_id,new_summary)
        logger.info(f"Memory updated for session {session_id}")
        
        return {"summaries":[new_summary]}

    async def build_graph(self,state:FinanceAgentState):
        checkpointer = await get_checkpointer()
        
        graph = StateGraph(state)
        
        graph.add_node("load_memory",self.load_memory_node)
        graph.add_node("retrieve",self.retrieve_node)
        graph.add_node("generate",self.generate_node)
        graph.add_node("save_memory",self.save_memory_node)

        graph.add_edge(START,"load_memory")
        graph.add_edge("load_memory","retrieve")
        graph.add_edge("retrieve","generate")
        graph.add_edge("generate","save_memory")
        graph.add_edge("save_memory",END)

        return graph.compile(checkpointer=checkpointer)