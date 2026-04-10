from __future__ import annotations
import asyncio
import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from src.agents.state import FinanceAgentState
from src.core.pdf_processor import PDFProcessor
from src.core.vector_store import VectorStore
from src.exceptions.custom_exceptions import AgentError
from src.models.schemas import ChartAnalysis
from src.prompts.chart_prompt import CHART_DESCRIPTION_PROMPT
from src.settings.config import get_vision_llm, settings
from src.logger.custom_logger import logger

class ChartAgent:
    """Vision based chart and table analysis agent
    Flow: load_image -> analyze_image"""

    def __init__(self):
        self.vision_llm = get_vision_llm()

    async def _find_best_chart_page(self, question: str, session_id: str) -> int:
        """Return the most question-relevant chart page for this session.

        Strategy:
        1. Load the pre-computed chart_pages.json (built at upload time by PDFProcessor).
        2. Run a ChromaDB MMR retrieval with the user's question to find text-relevant pages.
           PyPDFLoader stores 0-indexed 'page' metadata; add 1 to convert to 1-indexed.
        3. Return the first chart page that also appears in the ChromaDB results.
        4. If no overlap, return the first chart page overall.
        5. If chart_pages.json is missing, fall back to page 1.
        """
        chart_pages_path = settings.DATA_DIR / session_id / "chart_pages.json"

        if not chart_pages_path.exists():
            logger.warning(f"chart_pages.json not found for session {session_id} — defaulting to page 1")
            return 1

        chart_pages: list[int] = json.loads(chart_pages_path.read_text())
        if not chart_pages:
            logger.warning(f"No chart pages detected for session {session_id} — defaulting to page 1")
            return 1

        # Use ChromaDB to find text-relevant pages, then intersect with chart pages
        try:
            store = VectorStore(session_id=session_id)
            retriever = store.get_retriever(search_type="mmr", k=10)
            docs = await asyncio.to_thread(retriever.invoke, question)

            # PyPDFLoader page metadata is 0-indexed → convert to 1-indexed
            retrieved_pages = {doc.metadata.get("page", 0) + 1 for doc in docs}
            relevant_chart_pages = [p for p in chart_pages if p in retrieved_pages]

            if relevant_chart_pages:
                chosen = relevant_chart_pages[0]
                logger.info(f"Auto-detected chart page {chosen} (ChromaDB+heuristic) for session {session_id}")
                return chosen

        except Exception as e:
            logger.warning(f"ChromaDB chart page detection failed: {e} — falling back to first chart page")

        logger.info(f"Using first known chart page {chart_pages[0]} for session {session_id}")
        return chart_pages[0]

    async def load_image_node(self, state: FinanceAgentState) -> dict:
        session_id = state["session_id"]
        page_number = state.get("page_number")
        if not page_number:
            question = state.get("question", "")
            page_number = await self._find_best_chart_page(question, session_id)
            logger.info(f"No page_number provided — auto-selected page {page_number}")

        images_dir = settings.DATA_DIR / session_id / "page_images"
        image_path = images_dir / f"page_{page_number}.png"

        if not image_path.exists():
            raise AgentError(f"Path image not found for page {page_number}",
                             detail=(f"Expected: {image_path}"
                                     "Ensure the PDF was uploaded and page images were rendered"))
    
        image_b64 = PDFProcessor.image_to_base64(image_path)
        logger.info(f"Loaded page {page_number} image for session {session_id}")
        return {"image_b64":image_b64}
    
    async def analyze_image_node(self,state: FinanceAgentState) -> dict:
        """Sends page image to vision llm for chart/table analysis.
        Buids a multimodal message: text prompt + base64-encoded image"""
        image_b64 = state.get("image_b64")
        page_number = state.get("page_number",1)
        question = state.get("question") or f"Describe charts and tables on page {page_number}"

        if not image_b64:
            raise AgentError("No image loaded cannot run chart analysis",
                             detail="Image does not exist")
        
        message = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    f"{CHART_DESCRIPTION_PROMPT}\n\n"
                    f"User question: {question}\n\n"
                    f"This is page {page_number} of annual report."
                    "Describe all financial visuals precisely."
                    "CRITICAL: No markdown code blocks, no backticks, no preamble."
                    "Response must start with { and end with }."
                )
            },
            {
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/png;base64,{image_b64}",
                    "detail":"high"
                }
            }
        ])

        response = await self.vision_llm.ainvoke([message])

        try:
            parser = JsonOutputParser()
            result = parser.parse(response.content)

            chart_obj = ChartAnalysis(**result)
            answer = chart_obj.explanation
            structured = chart_obj.model_dump()
        
        except Exception as e:
            logger.warning(f"Could not parse ChartAnalysis: {e}")
            answer = response.content
            structured =  {"raw": answer}
        
        ai_message = AIMessage(content=answer)
        logger.info(f"Chart analysis complete for page {page_number}")

        return {"answer":answer,
                "structured_responses": structured,
                "messages":[ai_message]}
    
    async def run(self,state: FinanceAgentState) -> dict:
        """Full chart analysis workflow called by orchestrator"""
        try: 
            image_update = await self.load_image_node(state)
            state.update(image_update)

            analysis_update = await self.analyze_image_node(state)
            state.update(analysis_update)

            logger.info(f"ChartAgent workflow for session {state['session_id']}")

            return {"answer":state.get("answer"),
                    "structured_responses":state.get("structured_responses"),
                    "messages":state.get("messages"),
                    "image_b64":state.get("image_b64"),
                    "route":"chart"}
        
        except Exception as e:
            raise AgentError("ChartAgent execution failed",detail=str(e))

    async def stream(self,state: FinanceAgentState):
        try:
            image_update = await self.load_image_node(state)
            state.update(image_update)

            image_b64 = state.get("image_b64")
            page_number = state.get("page_number")
            question = state.get("question") or f"Describe charts on page {page_number}"
            message = HumanMessage(content=[
                {
                    "type":"text",
                    "text":(
                        f"{CHART_DESCRIPTION_PROMPT}\n\n"
                        f"User question: {question}\n\n"
                        f"This is page {page_number} of annual report."
                        "Describe all financial visuals precisely."
                    )
                },

                {
                    "type":"image_url",
                    "image_url":{
                        "url":
                        f"data:image/png;base64,{image_b64}"
                    }
                }

            ])

            async for chunk in self.vision_llm.astream([message]):
                if chunk.content:
                    yield chunk.content
            
        except Exception as e:
            raise AgentError("ChartAgent streaming failed")

