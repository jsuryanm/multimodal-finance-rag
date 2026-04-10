from __future__ import annotations
import asyncio
import json
import re

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

        chart_pages.json now stores rich entries:
            [{"page": 5, "tables": 2, "images": 1, "graphics": 14,
              "caption": "Revenue by segment FY2024"}, ...]

        Strategy:
        1. Load chart_pages.json. If missing or empty, raise — don't silently
           guess page 1 (old behaviour masked failures).
        2. Score each chart page by caption/question keyword overlap and pick
           the top-scoring page.
        3. If no caption keywords match, fall back to ChromaDB MMR retrieval ∩
           chart pages and return the first intersection.
        4. Final fallback: the chart page with the most tables+graphics (i.e.
           the most visually dense page), not just the first entry.
        """
        chart_pages_path = settings.DATA_DIR / session_id / "chart_pages.json"

        if not chart_pages_path.exists():
            raise AgentError(
                f"chart_pages.json not found for session {session_id}",
                detail="The PDF was not processed with the current pipeline. Re-upload the PDF.",
            )

        chart_pages: list[dict] = json.loads(chart_pages_path.read_text())
        if not chart_pages:
            raise AgentError(
                f"No charts, graphs, or tables detected in session {session_id}",
                detail="The uploaded PDF contains no visual elements the chart agent can analyse.",
            )

        # 1) Caption keyword ranking — cheap, deterministic, no extra LLM call.
        best = self._rank_by_caption(question, chart_pages)
        if best is not None:
            logger.info(
                f"Chart page {best['page']} selected by caption match "
                f"('{best['caption']}') for session {session_id}"
            )
            return best["page"]

        # 2) Fall back to ChromaDB MMR ∩ chart pages.
        try:
            store = VectorStore(session_id=session_id)
            retriever = store.get_retriever(search_type="mmr", k=10)
            docs = await asyncio.to_thread(retriever.invoke, question)
            chart_page_numbers = {cp["page"] for cp in chart_pages}
            overlap = [p for p in (doc.metadata.get("page", 0) + 1 for doc in docs) if p in chart_page_numbers]
            if overlap:
                chosen = overlap[0]
                logger.info(f"Chart page {chosen} selected by ChromaDB overlap for session {session_id}")
                return chosen
        except Exception as e:
            logger.warning(f"ChromaDB chart page detection failed: {e} — using visual-density fallback")

        # 3) Visual-density fallback: pick the most table/graphics-heavy page.
        densest = max(chart_pages, key=lambda cp: cp.get("tables", 0) * 3 + cp.get("figures", 0) + cp.get("vector_charts", 0) // 10)

        logger.info(
            f"Chart page {densest['page']} selected by visual density "
            f"(tables={densest.get('tables', 0)}, graphics={densest.get('graphics', 0)}) "
            f"for session {session_id}"
        )
        return densest["page"]

    @staticmethod
    def _rank_by_caption(question: str, chart_pages: list[dict]) -> dict | None:
        """Score chart pages by token overlap between caption and question.

        Returns the best-matching entry, or None if no caption shares any
        non-stopword token with the question.
        """
        stop = {
            "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
            "are", "was", "were", "show", "what", "which", "whats", "where",
            "how", "me", "my", "our", "their", "this", "that", "these", "those",
            "chart", "graph", "table", "describe", "explain",
        }

        def tokens(text: str) -> set[str]:
            return {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in stop and len(t) > 2}

        q_tokens = tokens(question)
        if not q_tokens:
            return None

        best_score = 0
        best_entry: dict | None = None
        for cp in chart_pages:
            score = len(q_tokens & tokens(cp.get("caption", "")))
            # Tiebreaker: prefer pages with more tables
            score = score * 10 + cp.get("tables", 0) * 3 + min(cp.get("vector_charts", 0) // 10, 3)

            if score > best_score:
                best_score = score
                best_entry = cp

        # Require at least one caption keyword match (score >= 10 after the *10 bump)
        return best_entry if best_score >= 10 else None

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

