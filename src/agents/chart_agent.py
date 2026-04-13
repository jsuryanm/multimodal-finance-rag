from __future__ import annotations
import asyncio
import json
import re
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from src.agents.state import FinanceAgentState
from src.core.pdf_processor import PDFProcessor
from src.core.vector_store import VectorStore
from src.exceptions.custom_exceptions import AgentError
from src.models.schemas import ChartAnalysis, ChartIntent
from src.prompts.chart_prompt import build_chart_prompt
from src.settings.config import get_vision_llm, settings
from src.logger.custom_logger import logger


# ── Chart type keyword map ────────────────────────────────────────────────────
# Maps natural-language keywords a user might say to a canonical chart type.
# Checked in order — first match wins. Keep "histogram" before "bar" so that
# "distribution histogram" doesn't collapse to "bar".

_CHART_TYPE_KEYWORDS: list[tuple[str, list[str]]] = [
    ("histogram",  ["histogram", "distribution", "frequency", "bins", "bin"]),
    ("waterfall",  ["waterfall", "bridge chart", "bridge graph", "contribution"]),
    ("scatter",    ["scatter", "scatter plot", "correlation", "bubble"]),
    ("heatmap",    ["heatmap", "heat map", "matrix", "correlation matrix"]),
    ("pie",        ["pie", "donut", "doughnut", "breakdown", "composition", "share"]),
    ("area",       ["area chart", "area graph", "stacked area", "cumulative"]),
    ("combo",      ["combo", "dual axis", "bar and line", "line and bar"]),
    ("line",       ["line chart", "line graph", "trend", "over time", "time series",
                    "growth", "trajectory"]),
    ("table",      ["table", "tabular", "row", "column", "grid", "figures"]),
    ("bar",        ["bar chart", "bar graph", "bar", "column chart", "column graph",
                    "grouped", "stacked bar", "revenue breakdown", "segment"]),
]


class ChartAgent:

    def __init__(self):
        self.vision_llm = get_vision_llm()


    @staticmethod
    def extract_chart_intent(question: str) -> ChartIntent:
        q = question.lower()

        page_match = re.search(r"\b(?:page|pg|p\.?)\s*(\d+)\b", q)
        explicit_page = int(page_match.group(1)) if page_match else None

        chart_type = "unknown"
        for ctype, keywords in _CHART_TYPE_KEYWORDS:
            if any(kw in q for kw in keywords):
                chart_type = ctype
                break

        _stop = {
            "the", "a", "an", "on", "in", "of", "is", "are", "show", "describe",
            "explain", "what", "does", "page", "chart", "graph", "table", "plot",
            "figure", "bar", "line", "histogram", "pie", "scatter", "this",
        }
        topic_keywords = [
            w for w in re.findall(r"[a-z]+", q)
            if w not in _stop and len(w) > 3
        ]

        return ChartIntent(
            chart_type=chart_type,
            explicit_page=explicit_page,
            topic_keywords=topic_keywords,
        )


    async def _find_best_chart_page(
        self,
        question: str,
        session_id: str,
        intent: Optional[ChartIntent] = None,
    ) -> int:
        chart_pages_path = settings.DATA_DIR / session_id / "chart_pages.json"

        if not chart_pages_path.exists():
            raise AgentError(
                f"chart_pages.json not found for session {session_id}",
                detail="Re-upload the PDF — it was processed with an older pipeline version.",
            )

        chart_pages: list[dict] = json.loads(chart_pages_path.read_text())
        if not chart_pages:
            raise AgentError(
                f"No charts detected in session {session_id}",
                detail="The PDF contains no visual elements.",
            )

        if intent and intent.explicit_page:
            available = {cp["page"] for cp in chart_pages}
            if intent.explicit_page in available:
                logger.info(f"Using explicit page {intent.explicit_page} from question")
                return intent.explicit_page

            logger.warning(
                f"Page {intent.explicit_page} is not in chart_pages.json — "
                "using anyway (user was explicit)"
            )
            return intent.explicit_page

        chart_type = intent.chart_type if intent else "unknown"
        best = self._rank_by_caption_and_type(question, chart_pages, chart_type)
        if best is not None:
            logger.info(
                f"Page {best['page']} selected (caption match, type='{chart_type}') "
                f"for session {session_id}"
            )
            return best["page"]

        try:
            store = VectorStore(session_id=session_id)
            retriever = store.get_retriever(search_type="mmr", k=10)
            
            docs = await asyncio.to_thread(retriever.invoke, question)
            
            chart_page_numbers = {cp["page"] for cp in chart_pages}
            
            overlap = [
                p for p in (doc.metadata.get("page", 0) + 1 for doc in docs)
                if p in chart_page_numbers
            ]
            
            if overlap:
                logger.info(f"Page {overlap[0]} selected by ChromaDB overlap")
                return overlap[0]
        except Exception as e:
            logger.warning(f"ChromaDB chart page detection failed: {e}")

        visual_pages = [cp for cp in chart_pages
                        if cp.get("tables", 0) > 0 
                        or cp.get("figures", 0) > 0 
                        or cp.get("vector_charts", 0) >= 20]
          
        candidates = visual_pages if visual_pages else chart_pages  

        densest = max(
            candidates,
            key=lambda cp: (
                cp.get("tables", 0) * 3 
                + cp.get("figures", 0) 
                + cp.get("vector_charts", 0) // 10
            ),
        )
        logger.info(f"Page {densest['page']} selected by visual density (fallback)")
        return densest["page"]

    @staticmethod
    def _rank_by_caption_and_type(
        question: str,
        chart_pages: list[dict],
        chart_type: str,
    ) -> dict | None:
        """Score pages by caption overlap, boosted by chart type.

        Chart-type boost: pages whose caption contains the chart type word
        (or a synonym) get +5 bonus points, making them prefer the page the
        user actually asked about when the chart type is explicit.
        """
        stop = {
            "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
            "are", "was", "were", "show", "what", "which", "where", "how",
            "me", "my", "our", "their", "this", "that", "chart", "graph",
            "table", "describe", "explain",
        }

        # Chart type → caption synonyms to boost
        _type_synonyms: dict[str, list[str]] = {
            "bar":       ["bar", "column", "revenue", "segment"],
            "line":      ["line", "trend", "growth", "over time"],
            "histogram": ["histogram", "distribution", "frequency"],
            "pie":       ["pie", "donut", "breakdown", "composition"],
            "scatter":   ["scatter", "correlation"],
            "waterfall": ["waterfall", "bridge"],
            "area":      ["area", "cumulative"],
            "heatmap":   ["heatmap", "matrix"],
            "table":     ["table", "figures", "financial"],
            "combo":     ["combined", "dual"],
        }
        type_hints = _type_synonyms.get(chart_type, [])

        def tokens(text: str) -> set[str]:
            return {
                t for t in re.findall(r"[a-z0-9]+", (text or "").lower())
                if t not in stop and len(t) > 2
            }

        q_tokens = tokens(question)
        if not q_tokens:
            return None

        best_score, best_entry = 0, None
        for cp in chart_pages:
            caption_tokens = tokens(cp.get("caption", ""))
            score = len(q_tokens & caption_tokens) * 10

            # Tiebreaker: visual density
            score += cp.get("tables", 0) * 3 + min(cp.get("vector_charts", 0) // 10, 3)

            # Chart-type boost
            if any(hint in caption_tokens for hint in type_hints):
                score += 5

            if score > best_score:
                best_score = score
                best_entry = cp

        return best_entry if best_score >= 10 else None


    async def load_image_node(self, state: FinanceAgentState) -> dict:
        session_id = state["session_id"]
        question = state.get("question", "")

        intent = self.extract_chart_intent(question)
        logger.info(f"Chart intent: type='{intent.chart_type}', page={intent.explicit_page}")


        page_number = state.get("page_number")
        if not page_number:
            page_number = await self._find_best_chart_page(question, session_id, intent)
            logger.info(f"Auto-selected page {page_number} for session {session_id}")

        images_dir = settings.DATA_DIR / session_id / "page_images"
        image_path = images_dir / f"page_{page_number}.png"

        if not image_path.exists():
            raise AgentError(
                f"Page image not found: page {page_number}",
                detail=f"Expected path: {image_path}. Ensure PDF was uploaded and images were rendered.",
            )

        image_b64 = PDFProcessor.image_to_base64(image_path)
        logger.info(f"Loaded page {page_number} image for session {session_id}")

        return {
            "image_b64": image_b64,
            "page_number": page_number,
            "_chart_intent": intent,
        }


    async def analyze_image_node(self, state: FinanceAgentState) -> dict:
        """Send the page image to the vision LLM and parse a structured ChartAnalysis."""
        image_b64 = state.get("image_b64")
        page_number = state.get("page_number", 1)
        question = state.get("question") or f"Describe charts on page {page_number}"

        intent: ChartIntent = state.get("_chart_intent") or self.extract_chart_intent(question)
        system_prompt = build_chart_prompt(intent.chart_type)

        message = HumanMessage(content=[
            {
                "type": "text",
                "text": (
                    f"{system_prompt}\n\n"
                    f"User question: {question}\n\n"
                    f"This is page {page_number} of an SGX annual report. "
                    "Describe all financial visuals with investment-grade precision. "
                    "Respond ONLY with the JSON object — no markdown, no preamble."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
        ])

        chain = self.vision_llm | JsonOutputParser()
        
        result = await chain.ainvoke([message])

        try:
            analysis_obj = ChartAnalysis(**result)
            answer = analysis_obj.explanation or analysis_obj.key_insight or str(result)
            structured = analysis_obj.model_dump()
        except Exception as e:
            logger.warning(f"Could not parse ChartAnalysis: {e}")
            answer = str(result)
            structured = {"raw": answer}

        ai_message = AIMessage(content=answer)
        logger.info(f"Chart analysis complete for page {page_number}")
        return {
            "answer": answer,
            "structured_responses": structured,
            "messages": [ai_message],
        }

    async def run(self, state: FinanceAgentState) -> dict:
        try:
            image_update = await self.load_image_node(state)
            state.update(image_update)
            analysis_update = await self.analyze_image_node(state)
            state.update(analysis_update)
            logger.info(f"ChartAgent workflow complete for session {state['session_id']}")
            return {
                "answer": state.get("answer"),
                "structured_responses": state.get("structured_responses"),
                "messages": state.get("messages"),
                "image_b64": state.get("image_b64"),
                "route": "chart",
            }
        except Exception as e:
            raise AgentError("ChartAgent execution failed", detail=str(e))

    async def stream(self, state: FinanceAgentState):
        try:
            image_update = await self.load_image_node(state)
            state.update(image_update)

            image_b64 = state.get("image_b64")
            page_number = state.get("page_number")
            question = state.get("question") or f"Describe charts on page {page_number}"
            intent: ChartIntent = state.get("_chart_intent") or self.extract_chart_intent(question)
            system_prompt = build_chart_prompt(intent.chart_type)

            message = HumanMessage(content=[
                {
                    "type": "text",
                    "text": (
                        f"{system_prompt}\n\n"
                        f"User question: {question}\n\n"
                        f"Page {page_number} of SGX annual report. "
                        "Describe all financial visuals precisely."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ])

            async for chunk in self.vision_llm.astream([message]):
                if chunk.content:
                    yield chunk.content

        except Exception as e:
            raise AgentError("ChartAgent streaming failed", detail=str(e))