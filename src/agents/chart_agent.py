from __future__ import annotations 

from langchain_core.messages import AIMessage,HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from src.agents.state import FinanceAgentState
from src.models.schemas import ChartAnalysis
from src.core.pdf_processor import PDFProcessor

from src.prompts.chart_prompt import CHART_DESCRIPTION_PROMPT
from src.settings.config import settings,get_vision_llm
from src.logger.custom_logger import logger 
from src.exceptions.custom_exceptions import AgentError

class ChartAgent: 
    """Vision based chart and table analysis agent
    Flow: load_image -> analyze_image"""

    def __init__(self):
        self.vision_llm = get_vision_llm()

    async def load_image_node(self,state: FinanceAgentState) -> dict:
        session_id = state["session_id"]
        page_number = state.get("page_number") or 1 

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

