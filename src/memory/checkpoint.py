from __future__ import annotations 

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from src.settings.config import settings 
from src.logger.custom_logger import logger


async def get_checkpointer() -> AsyncPostgresSaver:
    """Returns PostgreSQL checkpointer for LangGraph
    checkpointer saves full graph state at every node
    if request fails we can resume from last checkpoint"""

    # checkpointer creates DB connection
    checkpointer = AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL)
    await checkpointer.setup() # automatically creates tables 
    logger.info("PostgreSQL checkpointer initialized")
    return checkpointer
