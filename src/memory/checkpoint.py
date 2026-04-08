from __future__ import annotations 

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.settings.config import settings 
from src.logger.custom_logger import logger


async def get_checkpointer() -> AsyncSqliteSaver:
    """Returns async sqlite checkpointer for LangGraph
    checkpointer saves full graph state at every node
    if request fails we can resume from last checkpoint"""

    # checkpointer creates DB connection
    checkpointer = AsyncSqliteSaver.from_conn_string(settings.SQLITE_CHECKPOINT)
    await checkpointer.setup() # automatically creates tables 
    logger.info("PostgreSQL checkpointer initialized")
    return checkpointer
