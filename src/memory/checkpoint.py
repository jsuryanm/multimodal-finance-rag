from __future__ import annotations 

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from src.settings.config import settings 
from src.logger.custom_logger import logger


async def get_checkpointer():
    """Returns async sqlite checkpointer for LangGraph
    checkpointer saves full graph state at every node
    if request fails we can resume from last checkpoint"""

    # checkpointer creates DB connection
    db_path = settings.SQLITE_CHECKPOINT
    db_path.parent.mkdir(parents=True,exist_ok=True)

    checkpointer_cm = AsyncSqliteSaver.from_conn_string(str(db_path))
    checkpointer = await checkpointer_cm.__aenter__()
  
    await checkpointer.setup()
    logger.info("SQLite checkpointer initialized")
  
    return checkpointer_cm,checkpointer
