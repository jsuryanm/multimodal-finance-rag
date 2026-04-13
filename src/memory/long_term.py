from __future__ import annotations

from datetime import datetime
from typing import Optional

import aiosqlite
from pydantic import BaseModel

from src.settings.config import settings 
from src.logger.custom_logger import logger


class ConversationMemory(BaseModel):
    session_id: str 
    summary: str 
    created_at: datetime 
    updated_at: datetime

class LongTermMemory:

    def __init__(self):
        self.db_path = settings.SQLITE_MEMORY_DB

    async def setup(self) -> None:
        self.db_path.parent.mkdir(parents=True,exist_ok=True)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_memory(
                    
                        session_id TEXT PRIMARY KEY,
                        summary    TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMP  DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP  DEFAULT CURRENT_TIMESTAMP                   
                )""") 
            await db.commit()
            logger.info("SQLite memory table ready")
        
    async def get_memory(self,session_id: str) -> Optional[str]:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                SELECT summary 
                    FROM conversation_memory
                WHERE session_id = ?
                """,(session_id,))
            
            row = await cursor.fetchone()

            return row[0] if row else None


    async def save_memory(self,session_id: str,summary: str) -> None:
        
        async with aiosqlite.connect(self.db_path) as db:
            
            await db.execute(
                """INSERT INTO conversation_memory (
                    session_id,
                    summary,
                    updated_at
                )
                
                VALUES (?,?,CURRENT_TIMESTAMP) 
                ON CONFLICT(session_id)
                DO UPDATE SET
                    summary = excluded.summary,
                    updated_at = CURRENT_TIMESTAMP
                """,(session_id,summary)
            )

            await db.commit()
        
        logger.info(f"Memory saved: {session_id}")



    async def delete_memory(self,session_id: str) -> None:
        async with aiosqlite.connect(self.db_path) as db:

            await db.execute("""
                    DELETE FROM conversation_memory
                        WHERE session_id = ?
            """,(session_id,))

            await db.commit()

_long_term_memory: Optional[LongTermMemory] = None

def get_long_term_memory() -> LongTermMemory:
    global _long_term_memory
    if _long_term_memory is None:
        _long_term_memory = LongTermMemory()
    return _long_term_memory