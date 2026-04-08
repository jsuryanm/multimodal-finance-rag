from __future__ import annotations

from datetime import datetime
from typing import Optional

import aiosqlite
from pydantic import BaseModel

from src.settings.config import settings 
from src.logger.custom_logger import logger

DB_PATH = settings.DATA_DIR / "memory.db"

class ConversationMemory(BaseModel):
    """Represent a stored memory entry for a session"""
    session_id: str 
    summary: str 
    created_at: datetime 
    updated_at: datetime

class LongTermMemory:
    """
    Manages long-term memory in PostgreSQL.
    
    For each session, we store a running summary of what the user
    has discussed. This gets injected into new conversations so the
    agent has context from previous chats.
    
    Table: conversation_memory
    - session_id (PK)
    - summary: text summary of past conversations
    - created_at, updated_at: timestamps
    """

    def __init__(self):
        self.db_path = DB_PATH

    async def setup(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS converation_memory(
                    
                        session_id TEXT PRIMARY KEY,
                        summary    TEXT NOT NULL DEFAULT '',
                        created_at TIMESTAMP  DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP  DEFAULT CURRENT_TIMESTAMP                   
                )""") 
            await db.commit()
            logger.info("SQLite memory table ready")

    async def _get_connection(self) -> asyncpg.Connection:
        """Get the raw async conneciton to PostGRESQL"""
        return await asyncpg.connect(settings.DATABASE_URL)
    
    async def setup(self) -> None:
        """Create the memory table if doesn't exist"""
        conn = await self._get_connection()

        try:
            await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_memory (
                    session_id  TEXT PRIMARY KEY,
                    summary     TEXT NOT NULL DEFAULT '',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW())
            """)
            
            logger.info("Long-term memory table ready")
        
        finally:
            await conn.close()
    
    async def get_memory(self,session_id: str) -> Optional[str]:
        """
        Retrieves the stored summary for a session 
        Returns None if no memory exists
        """
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
        """Upsert the memory summary for a session. 
        Called after each conversation turn to update the summary"""
        
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
        """Clear memory for a session (useful for testing or reset)."""
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