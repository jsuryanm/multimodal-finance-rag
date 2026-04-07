from __future__ import annotations

from datetime import datetime
from typing import Optional

import asyncpg

from pydantic import BaseModel

from src.settings.config import settings 
from src.logger.custom_logger import logger

class ConversationMemory(BaseModel):
    """Represent a stored memory entry for a session"""
    session_id: str 
    summary: str 
    created_at: datetime 
    updated_at: datetime

class LongTerm:
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
                    created_at  TIMESTAMPZ NOT NULL DEFAULT NOW(),
                    updated_at  TIMESTAMPZ NOT NULL DEFAULT NOW())
            """)
            
            logger.info("Long-term memory table ready")
        
        finally:
            await conn.close()
    
    async def get_memory(self,session_id: str) -> Optional[str]:
        """
        Retrieves the stored summary for a session 
        Returns None if no memory exists
        """
        conn = await self._get_connection()
        try:
            row = await conn.fetchrow("""
                    SELECT summary 
                        FROM conversation_memory 
                    WHERE session_id = $1""",session_id) # $1 insert the first parameter here safely
            
            return row["summary"] if row else None 
        
        finally:
            await conn.close()

    async def save_memory(self,session_id: str,summary: str) -> None:
        """Upsert the memory summary for a session. 
        Called after each conversation turn to update the summary"""