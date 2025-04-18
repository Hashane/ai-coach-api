from datetime import datetime

from pydantic import BaseModel
from typing import Optional, Literal


class ChatRequest(BaseModel):
    text: str
    conversation_id: Optional[int] = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: int


class ChatMessage(BaseModel):
    text: str
    sender: Literal["user", "bot"]
    timestamp: datetime

    class Config:
        orm_mode = True