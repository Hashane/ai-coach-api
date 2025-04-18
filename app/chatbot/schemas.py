from pydantic import BaseModel
from typing import Optional


class ChatRequest(BaseModel):
    text: str
    conversation_id: Optional[int] = None


class ChatResponse(BaseModel):
    reply: str
    conversation_id: int
