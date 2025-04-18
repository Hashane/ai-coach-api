from pydantic import BaseModel


class ChatRequest(BaseModel):
    text: str
    conversation_id: int

class ChatResponse(BaseModel):
    reply: str