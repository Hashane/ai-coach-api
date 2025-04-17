from fastapi import Depends, APIRouter
from app.auth.dependencies import get_current_user, get_session_local
from app.auth.schemas import User
from sqlalchemy.orm import Session
from app.chatbot.engine import get_similar_response
from app.chatbot.schemas import ChatResponse, ChatRequest

router = APIRouter(
    prefix="/chatbot",
    tags=["chatbot"],
)


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user),
                  db: Session = Depends(get_session_local)):
    reply = get_similar_response(request.text, current_user, db)
    return ChatResponse(reply=reply)
