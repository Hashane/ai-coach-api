from fastapi import Depends, HTTPException, status, APIRouter
from app.auth.dependencies import authenticate_user, get_current_user, get_session_local
from app.auth.schemas import User, Token
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.auth.utils import create_access_token
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
