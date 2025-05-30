from typing import List

from fastapi import Depends, APIRouter
from app.auth.dependencies import get_current_user, get_session_local
from app.auth.schemas import User
from sqlalchemy.orm import Session
from app.chatbot.engine import chatbot_instance
from app.chatbot.schemas import ChatResponse, ChatRequest, ChatMessage, ChatConversation
from app.chatbot.trainer import ChatbotTrainer, initialize_knowledge_bases
from app.chatbot.utils import generate_title_from_message
from app.db.models import MessageHistory, Conversation

router = APIRouter(
    prefix="/chatbot",
    tags=["chatbot"],
)


@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(
        request: ChatRequest,
        current_user: User = Depends(get_current_user),
        db: Session = Depends(get_session_local),
):
    conversation_id = request.conversation_id

    # If no conversation_id, create a new conversation
    if not conversation_id:
        new_convo = Conversation(
            user_id=current_user.id,
            title= generate_title_from_message(request.text)
        )
        db.add(new_convo)
        db.commit()
        db.refresh(new_convo)
        conversation_id = new_convo.id

    # Get the bot's reply
    reply, conversation_id = chatbot_instance.get_similar_response(
        request.text, current_user, conversation_id, db
    )

    return {
        "reply": reply,
        "conversation_id": conversation_id
    }


@router.get("/conversation/{conversation_id}", response_model=List[ChatMessage])
def get_conversation(conversation_id: int, db: Session = Depends(get_session_local),
                     current_user: User = Depends(get_current_user)):
    messages = (
        db.query(MessageHistory)
        .filter(MessageHistory.conversation_id == conversation_id)
        .order_by(MessageHistory.timestamp.asc())
        .all()
    )

    result = [
        ChatMessage(
            sender="bot" if msg.is_bot else "user",
            text=msg.message,
            timestamp=msg.timestamp
        )
        for msg in messages
    ]

    return result


@router.get("/conversations", response_model=List[ChatConversation])
def list_conversations(db: Session = Depends(get_session_local),current_user: User = Depends(get_current_user)):
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc())
        .all()
    )
    return conversations


@router.get("/train")
def train(User = Depends(get_current_user)):
    trainer = ChatbotTrainer()
    trainer.initialize_from_json("data/data.json")
    initialize_knowledge_bases()