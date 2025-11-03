"""
Session management API routes
"""
from fastapi import APIRouter, HTTPException
import uuid
from datetime import datetime

from api.models.schemas import (
    CreateSessionRequest, CreateSessionResponse,
    SessionHistoryRequest, SessionHistoryResponse,
    ChatMessage
)

router = APIRouter()

# In-memory session storage (in production, use database)
sessions = {}

@router.post("/create", response_model=CreateSessionResponse)
async def create_session(request: CreateSessionRequest):
    """
    Create a new chat session
    """
    try:
        session_id = str(uuid.uuid4())
        created_at = datetime.utcnow()

        sessions[session_id] = {
            "user_id": request.user_id,
            "created_at": created_at,
            "messages": []
        }

        return CreateSessionResponse(
            session_id=session_id,
            created_at=created_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str, limit: int = 50):
    """
    Get chat history for a session
    """
    try:
        if session_id not in sessions:
            return SessionHistoryResponse(messages=[], count=0)

        messages = sessions[session_id]["messages"][-limit:]

        return SessionHistoryResponse(
            messages=[ChatMessage(**msg) for msg in messages],
            count=len(messages)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/message/{session_id}")
async def add_message(session_id: str, message: ChatMessage):
    """
    Add a message to session history
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        message_dict = message.dict()
        if not message_dict.get("timestamp"):
            message_dict["timestamp"] = datetime.utcnow()

        sessions[session_id]["messages"].append(message_dict)

        return {"success": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/clear/{session_id}")
async def clear_session(session_id: str):
    """
    Clear session history
    """
    try:
        if session_id in sessions:
            sessions[session_id]["messages"] = []

        return {"success": True}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))