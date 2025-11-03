"""
Chat API routes
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
import sys
import os
from typing import AsyncGenerator
import json

# Add parent directory to path to import Bot modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from api.models.schemas import (
    ChatRequest, ChatResponse, StreamChatRequest,
    CodeExecutionRequest, CodeExecutionResponse
)
from Bot.programming_assistant import ProgrammingAssistant
from langchain_openai import ChatOpenAI

router = APIRouter()

# Initialize assistant instances (will be created per request with user's API key)
def get_assistant(api_key: str, model: str = "gpt-4o-mini"):
    """Get a chat assistant instance with user's API key"""
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        temperature=0.7
    )

@router.post("/message", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    """
    Send a chat message and get a response
    """
    try:
        # Initialize assistant with user's API key
        assistant = get_assistant(request.api_key)

        # Create messages for the chat
        messages = []

        # Add system message if personality is provided
        if request.personality:
            messages.append({
                "role": "system",
                "content": request.personality + f"\n\nCRITICAL: You must respond in {request.language}."
            })

        # Add user message
        messages.append({
            "role": "user",
            "content": request.message
        })

        # Get response
        response = assistant.invoke(messages)

        return ChatResponse(
            response=response.content,
            session_id=request.session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def stream_message(request: StreamChatRequest):
    """
    Stream a chat response
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Initialize assistant with user's API key
            assistant = get_assistant(request.api_key)

            # Create messages
            messages = []

            # Add system message if personality is provided
            if request.personality:
                messages.append({
                    "role": "system",
                    "content": request.personality + f"\n\nCRITICAL: You must respond in {request.language}."
                })

            # Add chat history
            for msg in request.chat_history[-10:]:  # Last 10 messages
                messages.append({
                    "role": msg.role,
                    "content": msg.message
                })

            # Add current user message
            messages.append({
                "role": "user",
                "content": request.message
            })

            # Stream response
            for chunk in assistant.stream(messages):
                if chunk.content:
                    yield f"data: {json.dumps({'content': chunk.content})}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@router.post("/code", response_model=CodeExecutionResponse)
async def execute_code(request: CodeExecutionRequest):
    """
    Execute programming code
    """
    try:
        # Initialize programming assistant
        assistant = ProgrammingAssistant(request.api_key)

        # Convert chat history to expected format
        chat_history = [
            {"role": msg.role, "message": msg.message}
            for msg in request.chat_history
        ]

        # Execute code
        result = assistant.answer_programming_question(
            question=request.question,
            chat_history=chat_history,
            language=request.language,
            personality=request.personality or "You are a helpful programming assistant."
        )

        return CodeExecutionResponse(
            answer=result["answer"],
            code=result.get("code"),
            execution_result=result.get("execution_result"),
            code_executed=result["code_executed"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))