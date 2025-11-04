"""
Chat API routes
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import sys
import os
from typing import AsyncGenerator, Optional
import json

# Add parent directory to path to import Bot modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from api.models.schemas import (
    ChatRequest, ChatResponse, StreamChatRequest,
    CodeExecutionRequest, CodeExecutionResponse
)
from Bot.programming_assistant import ProgrammingAssistant
from Bot.audio_handler import AudioHandler
from langchain_openai import ChatOpenAI
from api.core.config import settings
from api.core.chain_manager import get_rag_chain, get_search_chain, get_programming_assistant

router = APIRouter()

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
    Smart chat router that streams responses using appropriate chain
    Routes to: RAG → Programming → Search → Basic Chat
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            # Convert chat history to dict format
            chat_history = [
                {"role": msg.role, "message": msg.message}
                for msg in request.chat_history
            ]

            # 1. Check if document mode is forced OR (documents exist AND query is relevant)
            rag_chain = get_rag_chain(request.api_key, request.session_id)
            should_use_rag = False

            if request.force_document_mode:
                # Force RAG mode when checkbox is checked
                should_use_rag = True
            elif rag_chain.has_documents_for_session(request.session_id):
                # Auto-detect if query is relevant to documents
                should_use_rag = rag_chain.is_relevant_to_documents(request.message, request.session_id)

            if should_use_rag:
                result = rag_chain.answer_question(
                    question=request.message,
                    session_id=request.session_id,
                    chat_history=chat_history,
                    language=request.language
                )
                yield f"data: {json.dumps({'content': result['answer']})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 2. Check if it's a programming question
            prog_assistant = get_programming_assistant(request.api_key)
            if prog_assistant.is_programming_question(request.message):
                result = prog_assistant.answer_programming_question(
                    question=request.message,
                    chat_history=chat_history,
                    language=request.language,
                    personality=request.personality or "You are a helpful programming assistant."
                )
                yield f"data: {json.dumps({'content': result['answer']})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 3. Check if web search is forced OR auto-detect need for web search
            search_chain = get_search_chain(
                request.api_key,
                settings.GOOGLE_API_KEY,
                settings.GOOGLE_CSE_ID
            )

            needs_search = False
            if request.force_web_search:
                # Force web search when checkbox is checked
                needs_search = True
            else:
                # Auto-detect if query needs web search (current events, real-time info, etc.)
                search_determination = search_chain.determine_search_need_with_llm(
                    query=request.message,
                    chat_history=chat_history,
                    language=request.language
                )
                needs_search = search_determination.get("needs_search", False)

            if needs_search:
                result = search_chain.search_with_web(
                    query=request.message,
                    chat_history=chat_history,
                    language=request.language
                )
                yield f"data: {json.dumps({'content': result.get('answer', '')})}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 4. Fall back to basic chat
            assistant = get_assistant(request.api_key)
            messages = []

            if request.personality:
                messages.append({
                    "role": "system",
                    "content": request.personality + f"\n\nCRITICAL: You must respond in {request.language}."
                })

            for msg in request.chat_history[-10:]:
                messages.append({
                    "role": msg.role,
                    "content": msg.message
                })

            messages.append({
                "role": "user",
                "content": request.message
            })

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

@router.post("/transcribe")
async def transcribe_audio(
    audio: UploadFile = File(...),
    api_key: str = Form(...)
):
    """
    Transcribe audio to text using OpenAI Whisper
    """
    try:
        # Validate file type
        allowed_types = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/m4a", "audio/x-m4a"]
        if audio.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid audio file type. Supported: WAV, MP3, M4A")

        # Initialize audio handler
        handler = AudioHandler(api_key)

        # Read audio file bytes
        audio_bytes = await audio.read()

        # Get file extension
        file_extension = audio.filename.split('.')[-1].lower()

        # Transcribe
        transcript = handler.transcribe_audio_bytes(audio_bytes, file_extension)

        if transcript is None:
            raise HTTPException(status_code=500, detail="Failed to transcribe audio")

        return {"transcript": transcript}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))