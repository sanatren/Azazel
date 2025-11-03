"""
Pydantic models for API request/response schemas
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Chat Models
class ChatMessage(BaseModel):
    role: str
    message: str
    timestamp: Optional[datetime] = None

class ChatRequest(BaseModel):
    message: str
    session_id: str
    api_key: str
    language: Optional[str] = "English"
    personality: Optional[str] = None
    use_rag: Optional[bool] = False
    force_search: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    session_id: str
    message_id: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None
    search_results: Optional[Dict[str, Any]] = None

class StreamChatRequest(BaseModel):
    message: str
    session_id: str
    api_key: str
    language: Optional[str] = "English"
    personality: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []
    use_rag: Optional[bool] = False

# Document Models
class DocumentUploadRequest(BaseModel):
    session_id: str
    api_key: str

class DocumentUploadResponse(BaseModel):
    success: bool
    message: str
    filename: Optional[str] = None

class DocumentQueryRequest(BaseModel):
    query: str
    session_id: str
    api_key: str
    k: Optional[int] = 8

class DocumentQueryResponse(BaseModel):
    results: List[Dict[str, Any]]
    count: int

class ClearDocumentsRequest(BaseModel):
    session_id: str

class ClearDocumentsResponse(BaseModel):
    success: bool
    message: str

# Search Models
class SearchRequest(BaseModel):
    query: str
    session_id: str
    api_key: str
    language: Optional[str] = "English"
    chat_history: Optional[List[ChatMessage]] = []

class SearchResponse(BaseModel):
    answer: str
    search_results: Optional[Dict[str, Any]] = None
    sources: Optional[List[str]] = None

# Session Models
class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None

class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: datetime

class SessionHistoryRequest(BaseModel):
    session_id: str
    limit: Optional[int] = 50

class SessionHistoryResponse(BaseModel):
    messages: List[ChatMessage]
    count: int

# Programming Assistant Models
class CodeExecutionRequest(BaseModel):
    question: str
    session_id: str
    api_key: str
    language: Optional[str] = "English"
    personality: Optional[str] = None
    chat_history: Optional[List[ChatMessage]] = []

class CodeExecutionResponse(BaseModel):
    answer: str
    code: Optional[str] = None
    execution_result: Optional[str] = None
    code_executed: bool

# Sentiment Analysis Models
class SentimentRequest(BaseModel):
    text: str
    session_id: str

class SentimentResponse(BaseModel):
    sentiment: str
    sentiment_score: float
    emotion: str
    emotion_score: float
    confidence: float

# Vision Models
class ImageAnalysisRequest(BaseModel):
    session_id: str
    query: str
    api_key: str

class ImageAnalysisResponse(BaseModel):
    analysis: List[Dict[str, Any]]
    count: int