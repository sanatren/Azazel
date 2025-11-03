"""
Document processing API routes
"""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
import sys
import os
from typing import List

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from api.models.schemas import (
    DocumentQueryRequest, DocumentQueryResponse,
    ClearDocumentsRequest, ClearDocumentsResponse,
    DocumentUploadResponse, ImageAnalysisRequest, ImageAnalysisResponse
)
from Bot.rag_chain import RAGChain

router = APIRouter()

# Store RAG chain instances per session (in production, use Redis or similar)
rag_chains = {}

def get_rag_chain(api_key: str, session_id: str) -> RAGChain:
    """Get or create RAG chain instance"""
    key = f"{session_id}_{api_key[:10]}"  # Use first 10 chars of API key as identifier
    if key not in rag_chains:
        rag_chains[key] = RAGChain(api_key)
    return rag_chains[key]

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    session_id: str = Form(...),
    api_key: str = Form(...)
):
    """
    Upload and process a document
    """
    try:
        # Get RAG chain
        rag_chain = get_rag_chain(api_key, session_id)

        # Process the file
        success = rag_chain.process_file(file, session_id)

        if success:
            return DocumentUploadResponse(
                success=True,
                message="Document uploaded and processed successfully",
                filename=file.filename
            )
        else:
            raise HTTPException(status_code=400, detail="Failed to process document")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=DocumentQueryResponse)
async def query_documents(request: DocumentQueryRequest):
    """
    Query processed documents
    """
    try:
        # Get RAG chain
        rag_chain = get_rag_chain(request.api_key, request.session_id)

        # Query documents
        results = rag_chain.document_processor.query_documents(
            query=request.query,
            session_id=request.session_id,
            k=request.k
        )

        return DocumentQueryResponse(
            results=results,
            count=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clear", response_model=ClearDocumentsResponse)
async def clear_documents(request: ClearDocumentsRequest):
    """
    Clear all documents for a session
    """
    try:
        # Clear from all RAG chain instances
        cleared = False
        for key in list(rag_chains.keys()):
            if key.startswith(request.session_id):
                rag_chain = rag_chains[key]
                rag_chain.clear_documents(request.session_id)
                cleared = True

        return ClearDocumentsResponse(
            success=cleared,
            message="Documents cleared successfully" if cleared else "No documents found"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-images", response_model=ImageAnalysisResponse)
async def analyze_images(request: ImageAnalysisRequest):
    """
    Analyze images for a session
    """
    try:
        # Get RAG chain
        rag_chain = get_rag_chain(request.api_key, request.session_id)

        # Analyze images
        analysis = rag_chain.document_processor.vision_processor.analyze_images(
            session_id=request.session_id,
            query=request.query
        )

        return ImageAnalysisResponse(
            analysis=analysis,
            count=len(analysis)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/has-documents/{session_id}")
async def has_documents(session_id: str, api_key: str):
    """
    Check if session has documents
    """
    try:
        # Check if any RAG chain exists for this session
        for key in rag_chains.keys():
            if key.startswith(session_id):
                rag_chain = rag_chains[key]
                has_docs = rag_chain.has_documents_for_session(session_id)
                return {"has_documents": has_docs}

        return {"has_documents": False}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))