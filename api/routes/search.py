"""
Web search API routes
"""
from fastapi import APIRouter, HTTPException
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from api.models.schemas import SearchRequest, SearchResponse
from Bot.search_chain import SearchChain
from api.core.config import settings

router = APIRouter()

# Store search chain instances
search_chains = {}

def get_search_chain(api_key: str) -> SearchChain:
    """Get or create search chain instance"""
    if api_key not in search_chains:
        search_chains[api_key] = SearchChain(
            api_key=api_key,
            google_api_key=settings.GOOGLE_API_KEY,
            google_cse_id=settings.GOOGLE_CSE_ID
        )
    return search_chains[api_key]

@router.post("/web", response_model=SearchResponse)
async def web_search(request: SearchRequest):
    """
    Perform web search and get answer
    """
    try:
        # Get search chain
        search_chain = get_search_chain(request.api_key)

        # Convert chat history
        chat_history = [
            {"role": msg.role, "message": msg.message}
            for msg in request.chat_history
        ]

        # Perform search
        result = search_chain.search_with_web(
            query=request.query,
            chat_history=chat_history,
            language=request.language
        )

        # Extract sources from search results
        sources = []
        if result.get("search_results"):
            google_results = result["search_results"].get("google_results", [])
            sources = [r.get("link") for r in google_results if r.get("link")]

        return SearchResponse(
            answer=result.get("answer", ""),
            search_results=result.get("search_results"),
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/determine-need")
async def determine_search_need(request: SearchRequest):
    """
    Determine if a query needs web search
    """
    try:
        # Get search chain
        search_chain = get_search_chain(request.api_key)

        # Convert chat history
        chat_history = [
            {"role": msg.role, "message": msg.message}
            for msg in request.chat_history
        ]

        # Determine if search is needed
        determination = search_chain.determine_search_need_with_llm(
            query=request.query,
            chat_history=chat_history,
            language=request.language
        )

        return determination

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))