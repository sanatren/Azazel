"""
Centralized chain management to ensure single instances are shared across routes
"""
from Bot.rag_chain import RAGChain
from Bot.search_chain import SearchChain
from Bot.programming_assistant import ProgrammingAssistant

# Global chain storage
rag_chains = {}
search_chains = {}
programming_assistants = {}

def get_rag_chain(api_key: str, session_id: str, google_api_key: str = None, google_cse_id: str = None) -> RAGChain:
    """Get or create RAG chain instance - shared across all routes"""
    key = f"{session_id}_{api_key[:10]}"
    if key not in rag_chains:
        rag_chains[key] = RAGChain(api_key)
    return rag_chains[key]

def get_search_chain(api_key: str, google_api_key: str, google_cse_id: str) -> SearchChain:
    """Get or create search chain instance"""
    key = api_key[:10]
    if key not in search_chains:
        search_chains[key] = SearchChain(
            api_key=api_key,
            google_api_key=google_api_key,
            google_cse_id=google_cse_id
        )
    return search_chains[key]

def get_programming_assistant(api_key: str) -> ProgrammingAssistant:
    """Get or create programming assistant instance"""
    key = api_key[:10]
    if key not in programming_assistants:
        programming_assistants[key] = ProgrammingAssistant(api_key)
    return programming_assistants[key]
