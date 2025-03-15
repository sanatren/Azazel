import os
import streamlit as st
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from supabase import create_client

# Constants
MAX_HISTORY_LENGTH = 35  # Maximum number of messages to keep in memory
MAX_TOKEN_LENGTH = 16000  # Maximum tokens for context window

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase = create_client(supabase_url, supabase_key)
model = None  # Initialize as None

session_data = {}

def get_model(api_key):
    """Get or create ChatOpenAI model with the given API key"""
    global model
    if model is None or model.openai_api_key != api_key:
        # Get the selected model from session state if available
        selected_model = "gpt-4"  # Default to GPT-4
        if hasattr(st, 'session_state') and 'selected_model' in st.session_state:
            model_map = {
                "GPT-3.5 Turbo": "gpt-3.5-turbo",
                "GPT-4": "gpt-4",
                "GPT-4 Turbo": "gpt-4-turbo-preview",
                "GPT-4o Mini": "gpt-4o-mini"
            }
            selected_model = model_map.get(st.session_state.selected_model, "gpt-4")
            
        model = ChatOpenAI(
            model=selected_model, 
            openai_api_key=api_key,
            streaming=True,  # Enable streaming
            temperature=0.7
        )
    return model

def get_session_history(session_id: str):
    """Retrieves chat history for a session, initializing if necessary."""
    if session_id not in session_data:
        session_data[session_id] = {
            "history": [],
            "language": "English",
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
    return session_data[session_id]["history"]

def trim_history(history, max_length=MAX_HISTORY_LENGTH):
    """Trims the chat history to prevent context window overflow."""
    if len(history) > max_length:
        return history[-max_length:]
    return history

def set_session_language(session_id: str, language: str = "English"):
    """Stores the preferred response language for a given session."""
    if session_id not in session_data:
        session_data[session_id] = {
            "history": [],
            "language": language,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat()
        }
    else:
        session_data[session_id]["language"] = language
        session_data[session_id]["last_accessed"] = datetime.now().isoformat()

def save_session_to_supabase(session_id, session_name, language="English"):
    """Saves or updates session metadata in Supabase."""
    try:
        existing = supabase.table("sessions").select("*").eq("session_id", session_id).execute()
        
        if existing.data:
            # Update existing session
            data = {
                "name": session_name,
                "language": language
                # Let Postgres handle last_accessed timestamp
            }
            response = supabase.table("sessions").update(data).eq("session_id", session_id).execute()
        else:
            # Create new session
            data = {
                "session_id": session_id,
                "name": session_name,
                "language": language
                # Let Postgres handle created_at and last_accessed timestamps
            }
            response = supabase.table("sessions").insert(data).execute()
        return response
    except Exception as e:
        print(f"Error saving session: {e}")
        return None

def save_message_to_supabase(session_id, role, message):
    """Stores chat messages in Supabase with enhanced error handling."""
    try:
        # First ensure session exists to satisfy foreign key constraint
        save_session_to_supabase(session_id, "Untitled Chat")
        
        data = {
            "session_id": session_id,
            "role": role,
            "message": message
            # No user_id field
        }
        response = supabase.table("history").insert(data).execute()
        return response
    except Exception as e:
        print(f"Supabase Error: {e}")
        return None

def get_chat_history_from_supabase(session_id):
    """Retrieves chat history from Supabase."""
    try:
        response = supabase.table("history").select("*").eq("session_id", session_id).order("timestamp").execute()
        return response.data
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def delete_session(session_id):
    """Deletes a session and its messages."""
    try:
        # Due to ON DELETE CASCADE, we only need to delete the session
        supabase.table("sessions").delete().eq("session_id", session_id).execute()
        if session_id in session_data:
            del session_data[session_id]
        
        # Reset the current_personality in session state when a session is deleted
        if 'current_personality' in st.session_state:
            st.session_state.current_personality = None
            
        return True
    except Exception as e:
        print(f"Error deleting session: {e}")
        return False

def get_all_sessions():
    """Retrieve all active sessions from the past week"""
    try:
        # Get sessions from the past week
        one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        response = supabase.table("sessions").select("*").gte("last_accessed", one_week_ago).order("last_accessed", desc=True).execute()
        return response.data
    except Exception as e:
        print(f"Error retrieving sessions: {e}")
        return []

def invoke_with_language(session_id, messages, language="English"):
    """
    Invoke the language model with the specified language preference
    
    Args:
        session_id: The session ID
        messages: The messages to send to the model
        language: The language to respond in
        
    Returns:
        str: The model's response
    """
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = st.session_state.get("openai_api_key")
            
        if not api_key:
            st.error("OpenAI API key not found")
            return "Error: API key not found"
        
        # Get the selected model from session state
        selected_model = st.session_state.get("selected_model", "GPT-4")
        model_name = {
             "GPT-3.5 Turbo": "gpt-3.5-turbo",
             "GPT-4": "gpt-4",
             "GPT-4 Turbo": "gpt-4-turbo",
             "GPT-4o Mini": "gpt-4o-mini" 
        }[selected_model]
        
        # Initialize the model with selected model
        model = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=0.7
        )
        
        # Get chat history for context
        history = get_chat_history_from_supabase(session_id)
        
        # Create a system message that instructs the model to respond in the specified language
        system_message = SystemMessage(content=f"You are a helpful assistant. CRITICAL INSTRUCTION: You MUST respond in {language}.")
        
        # Prepare messages with history and the new message
        formatted_messages = [system_message]
        
        # Add more recent chat history (increased from last 5 to last 20 messages)
        for msg in history[-10:]:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["message"]))
            else:
                formatted_messages.append(AIMessage(content=msg["message"]))
        
        # Add the new user message
        for msg in messages:
            formatted_messages.append(msg)
        
        # Generate the response
        response = model(formatted_messages)
        
        # Return the response content
        return response.content
    except Exception as e:
        st.error(f"Error invoking model: {str(e)}")
        return f"I'm having trouble generating a response right now. Error: {str(e)}"