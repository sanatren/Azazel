import sys
import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
import json
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
from openai import OpenAI
import docx
import tempfile
from PIL import Image
import base64

# Add this at the top of your app.py file (after imports)
def debug_basic_response(message):
    """Debug basic response generation"""
    st.info(f"Debug basic response for: {message}")
    
    try:
        # Try a direct call to OpenAI
        from openai import OpenAI
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Generate response
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message}
            ],
            temperature=0.7,
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        st.success("Basic response generated successfully")
        st.info(f"Response: {response_text}")
        return response_text
    except Exception as e:
        st.error(f"Error in basic response generation: {str(e)}")
        return None

# Update the direct_openai_response function to include chat history and sentiment analysis
def direct_openai_response(question, session_id):
    """Get direct response from OpenAI API"""
    if not st.session_state.openai_api_key:
        return "Please provide an OpenAI API key to continue."
        
    try:
        # Initialize the model
        current_personality = st.session_state.get(f"personality_{session_id}", "You are a helpful assistant.")
        client = OpenAI(api_key=st.session_state.openai_api_key)
        
        # Initialize sentiment analyzer if not already initialized
        if "sentiment_analyzer" not in st.session_state:
            from Bot.sentiment_analyzer import SentimentAnalyzer
            st.session_state.sentiment_analyzer = SentimentAnalyzer()
            
        # Analyze sentiment
        sentiment = st.session_state.sentiment_analyzer.analyze_sentiment(question)
        
        # Get chat history
        chat_history = get_chat_history_from_supabase(session_id)
        
        # Format chat history for the API
        formatted_messages = [
            {"role": "system", "content": current_personality}
        ]
        
        # Add recent chat history
        for msg in chat_history[-5:]:  # Last 5 messages
            formatted_messages.append({"role": msg["role"], "content": msg["message"]})
        
        # Add the current message
        formatted_messages.append({"role": "user", "content": question})
        
        # Get the selected model from session state
        model_name = available_models[st.session_state.get("selected_model", "GPT-4")]
        
        # Make API call with selected model
        response = client.chat.completions.create(
            model=model_name,
            messages=formatted_messages,
            temperature=0.7,
        )
        
        # Return the response text
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in direct OpenAI call: {str(e)}")
        return f"I'm having trouble generating a response right now. Error: {str(e)}"

# Adjust Python path to include the `Bot/` directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Bot")))
from search_chain import SearchChain
from programming_assistant import ProgrammingAssistant
from rag_chain import RAGChain
# Import chatbot functions
from audio_handler import AudioHandler
from chatbot_memory import (
    invoke_with_language,
    get_session_history,
    set_session_language,
    save_message_to_supabase
)
from supabase import create_client  # Import Supabase client

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize the search chain with the OpenAI API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
google_cse_id = os.getenv("GOOGLE_CSE_ID")

if google_api_key and google_cse_id and openai_api_key:
    search_chain = SearchChain(openai_api_key, google_api_key, google_cse_id)
else:
    search_chain = None
    if not google_api_key or not google_cse_id:
        st.sidebar.warning("Web search disabled. Set GOOGLE_API_KEY and GOOGLE_CSE_ID in .env file to enable.")
    if not openai_api_key:
        st.sidebar.warning("OpenAI API key not found. Set OPENAI_API_KEY in .env file.")

# Custom CSS
st.markdown(
    """
    <style>
    /* Azazel Chat Interface for Streamlit */
    /* Main color scheme variables */
    :root {
      --background: #121212;
      --sidebar-bg: #1E1E1E;
      --primary: #6C5CE7;
      --secondary: #A29BFE;
      --accent: #8A2BE2;
      --text: #E4E4E4;
      --muted-text: #A0A0A0;
      --success: #00B894;
      --error: #FF6B6B;
      --warning: #FDCB6E;
      --border: #333333;
      --message-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
      
      /* Shadow variables */
      --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.2);
      --shadow-md: 0 4px 6px rgba(0, 0, 0, 0.3);
      --shadow-lg: 0 10px 15px rgba(0, 0, 0, 0.4);
      
      /* Spacing variables */
      --space-xs: 0.25rem;
      --space-sm: 0.5rem;
      --space-md: 1rem;
      --space-lg: 1.5rem;
      --space-xl: 2rem;
    }

    /* Global styles */
    body {
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background-color: var(--background);
      color: var(--text);
      margin: 0;
      padding: 0;
      overflow-x: hidden;
    }

    /* Chat Container - Make it wider */
    .block-container {
      max-width: 1200px !important;
      padding-left: 5% !important;
      padding-right: 5% !important;
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6 {
      color: var(--text);
      font-weight: 600;
    }

    h1 {
      font-size: 24px;
    }

    h2 {
      font-size: 22px;
    }

    h3 {
      font-size: 20px;
    }

    h4, h5, h6 {
      font-size: 18px;
    }

    p, div, span {
      font-size: 15px;
      line-height: 1.6;
    }

    small {
      font-size: 12px;
      color: var(--muted-text);
    }

    /* Streamlit specific overrides */
    /* Main container */
    .stApp {
      background-color: var(--background);
    }

    /* Sidebar */
    .css-1d391kg, .css-12oz5g7 {
      background-color: var(--sidebar-bg) !important;
      border-right: 1px solid var(--border);
      width: 220px !important;
    }

    .sidebar .sidebar-content {
      background-color: var(--sidebar-bg);
      padding: var(--space-md);
    }

    /* Sidebar headers */
    .sidebar h2 {
      color: var(--text);
      border-bottom: 1px solid var(--border);
      padding-bottom: var(--space-sm);
      margin-bottom: var(--space-md);
    }

    /* Sidebar sections */
    .sidebar-section {
      margin-bottom: var(--space-lg);
    }

    /* Input fields */
    .stTextInput > div > div > input, 
    .stTextArea > div > div > textarea {
      background-color: var(--sidebar-bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
    }

    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 1px var(--primary);
    }

    /* Buttons */
    .stButton > button {
      background-color: var(--primary);
      color: white;
      border: none;
      border-radius: 8px;
      padding: var(--space-sm) var(--space-md);
      font-weight: 500;
      transition: all 0.2s ease;
    }

    .stButton > button:hover {
      background-color: var(--accent);
      box-shadow: var(--shadow-sm);
      transform: translateY(-1px);
    }

    /* Message input area */
    .chat-input-container {
      display: flex;
      position: sticky;
      bottom: 0;
      background-color: var(--background);
      padding: var(--space-md);
      border-top: 1px solid var(--border);
      z-index: 100;
      width: 100%;
    }

    .chat-input {
      flex-grow: 1;
      background-color: var(--sidebar-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: var(--space-sm);
      color: var(--text);
      resize: none;
      min-height: 60px;
    }

    .send-button {
      background-color: var(--primary);
      color: white;
      border: none;
      border-radius: 8px;
      margin-left: var(--space-sm);
      padding: 0 var(--space-md);
      cursor: pointer;
    }

    /* Chat message bubbles - Enhance width and highlighting */
    .message-container {
      display: flex;
      flex-direction: column;
      gap: var(--space-md);
      padding: var(--space-md);
      width: 100%;
    }

    .message {
      max-width: 95% !important;
      padding: var(--space-lg);
      border-radius: 8px;
      box-shadow: var(--message-shadow);
      margin-bottom: 16px;
      border: 1px solid var(--border);
    }

    .user-message {
      align-self: flex-end;
      background-color: var(--primary);
      border-top-right-radius: 2px;
    }

    .bot-message {
      align-self: flex-start;
      background-color: var(--sidebar-bg);
      border-top-left-radius: 2px;
    }

    .message-meta {
      font-size: 12px;
      color: var(--muted-text);
      margin-top: var(--space-xs);
    }

    /* Status indicators */
    .status-indicator {
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 50%;
      margin-right: var(--space-xs);
    }

    .status-online {
      background-color: var(--success);
    }

    .status-processing {
      background-color: var(--warning);
    }

    .status-error {
      background-color: var(--error);
    }

    /* Dividers */
    hr {
      border: 0;
      height: 1px;
      background-color: var(--border);
      margin: var(--space-md) 0;
    }

    /* Custom select/dropdown */
    .stSelectbox label {
      color: var(--text);
    }

    .stSelectbox > div > div {
      background-color: var(--sidebar-bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
      width: 8px;
      height: 8px;
    }

    ::-webkit-scrollbar-track {
      background: var(--background);
    }

    ::-webkit-scrollbar-thumb {
      background: var(--border);
      border-radius: 4px;
    }

    ::-webkit-scrollbar-thumb:hover {
      background: var(--primary);
    }

    /* Streamlit chat component custom styling - Enhanced for wider display */
    .stChatFloatingInputContainer {
      background-color: var(--background);
      border-top: 1px solid var(--border);
      padding: var(--space-sm);
      width: 100% !important;
      max-width: 1000px !important;
    }

    .stChatInputContainer {
      background-color: var(--sidebar-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      width: 100% !important;
    }

    /* Streamlit chat messages - make wider and more prominent */
    .stChatMessage {
      background-color: transparent;
      margin-bottom: 16px !important;
      padding: 0 !important;
      width: 95% !important;
      max-width: 95% !important;
    }

    .stChatMessage > div {
      border-radius: 8px !important;
      padding: var(--space-lg) !important;
      box-shadow: var(--message-shadow) !important;
      border: 1px solid var(--border) !important;
      transition: all 0.2s ease !important;
    }

    .stChatMessage > div:hover {
      box-shadow: var(--shadow-md) !important;
      transform: translateY(-1px);
    }

    .stChatMessage.user {
      margin-left: auto !important;
      margin-right: 0 !important;
    }

    .stChatMessage.assistant {
      margin-left: 0 !important;
      margin-right: auto !important;
    }

    .stChatMessage.user > div {
      background-color: var(--primary) !important;
      border-top-right-radius: 2px !important;
      color: white !important;
    }

    .stChatMessage.assistant > div {
      background-color: var(--sidebar-bg) !important;
      border-top-left-radius: 2px !important;
      color: var(--text) !important;
    }

    /* Add highlight effect to new messages */
    @keyframes messageHighlight {
      0% { transform: scale(1); opacity: 0.7; }
      50% { transform: scale(1.01); opacity: 1; }
      100% { transform: scale(1); opacity: 1; }
    }

    .stChatMessage:last-child > div {
      animation: messageHighlight 0.5s ease-out;
    }

    /* Main content area - Ensure it takes maximum width */
    .main .block-container {
      max-width: 1200px !important;
      padding-left: 2rem !important;
      padding-right: 2rem !important;
      margin: 0 auto !important;
    }

    /* Chat header styling */
    .stApp h1, .stApp h2 {
      text-align: center;
      margin-bottom: 20px;
    }

    /* Toast notifications */
    .toast {
      position: fixed;
      bottom: var(--space-lg);
      right: var(--space-lg);
      padding: var(--space-md);
      border-radius: 8px;
      box-shadow: var(--shadow-md);
      z-index: 1000;
      animation: slideIn 0.3s ease-out;
    }

    .toast-success {
      background-color: var(--success);
      color: white;
    }

    .toast-error {
      background-color: var(--error);
      color: white;
    }

    .toast-warning {
      background-color: var(--warning);
      color: var(--background);
    }

    @keyframes slideIn {
      from {
        transform: translateY(100%);
        opacity: 0;
      }
      to {
        transform: translateY(0);
        opacity: 1;
      }
    }

    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
      gap: var(--space-sm);
      background-color: var(--background);
      border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
      background-color: var(--sidebar-bg);
      color: var(--text);
      border-radius: 8px 8px 0 0;
      padding: var(--space-sm) var(--space-md);
    }

    .stTabs [aria-selected="true"] {
      background-color: var(--primary);
      color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add global auto-scrolling JavaScript right after the new CSS
st.markdown("""
<script>
    // Super aggressive scroll function that works in different contexts
    function superForceScroll() {
        // Standard approaches
        window.scrollTo(0, document.body.scrollHeight);
        window.scrollTo(0, 999999);
        
        // Try scrolling specific Streamlit elements
        document.querySelectorAll('.main, .stApp, .element-container, .block-container, [data-testid="stVerticalBlock"]').forEach(el => {
            if (el) {
                el.scrollTop = el.scrollHeight;
            }
        });
        
        // Find any containers with overflow
        document.querySelectorAll('div').forEach(div => {
            const style = window.getComputedStyle(div);
            if (style.overflow === 'auto' || style.overflow === 'scroll' || style.overflowY === 'auto' || style.overflowY === 'scroll') {
                div.scrollTop = div.scrollHeight;
            }
        });
        
        // Try to scroll to our specific scroll target if it exists
        const scrollTarget = document.getElementById('scroll-target');
        if (scrollTarget) {
            scrollTarget.scrollIntoView({behavior: 'auto', block: 'end'});
        }
    }
    
    // Create a MutationObserver to watch for content changes
    function setupScrollObserver() {
        const observer = new MutationObserver((mutations) => {
            superForceScroll();
        });
        
        // Observer configuration: watch for any changes to the DOM
        const config = { 
            childList: true, 
            subtree: true,
            characterData: true,
            attributes: true
        };
        
        // Target main Streamlit containers
        ['main', 'stApp', 'element-container', 'block-container'].forEach(className => {
            const elements = document.getElementsByClassName(className);
            if (elements.length > 0) {
                observer.observe(elements[0], config);
            }
        });
        
        // Also observe the body as fallback
        observer.observe(document.body, config);
        
        return observer;
    }
    
    // Handle session changes by looking for Streamlit-specific events
    function setupStreamlitEvents() {
        // Track hash changes which often happen with Streamlit navigation
        window.addEventListener('hashchange', () => {
            console.log('Hash changed, scrolling...');
            scrollSequence();
        });
        
        // Intercept History API for Streamlit navigation
        const originalPushState = history.pushState;
        history.pushState = function() {
            originalPushState.apply(this, arguments);
            console.log('History changed, scrolling...');
            scrollSequence();
        };
        
        const originalReplaceState = history.replaceState;
        history.replaceState = function() {
            originalReplaceState.apply(this, arguments);
            console.log('History replaced, scrolling...');
            scrollSequence();
        };
        
        // Watch for API key input
        document.addEventListener('input', (e) => {
            if (e.target && (e.target.type === 'password' || e.target.placeholder === 'sk-...')) {
                console.log('API key input detected');
                setTimeout(() => scrollSequence(), 500);
            }
        });
        
        // Watch for form submissions
        document.addEventListener('submit', (e) => {
            console.log('Form submitted');
            scrollSequence();
        });
        
        // Watch for click events that might trigger navigation
        document.addEventListener('click', (e) => {
            if (e.target && (e.target.tagName === 'BUTTON' || e.target.tagName === 'A' || 
                e.target.closest('button') || e.target.closest('a'))) {
                console.log('Navigation element clicked');
                setTimeout(() => scrollSequence(), 500);
            }
        });
    }
    
    // Perform a sequence of scrolls with delays
    function scrollSequence() {
        superForceScroll();
        
        // Multiple attempts with increasing delays
        [10, 100, 300, 500, 1000, 2000, 3000].forEach(delay => {
            setTimeout(superForceScroll, delay);
        });
    }
    
    // Initialize everything on page load
    function initializeScrolling() {
        console.log('Initializing scrolling behavior');
        scrollSequence();
        setupScrollObserver();
        setupStreamlitEvents();
        
        // Periodic check as a failsafe
        setInterval(superForceScroll, 2000);
    }
    
    // Make sure we initialize as soon as possible
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeScrolling);
    } else {
        initializeScrolling();
    }
    
    // Also run on load event
    window.addEventListener('load', scrollSequence);
</script>
""", unsafe_allow_html=True)

# Function to save message to Supabase
def save_message_to_supabase(session_id, role, message):
    """Stores chat messages in Supabase with enhanced error handling."""
    try:
        # First ensure session exists to satisfy foreign key constraint
        save_session_to_supabase(session_id, "Untitled Chat")
        
        data = {
            "session_id": session_id,
            "role": role,
            "message": message
        }
        response = supabase.table("history").insert(data).execute()
        return response
    except Exception as e:
        st.error(f"Error saving message: {str(e)}")
        return None

# Move the display_chat_messages function to the top with other functions
def display_chat_messages(chat_history):
    """Displays chat messages with proper formatting and error handling."""
    try:
        if not chat_history:
            st.info("No messages yet. Start a conversation!")
            return
            
        if "chat_message" in dir(st):
            for msg in chat_history:
                if msg["role"] == "assistant":
                    with st.chat_message("assistant"):
                        st.markdown(msg["message"])
                else:
                    with st.chat_message("user"):
                        st.markdown(msg["message"])
        else:
            st.write("### Chat History")
            for msg in chat_history:
                timestamp = msg.get("timestamp", "").split(".")[0].replace("T", " ") if msg.get("timestamp") else ""
                if msg["role"] == "assistant":
                    st.markdown(f"<div class='assistant-msg'><small>{timestamp}</small><br>{msg['message']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='user-msg'><small>{timestamp}</small><br>{msg['message']}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying messages: {e}")

# Move this function definition to the top with other functions
def save_session_to_supabase(session_id, session_name, language="English", personality=None):
    """Save session metadata to Supabase sessions table"""
    try:
        # Check if session exists
        existing = supabase.table("sessions").select("*").eq("session_id", session_id).execute()
        
        # Prepare data
        data = {
            "name": session_name,
            "language": language
            # Let Postgres handle last_accessed timestamp
        }
        
        # Add personality if provided
        if personality:
            data["personality"] = personality
        
        if existing.data:
            # Update existing session
            response = supabase.table("sessions").update(data).eq("session_id", session_id).execute()
        else:
            # Create new session
            data["session_id"] = session_id
            response = supabase.table("sessions").insert(data).execute()
        
        return response
    except Exception as e:
        st.error(f"Error saving session: {e}")
        return None

# Functions for session management with Supabase
def get_all_sessions():
    """Retrieve all active sessions from the past week"""
    try:
        # Get sessions from the past week
        one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        response = supabase.table("sessions").select("*").gte("last_accessed", one_week_ago).order("last_accessed", desc=True).execute()
        return response.data
    except Exception as e:
        st.error(f"Error retrieving sessions: {e}")
        return []

def get_chat_history_from_supabase(session_id):
    """Retrieve chat history for a specific session"""
    try:
        response = supabase.table("history").select("*").eq("session_id", session_id).order("timestamp", desc=False).execute()
        
        # Transform to the format expected by the UI
        chat_history = []
        for msg in response.data:
            chat_history.append({
                "role": msg["role"],
                "message": msg["message"]
            })
        
        return chat_history
    except Exception as e:
        st.error(f"Error retrieving chat history: {e}")
        return []

def delete_session(session_id):
    """Delete a session and its associated messages"""
    try:
        # Due to ON DELETE CASCADE, we only need to delete the session
        supabase.table("sessions").delete().eq("session_id", session_id).execute()
        return True
    except Exception as e:
        st.error(f"Error deleting session: {e}")
        return False

# Initialize Streamlit session state
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None

# Map predefined personalities to actual instructions
personality_map = {
    "Default (Helpful Assistant)": "You are a helpful, respectful, and honest assistant.",
    "Professional & Formal": "You are a professional assistant who communicates with formal language, precise terminology, and maintains a respectful, business-like tone. You prioritize accuracy and clarity in your responses.",
    "Friendly & Casual": "You are a friendly and casual assistant who uses conversational language, occasional slang, and a warm, approachable tone. You're like chatting with a helpful friend.",
    "Creative & Imaginative": "You are a creative assistant with a vivid imagination. You use descriptive language, analogies, and metaphors to explain concepts, and you're not afraid to think outside the box.",
    "Concise & Direct": "You are a concise and direct assistant who values brevity. You get straight to the point with short sentences, avoid unnecessary details, and prioritize actionable information.",
    "Empathetic & Supportive": "You are an empathetic assistant who shows understanding and emotional intelligence. You acknowledge feelings, use supportive language, and prioritize the user's wellbeing in your responses.",
    "Humorous & Playful": "You are a humorous assistant with a playful tone. You use appropriate jokes, wordplay, and a light-hearted approach while still being helpful and respectful.",
    "Educational & Informative": "You are an educational assistant who focuses on teaching and explaining. You provide context, examples, and background information while maintaining an accessible and informative tone."
}

if "current_session" not in st.session_state:
    existing_sessions = get_all_sessions()
    
    if existing_sessions:
        # Use the most recent session
        st.session_state.current_session = existing_sessions[0]["session_id"]
        st.session_state.current_session_name = existing_sessions[0]["name"]
        st.session_state.current_language = existing_sessions[0]["language"]
        # Set personality if available
        if "personality" in existing_sessions[0] and existing_sessions[0]["personality"]:
            st.session_state.current_personality = existing_sessions[0]["personality"]
            # Find matching predefined personality or set to custom
            found = False
            for option, text in personality_map.items():
                if text == st.session_state.current_personality.split("\n\n")[0]:  # Only compare the first part before the directive
                    st.session_state.personality_option = option
                    found = True
                    break
            if not found:
                st.session_state.personality_option = "Custom..."
                st.session_state.custom_personality_text = st.session_state.current_personality.split("\n\n")[0]  # Strip directives
        else:
            st.session_state.current_personality = "You are a helpful, respectful, and honest assistant."
            st.session_state.personality_option = "Default (Helpful Assistant)"
    else:
        # Create a new session
        new_session_id = str(uuid.uuid4())
        st.session_state.current_session = new_session_id
        st.session_state.current_session_name = "New Chat"
        st.session_state.current_language = "English"
        st.session_state.current_personality = "You are a helpful, respectful, and honest assistant."
        st.session_state[f"personality_{new_session_id}"] = "You are a helpful, respectful, and honest assistant."
        st.session_state.personality_option = "Default (Helpful Assistant)"
        save_session_to_supabase(new_session_id, "New Chat")

# Sidebar: Manage Sessions and Preferred Language
st.sidebar.title("🔑 API Key")

# Add API key input to sidebar
api_key_input = st.sidebar.text_input(
    "Enter your OpenAI API key",
    type="password",
    placeholder="sk-...",
    help="Get your API key from https://platform.openai.com/account/api-keys",
    value=st.session_state.openai_api_key if st.session_state.openai_api_key else ""
)

# Add this after the API key input section in the sidebar, before the Chat Sessions title
st.sidebar.title("🤖 Model Settings")

# Model selection dropdown
available_models = {
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
    "GPT-4": "gpt-4",
    "GPT-4 Turbo": "gpt-4-turbo",
    "GPT-4o Mini": "gpt-4o-mini"  # Replacing Vision with Mini
}

# Initialize selected_model in session state if not present
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "GPT-4"

# Model selection with tooltip
selected_model = st.sidebar.selectbox(
    "Choose AI Model",
    options=list(available_models.keys()),
    index=list(available_models.keys()).index(st.session_state.selected_model),
    help="Select the AI model to use for chat. Note: GPT-4 and GPT-4 Turbo require appropriate API access."
)

# Update session state
st.session_state.selected_model = selected_model

# Display model information
model_info = {
    "GPT-3.5 Turbo": "Fast and cost-effective for most tasks",
    "GPT-4": "More capable for complex tasks",
    "GPT-4 Turbo": "Latest model with current knowledge",
    "GPT-4o Mini": "Lightweight version of GPT-40 for faster responses"  # Updated info
}

st.sidebar.info(f"📝 {model_info[selected_model]}")


if api_key_input:
    if not api_key_input.startswith('sk-'):
        st.sidebar.error("Invalid API key format. Key should start with 'sk-'")
        st.stop()
        
    try:
        # Store API key in session state
        st.session_state.openai_api_key = api_key_input
        
        # Get the selected model
        model_name = available_models[st.session_state.selected_model]
        
        # Initialize audio handler with the API key
        audio_handler = AudioHandler(api_key_input)
        
        # Initialize the model with the new API key and selected model
        model = ChatOpenAI(model=model_name, openai_api_key=api_key_input)
        
        # Initialize the RAG chain after initializing the audio handler
        rag_chain = RAGChain(api_key_input)
        
        # Initialize the programming assistant after initializing the RAG chain
        programming_assistant = ProgrammingAssistant(api_key_input)
        
        st.sidebar.success("API key successfully validated!")
    except Exception as e:
        st.sidebar.error(f"Error initializing with API key: {str(e)}")
        st.stop()
else:
    st.sidebar.error("Please enter your OpenAI API key to start chatting!")
    st.stop()

# Update the sidebar section
st.sidebar.title("💬 Chat Sessions")

# New Chat button
if st.sidebar.button("➕ New Chat"):
    if st.session_state.openai_api_key:  # Only allow new chats if API key is set
        # Clear any previous personality state to avoid persistence issues
        if "personality_option" in st.session_state:
            del st.session_state.personality_option
        if "current_personality" in st.session_state:
            del st.session_state.current_personality
        if "custom_personality_text" in st.session_state:
            del st.session_state.custom_personality_text
            
        # Create new session
        new_session_id = str(uuid.uuid4())
        st.session_state.current_session = new_session_id
        st.session_state.current_session_name = "New Chat"
        st.session_state.current_language = "English"
        
        # Set default personality for new session
        st.session_state.current_personality = "You are a helpful, respectful, and honest assistant."
        st.session_state[f"personality_{new_session_id}"] = "You are a helpful, respectful, and honest assistant."
        st.session_state.personality_option = "Default (Helpful Assistant)"
        
        # Save the new session with default personality
        save_session_to_supabase(
            new_session_id,
            "New Chat",
            "English",
            st.session_state.current_personality
        )
        st.rerun()
    else:
        st.sidebar.error("Please enter your API key first!")

# List existing sessions with a radio button to switch between them
existing_sessions = get_all_sessions()
session_options = {session["session_id"]: session.get("name", f"Chat {i+1}")
                  for i, session in enumerate(existing_sessions)}

# Create a mapping of display names to session objects for deletion lookup
session_display_map = {}
for session in existing_sessions:
    display_name = f"{session.get('name', 'Untitled')} ({session['session_id'][:8]}...)"
    session_display_map[display_name] = session

if session_options:
    # Format session names for display
    formatted_options = [f"{name} ({session_id[:8]}...)" for session_id, name in session_options.items()]
    
    # Display sessions as radio buttons
    selected_index = list(session_options.keys()).index(st.session_state.current_session) if st.session_state.current_session in session_options else 0
    selected_option = st.sidebar.radio("Select Session", formatted_options, index=selected_index)
    
    # Extract the session ID from the selected option
    selected_session_id = list(session_options.keys())[formatted_options.index(selected_option)]
    
    # Switch to the selected session if different from the current one
    if selected_session_id != st.session_state.current_session:
        # Clear any previous personality state to avoid persistence issues
        if "personality_option" in st.session_state:
            del st.session_state.personality_option
        if "current_personality" in st.session_state:
            del st.session_state.current_personality
        if "custom_personality_text" in st.session_state:
            del st.session_state.custom_personality_text
        
        # Set the new session
        st.session_state.current_session = selected_session_id
        selected_session = next((s for s in existing_sessions if s["session_id"] == selected_session_id), None)
        
        if selected_session:
            st.session_state.current_session_name = selected_session.get("name", "Untitled Chat")
            st.session_state.current_language = selected_session.get("language", "English")
            
            # Load the personality for this specific session
            if "personality" in selected_session and selected_session["personality"]:
                st.session_state.current_personality = selected_session["personality"]
            
            # Find matching predefined personality or set to custom
            found = False
            for option, text in personality_map.items():
                base_personality = selected_session["personality"].split("\n\n")[0]  # Only compare the first part
                if text == base_personality:
                    st.session_state.personality_option = option
                    found = True
                    break
            
            if not found:
                st.session_state.personality_option = "Custom..."
                # Store just the base personality without directives for editing
                st.session_state.custom_personality_text = base_personality
            else:
                # Add importance directive to predefined personalities too
                current_personality = personality_map[st.session_state.personality_option] + "\n\nYou MUST maintain this personality in ALL your responses, regardless of the subject matter. This personality dictates HOW you respond, not WHAT information you provide."
            
            # Store the current personality in session state (only for this specific session)
            st.session_state.current_personality = current_personality
            st.session_state[f"personality_{st.session_state.current_session}"] = current_personality

            # Update personality in database when changed
            if st.session_state.get(f"previous_personality_{st.session_state.current_session}") != current_personality:
                save_session_to_supabase(
                    st.session_state.current_session,
                    st.session_state.current_session_name,
                    st.session_state.current_language,
                    current_personality
                )
                # Store the previous personality with session ID in the key to make it session-specific
                st.session_state[f"previous_personality_{st.session_state.current_session}"] = current_personality
        
        st.rerun()

# Session management section
st.sidebar.subheader("Session Management")

# Session name edit field
new_session_name = st.sidebar.text_input("Chat Name", value=st.session_state.current_session_name)
if new_session_name != st.session_state.current_session_name:
    st.session_state.current_session_name = new_session_name
    save_session_to_supabase(st.session_state.current_session, new_session_name, st.session_state.current_language)

# Delete Session Options
if len(existing_sessions) > 1:  # Only show if there's more than one session
    st.sidebar.subheader("Delete Sessions")
    
    # Multi-select for sessions to delete
    sessions_to_delete = st.sidebar.multiselect(
        "Select sessions to delete:",
        options=[f"{session.get('name', 'Untitled')} ({session['session_id'][:8]}...)" for session in existing_sessions],
        help="Select one or more sessions to delete"
    )
    
    # Always show the confirmation checkbox
    confirm_delete = st.sidebar.checkbox("Confirm deletion? This action cannot be undone.")
    
    # Delete button
    if sessions_to_delete and st.sidebar.button("🗑️ Delete Selected Sessions"):
        if not confirm_delete:
            st.sidebar.error("Please confirm deletion by checking the box above.")
        else:
            deleted_any = False
            need_rerun = False
            deleted_current_session = False
            
            for session_display in sessions_to_delete:
                # Look up the full session object using our map
                if session_display in session_display_map:
                    session = session_display_map[session_display]
                    session_id = session["session_id"]
                    
                    # If deleting current session, flag for rerun and session change
                    if session_id == st.session_state.current_session:
                        deleted_current_session = True
                        need_rerun = True
                    
                    # Delete the session
                    if delete_session(session_id):
                        st.sidebar.success(f"Deleted: {session.get('name', 'Untitled')}")
                        deleted_any = True
            
            if deleted_any:
                # Only get remaining sessions if we deleted the current session or need to rerun
                if deleted_current_session:
                    # Get remaining sessions to switch to
                    remaining_sessions = get_all_sessions()
                    if remaining_sessions:
                        st.session_state.current_session = remaining_sessions[0]["session_id"]
                        st.session_state.current_session_name = remaining_sessions[0].get("name", "Untitled Chat")
                        st.session_state.current_language = remaining_sessions[0].get("language", "English")
                        
                        # Reset personality to the one for the new session or to default
                        if "personality" in remaining_sessions[0] and remaining_sessions[0]["personality"]:
                            st.session_state.current_personality = remaining_sessions[0]["personality"]
                            st.session_state[f"personality_{remaining_sessions[0]['session_id']}"] = remaining_sessions[0]["personality"]
                            
                            # Find matching predefined personality or set to custom
                            found = False
                            for option, text in personality_map.items():
                                if text == remaining_sessions[0]["personality"].split("\n\n")[0]:  # Only compare the first part
                                    st.session_state.personality_option = option
                                    found = True
                                    break
                            
                            if not found:
                                st.session_state.personality_option = "Custom..."
                                st.session_state.custom_personality_text = remaining_sessions[0]["personality"].split("\n\n")[0]
                        else:
                            # Set to default personality if none is set for the new session
                            st.session_state.current_personality = "You are a helpful, respectful, and honest assistant."
                            st.session_state[f"personality_{remaining_sessions[0]['session_id']}"] = "You are a helpful, respectful, and honest assistant."
                            st.session_state.personality_option = "Default (Helpful Assistant)"
                
                if need_rerun:
                    st.rerun()

# Delete Current Chat button (ensures at least one session remains)
if st.sidebar.button("🗑️ Delete Current Chat"):
    if len(existing_sessions) > 1:
        current_session_id = st.session_state.current_session
        if delete_session(current_session_id):
            # Set current session to the next available session
            remaining_sessions = [s for s in existing_sessions if s["session_id"] != current_session_id]
            if remaining_sessions:
                st.session_state.current_session = remaining_sessions[0]["session_id"]
                st.session_state.current_session_name = remaining_sessions[0].get("name", "Untitled Chat")
                st.session_state.current_language = remaining_sessions[0].get("language", "English")
                
                # Reset personality to the one for the new session or to default
                if "personality" in remaining_sessions[0] and remaining_sessions[0]["personality"]:
                    st.session_state.current_personality = remaining_sessions[0]["personality"]
                    st.session_state[f"personality_{remaining_sessions[0]['session_id']}"] = remaining_sessions[0]["personality"]
                    
                    # Find matching predefined personality or set to custom
                    found = False
                    for option, text in personality_map.items():
                        if text == remaining_sessions[0]["personality"].split("\n\n")[0]:  # Only compare the first part
                            st.session_state.personality_option = option
                            found = True
                            break
                    
                    if not found:
                        st.session_state.personality_option = "Custom..."
                        st.session_state.custom_personality_text = remaining_sessions[0]["personality"].split("\n\n")[0]
                else:
                    # Set to default personality if none is set for the new session
                    st.session_state.current_personality = "You are a helpful, respectful, and honest assistant."
                    st.session_state[f"personality_{remaining_sessions[0]['session_id']}"] = "You are a helpful, respectful, and honest assistant."
                    st.session_state.personality_option = "Default (Helpful Assistant)"
                st.rerun()
    else:
        st.sidebar.error("Cannot delete the only remaining session. Create a new session first.")

# Preferred language input
st.sidebar.subheader("🌍 Language Settings")

# Add a hint about available languages
st.sidebar.caption("Type any language (e.g., Hindi, Spanish, French, etc.)")

language = st.sidebar.text_input(
    "Enter Response Language",
    value=st.session_state.current_language,
    placeholder="Enter any language...",
    help="The bot will respond in this language regardless of the language you use to ask questions."
)

# Add language validation and feedback
if language and language != st.session_state.current_language:
    # Convert first letter to uppercase for consistency
    language = language.strip().title()
    st.session_state.current_language = language
    
    # Update the language in the database
    save_session_to_supabase(
        st.session_state.current_session,
        st.session_state.current_session_name,
        language
    )
    
    st.sidebar.success(f"Now responding in {language}")
    st.rerun()  # Force a rerun to apply the language change

# Add a new tab in the sidebar for document upload
with st.sidebar.expander("📁 Document & Image Upload", expanded=False):
    uploaded_files = st.file_uploader(
        "Upload documents or images",
        type=["pdf", "docx", "xlsx", "pptx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="multi_file_upload"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            with st.spinner(f"Processing {uploaded_file.name}..."):
                if rag_chain.document_processor.process_file(uploaded_file, st.session_state.current_session):
                    st.success(f"Processed {uploaded_file.name}")
                else:
                    st.error(f"Failed to process {uploaded_file.name}")
    
    # Clear All Data button removed as requested

# Add these options to your sidebar
st.sidebar.markdown("### Advanced Options")
st.sidebar.markdown("*Configure how the assistant processes your queries:*")

# Force RAG Mode
if st.sidebar.checkbox("Force RAG Mode", value=st.session_state.get("force_rag", False),
                      help="Always use document knowledge to answer questions"):
    st.session_state.force_rag = True
else:
    st.session_state.force_rag = False

# Force Web Search
if st.sidebar.checkbox("Force Web Search", value=st.session_state.get("force_search", False),
                      help="Always search the web for the latest information"):
    st.session_state.force_search = True
else:
    st.session_state.force_search = False

# Disable Code Execution
if st.sidebar.checkbox("Disable Code Execution", value=st.session_state.get("disable_code_execution", True),
                      help="Only provide code examples without executing them"):
    st.session_state.disable_code_execution = True
else:
    st.session_state.disable_code_execution = False

# Add this after preferred language input in the sidebar
st.sidebar.subheader("🎭 Personality Settings")

# Info about personalities
st.sidebar.info("Choose how you want the AI to respond to you. Select a pre-defined personality or create your own.")

# Personality selection options
personality_options = [
    "Default (Helpful Assistant)",
    "Professional & Formal",
    "Friendly & Casual",
    "Creative & Imaginative",
    "Concise & Direct",
    "Empathetic & Supportive",
    "Humorous & Playful",
    "Educational & Informative",
    "Custom..."
]

# Add personality selection with a default value
selected_personality = st.sidebar.selectbox(
    "Choose AI Personality",
    options=personality_options,
    index=personality_options.index(st.session_state.get("personality_option", "Default (Helpful Assistant)")),
    help="Select how you want the AI to respond to your messages"
)

# Store the selected option for persistence
st.session_state.personality_option = selected_personality

# Custom personality input field (shown only when "Custom..." is selected)
custom_personality = ""
if selected_personality == "Custom...":
    st.sidebar.markdown("#### How to Write an Effective Personality")
    st.sidebar.info("""
    For best results, be specific and detailed:
    1. Start with "You are a..." followed by character traits
    2. Include speaking style and tone details
    3. Explain HOW the AI should respond to information
    4. Be explicit about attitude and approach
    """)
    
    custom_personality = st.sidebar.text_area(
        "Describe the AI's personality and tone",
        value=st.session_state.get("custom_personality_text", ""),
        placeholder="Example: You are a witty assistant with deep knowledge of science. You use analogies and occasionally quote famous scientists.",
        help="Be specific about tone, style, and character traits you want the AI to exhibit"
    )
    # Store the custom text for persistence
    st.session_state.custom_personality_text = custom_personality

# Set the current personality in session state based on selection
if selected_personality == "Custom...":
    # Format custom personality for better results
    if custom_personality:
        # Ensure it starts with "You are a" if not already
        if not custom_personality.lower().startswith("you are"):
            custom_personality = "You are " + custom_personality
            
        # Add importance directive to custom personalities
        custom_personality = custom_personality.strip() + "\n\nYou MUST maintain this personality in ALL your responses, regardless of the subject matter. This personality dictates HOW you respond, not WHAT information you provide."
        current_personality = custom_personality
    else:
        current_personality = "You are a helpful assistant."
else:
    # Add importance directive to predefined personalities too
    current_personality = personality_map[selected_personality] + "\n\nYou MUST maintain this personality in ALL your responses, regardless of the subject matter. This personality dictates HOW you respond, not WHAT information you provide."

# Store the current personality in session state (only for this specific session)
st.session_state.current_personality = current_personality
st.session_state[f"personality_{st.session_state.current_session}"] = current_personality

# Update personality in database when changed
if st.session_state.get(f"previous_personality_{st.session_state.current_session}") != current_personality:
    save_session_to_supabase(
        st.session_state.current_session,
        st.session_state.current_session_name,
        st.session_state.current_language,
        current_personality
    )
    # Store the previous personality with session ID in the key to make it session-specific
    st.session_state[f"previous_personality_{st.session_state.current_session}"] = current_personality

# Main Chat Interface
# Function to load and display the logo
def display_logo():
    # Use the absolute path to the workspace
    workspace_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logo_path = os.path.join(workspace_path, "Streamlit", "static", "images", "azaz.png")
    
    # Check if logo exists
    if os.path.exists(logo_path):
        # Display logo with custom HTML for better styling
        with open(logo_path, "rb") as f:
            logo_bytes = f.read()
            encoded_logo = base64.b64encode(logo_bytes).decode()
        
        # Create HTML with logo and title
        logo_html = f"""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <img src="data:image/png;base64,{encoded_logo}" alt="Azazel Logo" style="height: 60px; margin-right: 10px;">
            <h1 style="margin: 0; color: white;">Azazel</h1>
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
    else:
        # Fallback to text title if image not found
        st.title("🤖 Azazel")
        st.warning(f"Logo file not found at {logo_path}. Please add your logo file named 'azaz.png' to the Streamlit/static/images directory.")

# Display the logo and title
display_logo()

if not st.session_state.openai_api_key:
    st.error("Please enter your OpenAI API key in the sidebar to start chatting!")
else:
    st.subheader(st.session_state.current_session_name)
    
    # Get chat history
    chat_history = get_chat_history_from_supabase(st.session_state.current_session)
    
    # Create a container for chat messages
    chat_container = st.container()
    
    # Add JavaScript for auto-scrolling
    st.markdown("""
        <script>
            // Scroll to bottom when page loads
            window.onload = function() {
                window.scrollTo(0, document.body.scrollHeight);
            };
            
            // Scroll to bottom when new messages are added
            const observer = new MutationObserver(function() {
                window.scrollTo(0, document.body.scrollHeight);
            });
            observer.observe(document.body, { childList: true, subtree: true });
        </script>
    """, unsafe_allow_html=True)
    
    # Display chat history
    with chat_container:
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["message"])
        
        # Add a div at the bottom for auto-scrolling
        st.markdown('<div id="scroll-to-bottom"></div>', unsafe_allow_html=True)
    
    # Check if we should use RAG (if documents have been uploaded)
    use_rag = False
    try:
        # First check if the rag_chain has the needed attributes
        if hasattr(rag_chain, "document_processor") and hasattr(rag_chain.document_processor, "vectorstores"):
            use_rag = st.session_state.current_session in rag_chain.document_processor.vectorstores
        # Fallback method - use the has_documents_for_session method which should be more reliable
        else:
            use_rag = rag_chain.has_documents_for_session(st.session_state.current_session)
    except Exception as e:
        st.warning(f"Error checking document status: {e}")
        use_rag = False
    
    # Add this after the chat history display
    if use_rag:
        st.info("📄 Document mode active: I'll use your uploaded documents to answer questions, but I can still chat about other topics too.")
    
    # Input area container
    input_container = st.container()
    with input_container:
        # Create tabs for text and voice input
        input_tab, voice_tab = st.tabs(["Text Input", "Voice Input"])
        
        with input_tab:
            # Text input
            if prompt := st.chat_input("Message Azazel..."):
                # Display the user's message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Save user message to Supabase
                save_message_to_supabase(
                    st.session_state.current_session,
                    "user",
                    prompt
                )
                
                # Store the message in session state for processing
                st.session_state.pending_user_message = prompt
                
                # Rerun to process the message
                st.rerun()
        
        with voice_tab:
            # This will show the audio recorder directly without needing a button click
            try:
                recording_result = audio_handler.record_audio(duration=5)
                
                # Check if we got a recording
                if recording_result and recording_result[0] is not None:
                    audio_path, _ = recording_result
                    
                    # Process the recording with a timeout
                    with st.spinner("Transcribing audio..."):
                        try:
                            # Set a timeout for transcription (30 seconds)
                            import threading
                            import time
                            
                            result = [None]
                            def transcribe_with_timeout():
                                result[0] = audio_handler.transcribe_audio(audio_path)
                            
                            thread = threading.Thread(target=transcribe_with_timeout)
                            thread.start()
                            
                            # Wait for up to 30 seconds
                            timeout = 30
                            start_time = time.time()
                            while thread.is_alive() and time.time() - start_time < timeout:
                                time.sleep(0.1)
                            
                            if thread.is_alive():
                                st.error(f"Transcription timed out after {timeout} seconds. Please try again.")
                                transcript = None
                            else:
                                transcript = result[0]
                        except Exception as e:
                            st.error(f"Error during transcription: {str(e)}")
                            transcript = None
                    
                    if transcript:
                        st.success(f"Transcribed: {transcript}")
                        
                        # Add a button to send the transcript
                        if st.button("Send this message"):
                            # Display the transcript to the user
                            with st.chat_message("user"):
                                st.write(transcript)
                                
                            # Save user message
                            save_message_to_supabase(
                                st.session_state.current_session,
                                "user",
                                transcript
                            )
                            
                            # Store the transcript in session state for processing
                            st.session_state.pending_user_message = transcript
                            
                            # Rerun to process the message
                            st.rerun()
            except Exception as e:
                st.error(f"Error with audio recording: {str(e)}")
                st.info("Please make sure you have the necessary permissions for microphone access.")

    # Modify the section where you handle the pending user message
    if "pending_user_message" in st.session_state:
        chat_history = get_chat_history_from_supabase(st.session_state.current_session)
        
        # Initialize sentiment analyzer if not already initialized
        if "sentiment_analyzer" not in st.session_state:
            from sentiment_analyzer import SentimentAnalyzer
            st.session_state.sentiment_analyzer = SentimentAnalyzer()
        
        # Get sentiment analysis for the current message
        sentiment_data = st.session_state.sentiment_analyzer.track_sentiment(
            st.session_state.current_session,
            st.session_state.pending_user_message
        )
        
        # Determine processing mode
        is_url = search_chain.is_url(st.session_state.pending_user_message) if search_chain else False
        is_programming = programming_assistant.is_programming_question(st.session_state.pending_user_message)
        
        # Extra check to prevent false positives for informational queries
        if is_programming:
            # Check for common informational query patterns that should not trigger programming mode
            query_lower = st.session_state.pending_user_message.lower()
            info_indicators = ["top 10", "richest", "people", "wealthiest", "billionaire", "list of", "show me"]
            
            # If query contains informational indicators and doesn't explicitly ask for code
            if any(indicator in query_lower for indicator in info_indicators) and not any(
                code_req in query_lower for code_req in ["write code", "python", "code to", "program", "function"]
            ):
                is_programming = False
                st.info("Query appears to be an informational request rather than programming.")
        
        # Check for document/image availability
        has_resources = rag_chain.has_documents_for_session(st.session_state.current_session)
        
        # Determine if RAG should be used - only if documents are available AND query is relevant
        should_use_rag = False
        if has_resources:
            # If force_rag is enabled, always use RAG
            if st.session_state.get("force_rag"):
                should_use_rag = True
            else:
                # Check if the query is relevant to the documents with higher threshold for relevance
                try:
                    should_use_rag = rag_chain.is_relevant_to_documents(
                        st.session_state.pending_user_message,
                        st.session_state.current_session
                    )
                except Exception as e:
                    st.warning(f"Error checking document relevance: {e}")
                    should_use_rag = False
        
        # Check if it needs a web search regardless of whether documents are available
        needs_search = False
        if search_chain:
            if is_url:
                needs_search = True
            elif st.session_state.get("force_search", False):
                # Force search if requested regardless of documents
                needs_search = True
            else:
                # Check if the query matches web search criteria
                needs_search = search_chain.needs_search(st.session_state.pending_user_message)
            
            # Important: If both RAG and web search are possible, prioritize web search
            # for queries that seem more suited for general web information
            if should_use_rag and needs_search:
                # Check if query contains search-specific keywords
                search_keywords = ["today", "latest", "news", "current", "recent", 
                                   "2023", "2024", "weather", "covid", "election", 
                                   "stock", "market", "price", "release", "update"]
                
                query_lower = st.session_state.pending_user_message.lower()
                has_search_keywords = any(kw in query_lower for kw in search_keywords)
                
                # If search keywords are present, prioritize web search over documents
                if has_search_keywords:
                    should_use_rag = False
                    st.info("Using web search for time-sensitive or external information request.")

        # Response generation logic
        try:
            response = None
            
            # First try direct model response if no special processing needed
            if not (should_use_rag or needs_search or is_url or is_programming):
                response = direct_openai_response(
                    st.session_state.pending_user_message,
                    st.session_state.current_session
                )
            # Then handle special cases
            elif is_url:
                if search_chain:
                    url = search_chain.extract_url(st.session_state.pending_user_message)
                    with st.spinner(f"Processing URL: {url}"):
                        url_response = search_chain.process_url(
                            url,
                            st.session_state.pending_user_message,
                            chat_history,
                            st.session_state.current_language
                        )
                        response = url_response["answer"]
            elif is_programming:
                # Pass the current personality to the programming assistant
                with st.spinner("Analyzing programming question..."):
                    programming_response = programming_assistant.answer_programming_question(
                        st.session_state.pending_user_message,
                        chat_history,
                        st.session_state.current_language,
                        personality=st.session_state.get(f"personality_{st.session_state.current_session}", "You are a helpful assistant.")
                    )
                    response = programming_response["answer"]
            elif should_use_rag:
                # Pass the current personality to the RAG chain
                with st.spinner("Analyzing documents..."):
                    rag_chain.personality = st.session_state.get(f"personality_{st.session_state.current_session}", "You are a helpful assistant.")
                    rag_response = rag_chain.answer_question(
                        st.session_state.pending_user_message,
                        st.session_state.current_session,
                        chat_history,
                        st.session_state.current_language
                    )
                    response = rag_response["answer"]
            elif needs_search:
                with st.spinner("Searching the web for information..."):
                    # Pass the current personality to the search chain
                    search_chain.personality = st.session_state.get(f"personality_{st.session_state.current_session}", "You are a helpful assistant.")
                    search_response = search_chain.search_with_web(
                        st.session_state.pending_user_message,
                        chat_history,
                        st.session_state.current_language
                    )
                    response = search_response["answer"]

            # If we got a response, display it
            if response:
                with st.chat_message("assistant"):
                    st.write(response)
                
                # Save assistant response
                save_message_to_supabase(
                    st.session_state.current_session,
                    "assistant",
                    response
                )
                
                # Update session metadata
                save_session_to_supabase(
                    st.session_state.current_session,
                    st.session_state.current_session_name,
                    st.session_state.current_language
                )
            
            # Clear the pending message
            del st.session_state.pending_user_message
            
            # Rerun to refresh the chat
            st.rerun()
        
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.info("Please try again with a different query.")
