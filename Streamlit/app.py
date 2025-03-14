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

# Create a function to get language instruction
def get_language_instruction(language):
    """Get a standardized language instruction for system prompts"""
    # Make instruction very clear and prominent for the model
    return f"CRITICAL PRIORITY INSTRUCTION: YOU MUST RESPOND ONLY IN {language} LANGUAGE."

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
        
        # Get the selected model from session state
        selected_model_name = st.session_state.get("selected_model", "GPT-3.5 Turbo")
        model_name = available_models[selected_model_name]
        
        # Log the model being used
        st.info(f"🤖 Using {selected_model_name} model")
        
        # Format chat history for the API
        formatted_messages = [
            {"role": "system", "content": current_personality + f"\n\nIMPORTANT: You MUST respond in {st.session_state.current_language} language ONLY, regardless of the language used in the question."}
        ]
        
        # Add recent chat history
        for msg in chat_history[-5:]:  # Last 5 messages
            formatted_messages.append({"role": msg["role"], "content": msg["message"]})
        
        # Add the current message
        formatted_messages.append({"role": "user", "content": question})
        
        # Always use streaming for better user experience
        response = client.chat.completions.create(
            model=model_name,
            messages=formatted_messages,
            temperature=0.7,
            stream=True,  # Always enable streaming
        )
        
        # Return the stream object
        return response
            
    except Exception as e:
        st.error(f"Error in direct OpenAI call: {str(e)}")
        return f"I'm having trouble generating a response right now. Error: {str(e)}"

# Create a function to handle streaming responses and display them
def display_streaming_response(response_stream):
    """Display streaming response from OpenAI API"""
    # Create an empty container for the streaming response
    placeholder = st.empty()
    full_response = ""
    
    # First show typing indicator
    with st.chat_message("assistant"):
        placeholder.markdown('<div class="typing-indicator"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
        
        # Sleep briefly to ensure the typing indicator is visible
        import time
        time.sleep(0.5)
        
        # Loop through the streaming response
        for chunk in response_stream:
            # Check if there is content in the chunk
            if hasattr(chunk.choices[0].delta, "content"):
                content = chunk.choices[0].delta.content
                if content is not None:
                    full_response += content
                    # Update the displayed message with the accumulated response
                    placeholder.markdown(full_response)
                    
                    # Small delay for very fast responses to appear more natural
                    if len(full_response) < 20:  # Only for the beginning
                        time.sleep(0.01)
    
    # Return the full response for storing in the database
    return full_response

# Create a function to generate streaming responses for various chains
def generate_streaming_response(chain_type, question, chat_history, language, personality=None, url=None):
    """Generate streaming response from different chains (RAG, Web Search, Programming)"""
    try:
        # Create an OpenAI client with streaming enabled
        client = OpenAI(api_key=st.session_state.openai_api_key)
        
        # Get the selected model
        selected_model_name = st.session_state.get("selected_model", "GPT-3.5 Turbo")
        model_name = available_models[selected_model_name]
        
        # Log which language is being used
        print(f"Generating response in language: {language}")
        
        # Create a placeholder for the streaming response
        placeholder = st.empty()
        full_response = ""
        
        # Get language instruction first - this will be prepended to all system prompts
        language_instruction = get_language_instruction(language)
        
        # Prepare the appropriate prompt based on chain type
        system_prompt = language_instruction + "\n\n" + (personality if personality else "You are a helpful assistant.")
        
        # Add specific instructions based on the chain type
        if chain_type == "rag":
            # Determine appropriate chunk size based on query complexity
            query_words = len(question.split())
            
            # For complex queries (longer questions), retrieve larger chunks for more context
            # For simple queries, retrieve smaller chunks for more precise answers
            if query_words > 20:  # Very complex question
                chunk_size = 2000  # Extra large chunk size
                k = 6  # Fewer chunks but larger ones
            elif query_words > 15:  # Complex question
                chunk_size = 1500  # Larger chunk size
                k = 8  # Standard number of chunks
            elif query_words > 8:  # Medium complexity
                chunk_size = 1000  # Default chunk size
                k = 10  # More chunks of standard size
            else:  # Simple question
                chunk_size = 800   # Smaller chunk size
                k = 12  # More chunks but smaller ones
            
            # Get relevant documents with dynamic retrieval parameters
            docs = rag_chain.get_relevant_documents(
                question, 
                st.session_state.current_session, 
                k=k
            )
            
            # Calculate total tokens to avoid exceeding context limits
            doc_content_list = []
            total_chars = 0
            max_chars = 12000  # Generous limit but still within token constraints
            
            # Format document chunks with priority to most relevant ones
            for i, doc in enumerate(docs):
                # Check if doc is a dictionary or an object and access page_content accordingly
                if isinstance(doc, dict) and "page_content" in doc:
                    content = doc["page_content"]
                elif hasattr(doc, "page_content"):
                    content = doc.page_content
                else:
                    # Skip this document if we can't determine its content format
                    continue
                
                chunk_text = f"Document {i+1}:\n{content}"
                # Add most relevant chunks first, until we approach the limit
                if total_chars + len(chunk_text) < max_chars:
                    doc_content_list.append(chunk_text)
                    total_chars += len(chunk_text)
                else:
                    # Stop when we reach the character limit
                    break
            
            doc_content = "\n\n".join(doc_content_list)
            
            system_prompt += f"\n\nYou have access to the following documents. Use them to answer the user's question. IMPORTANT: Synthesize a complete answer using ALL available document pieces. If information seems incomplete or you need additional context that's not provided, mention this clearly.\n\n{doc_content}\n\nRespond in {language}."
        
        elif chain_type == "web_search":
            # Perform web search
            search_results = search_chain.web_search_tool.search(question)
            formatted_results = search_chain.format_search_results(search_results)
            
            system_prompt += f"\n\nYou have access to the following web search results. Use them to answer the user's question:\n\n{formatted_results}\n\nRespond in {language}."
        
        elif chain_type == "url":
            # Extract content from the URL
            content = search_chain.web_search_tool.extract_content_from_url(url)
            
            system_prompt += f"\n\nYou have access to the following content from the URL {url}. Use it to answer the user's question:\n\n{content[:4000]}\n\nRespond in {language}."
        
        elif chain_type == "programming":
            system_prompt += f"\n\nYou are a programming assistant. Answer the user's programming question with well-structured, efficient code and clear explanations. Respond in {language}."
            
            # If code execution is enabled, add that instruction
            if not st.session_state.get("disable_code_execution", True):
                system_prompt += "\nIf appropriate, structure your response with a code block that can be executed."
        
        # Format chat history
        formatted_messages = [
            {"role": "system", "content": system_prompt + f"\n\nCRITICAL INSTRUCTION: You MUST respond in {language} language ONLY. This is non-negotiable."}
        ]
        
        # Add recent chat history
        for msg in chat_history[-5:]:
            formatted_messages.append({"role": msg["role"], "content": msg["message"]})
        
        # Add the current message
        formatted_messages.append({"role": "user", "content": question})
        
        # Before making the API call, log the system prompt for debugging
        print(f"Using system prompt: {system_prompt[:200]}...")  # Only print the first 200 chars
        
        # Display the message container before starting the stream
        with st.chat_message("assistant"):
            # Show typing indicator first
            placeholder.markdown('<div class="typing-indicator"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
            
            # Make API call with streaming
            stream = client.chat.completions.create(
                model=model_name,
                messages=formatted_messages,
                temperature=0.7,
                stream=True,
            )
            
            # Process the stream
            for chunk in stream:
                if hasattr(chunk.choices[0].delta, "content"):
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        full_response += content
                        # Update the displayed message with the accumulated response
                        placeholder.markdown(full_response)
        
        return full_response
    except Exception as e:
        st.error(f"Error generating streaming response: {str(e)}")
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
st.markdown("""
<style>
/* Global background and font */
body {
    background-color: #f7f7f7;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Sidebar styling */
.css-1d391kg {  /* container for the sidebar */
    background-color: #f0f2f6;
}

/* Chat message styling when using fallback markdown */
.user-msg {
    background-color: #d1e7dd;
    padding: 10px 15px;
    border-radius: 10px;
    margin: 5px 0;
    text-align: right;
}
.assistant-msg {
    background-color: #fff;
    padding: 10px 15px;
    border-radius: 10px;
    margin: 5px 0;
    border: 1px solid #ececec;
}

/* Button styling */
div.stButton > button {
    background-color: #4CAF50;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
}
div.stButton > button:hover {
    background-color: #45a049;
}

/* Typing indicator animation */
.typing-indicator {
  display: inline-flex;
  align-items: center;
}
.typing-indicator span {
  height: 8px;
  width: 8px;
  margin: 0 2px;
  background-color: #606060;
  border-radius: 50%;
  opacity: 0.4;
  animation: typing 1.4s infinite both;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0% { opacity: 0.4; transform: scale(1) }
  50% { opacity: 1; transform: scale(1.2) }
  100% { opacity: 0.4; transform: scale(1) }
}

/* Add a hidden div at the bottom to scroll to */
.scroll-target {
  height: 1px;
  width: 100%;
  margin-bottom: 300px; /* Extra space to ensure content isn't cut off */
}
</style>
""", unsafe_allow_html=True)

# Add global auto-scrolling JavaScript right after the CSS
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
    "GPT-4 Turbo": "gpt-4-turbo-preview",
    "GPT-4o Mini": "gpt-4o-mini"  # Replacing Vision with Mini
}

# Initialize selected_model in session state if not present
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "GPT-3.5 Turbo"

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
        
        # Add an auto-scroll trigger right after API key validation
        st.sidebar.markdown("""
        <script>
            // API key validation successful - trigger scrolling
            console.log("API key validated - triggering scrolls");
            setTimeout(() => {
                if (window.scrollSequence) {
                    window.scrollSequence();
                } else if (window.superForceScroll) {
                    window.superForceScroll();
                } else {
                    // Fallback if our global functions aren't available yet
                    window.scrollTo(0, document.body.scrollHeight);
                    
                    // Try to scroll any common Streamlit containers
                    document.querySelectorAll('.main, .stApp, .element-container, .block-container').forEach(el => {
                        if (el) el.scrollTop = el.scrollHeight;
                    });
                    
                    // Also try to find our scroll target
                    const target = document.getElementById('scroll-target');
                    if (target) target.scrollIntoView({behavior: 'auto', block: 'end'});
                }
            }, 500);
        </script>
        """, unsafe_allow_html=True)
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
                st.session_state[f"personality_{selected_session_id}"] = selected_session["personality"]
                
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
                # Default personality if none is set for this session
                st.session_state.current_personality = "You are a helpful, respectful, and honest assistant."
                st.session_state[f"personality_{selected_session_id}"] = "You are a helpful, respectful, and honest assistant."
                st.session_state.personality_option = "Default (Helpful Assistant)"
            
            # Reset personality to the one for the new session or to default
            if "personality" in selected_session and selected_session["personality"]:
                st.session_state.current_personality = selected_session["personality"]
                st.session_state[f"personality_{selected_session_id}"] = selected_session["personality"]
            
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
    
    # Display test message in each language
    language_test_messages = {
        "English": "Language set to English. All responses will be in English.",
        "Spanish": "Idioma configurado a Español. Todas las respuestas serán en Español.",
        "French": "Langue définie sur Français. Toutes les réponses seront en Français.",
        "German": "Sprache auf Deutsch eingestellt. Alle Antworten werden auf Deutsch sein.",
        "Hindi": "भाषा हिंदी पर सेट की गई है। सभी उत्तर हिंदी में होंगे।",
        "Chinese": "语言设置为中文。所有回复将使用中文。",
        "Japanese": "言語が日本語に設定されました。すべての応答は日本語になります。",
        "Russian": "Язык установлен на русский. Все ответы будут на русском языке.",
        "Arabic": "تم ضبط اللغة على العربية. ستكون جميع الردود باللغة العربية.",
        "Portuguese": "Idioma definido para Português. Todas as respostas serão em Português."
    }
    
    # If we have a predefined message for this language, show it
    if language in language_test_messages:
        success_message = language_test_messages[language]
    else:
        # For other languages, use a generic English message
        success_message = f"Now responding in {language}. All responses will be in {language}."
    
    # Update the language in the database
    save_session_to_supabase(
        st.session_state.current_session,
        st.session_state.current_session_name,
        language
    )
    
    st.sidebar.success(success_message)
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
st.title("🤖 Azazel")

# Add JavaScript for auto-scrolling BEFORE the API key check
# This ensures it's always loaded regardless of API key status
# Removed: JavaScript is now loaded globally near the top of the file

if not st.session_state.openai_api_key:
    st.error("Please enter your OpenAI API key in the sidebar to start chatting!")
else:
    st.subheader(st.session_state.current_session_name)
    
    # Get chat history
    chat_history = get_chat_history_from_supabase(st.session_state.current_session)
    
    # Create a container for chat messages
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["message"])
        
        # Override the scroll-target style to reduce the gap
        st.markdown("""
        <style>
        /* Override to reduce the gap */
        .scroll-target {
            margin-bottom: 50px !important; 
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Add a scroll target div at the bottom for auto-scrolling
        st.markdown('<div id="scroll-target" class="scroll-target"></div>', unsafe_allow_html=True)
        
        # Add an additional auto-scroll trigger specifically after displaying messages
        st.markdown("""
        <script>
            // Immediate scroll trigger after messages are displayed
            (function() {
                console.log("Messages displayed - triggering scroll");
                
                // Try scrolling immediately and with slight delays
                setTimeout(() => {
                    if (window.superForceScroll) {
                        window.superForceScroll();
                    } else {
                        window.scrollTo(0, document.body.scrollHeight);
                    }
                }, 100);
                
                setTimeout(() => {
                    if (window.superForceScroll) {
                        window.superForceScroll();
                    } else {
                        window.scrollTo(0, document.body.scrollHeight);
                    }
                }, 500);
            })();
        </script>
        """, unsafe_allow_html=True)
    
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
        
        # Log the decision for transparency
        if is_programming:
            st.info("💻 Detected as a programming question")
            
        # Check for document/image availability
        has_resources = rag_chain.has_documents_for_session(st.session_state.current_session)
        
        # Determine if RAG should be used - only if documents are available AND query is relevant
        should_use_rag = False
        has_images = rag_chain.has_images_for_session(st.session_state.current_session)
        has_documents = rag_chain.document_processor.has_documents(st.session_state.current_session)
        
        if has_resources:
            # If force_rag is enabled, always use RAG
            if st.session_state.get("force_rag"):
                should_use_rag = True
                st.info("📄 Using document and image knowledge (forced)")
            else:
                # Check if the query is relevant to the documents with higher threshold for relevance
                try:
                    should_use_rag = rag_chain.is_relevant_to_documents(
                        st.session_state.pending_user_message,
                        st.session_state.current_session
                    )
                    if should_use_rag:
                        st.info("📄 Using document knowledge (relevant query)")
                except Exception as e:
                    st.warning(f"Error checking document relevance: {e}")
                    should_use_rag = False
        
        # Check if it needs a web search regardless of whether documents are available
        needs_search = False
        if search_chain:
            if is_url:
                needs_search = True
                st.info("🔗 Processing URL")
            elif st.session_state.get("force_search", False):
                # Force search if requested regardless of documents
                needs_search = True
                should_use_rag = False  # Explicitly disable RAG when force_search is enabled
                st.info("🌐 Using web search (forced)")
            else:
                # Check if the query matches web search criteria using our improved method
                needs_search = search_chain.needs_search(st.session_state.pending_user_message)
                if needs_search:
                    st.info("🌐 Using web search for up-to-date information")
            
            # Important: If both RAG and web search are possible, make a smarter decision
            # Only do conflict resolution if force_search is not enabled
            if should_use_rag and needs_search and not st.session_state.get("force_search", False):
                # Get the query text and analyze it
                query_lower = st.session_state.pending_user_message.lower()
                
                # Time-sensitive keywords that should prioritize web search
                time_keywords = ["current", "latest", "recent", "today", "now", "this year", "this month"]
                has_time_keywords = any(kw in query_lower for kw in time_keywords)
                
                # Keywords for things that change frequently
                dynamic_data_keywords = ["price", "stock", "value", "rank", "rating", "richest", "population"]
                has_dynamic_keywords = any(kw in query_lower for kw in dynamic_data_keywords)
                
                # If the query is time-sensitive or about dynamic data, prioritize web search
                if has_time_keywords or has_dynamic_keywords:
                    should_use_rag = False
                    st.info("🌐 Prioritizing web search for time-sensitive information")
                # Otherwise, prefer documents (they're more reliable than web search)
                else:
                    needs_search = False
                    st.info("📄 Prioritizing document knowledge over web search")

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
                        # Use streaming for URL processing
                        personality = st.session_state.get(f"personality_{st.session_state.current_session}", "You are a helpful assistant.")
                        response = generate_streaming_response(
                            "url",
                            st.session_state.pending_user_message,
                            chat_history,
                            st.session_state.current_language,
                            personality=personality,
                            url=url
                        )
            elif is_programming:
                # Pass the current personality to the programming assistant
                with st.spinner("Analyzing programming question..."):
                    # Use streaming response for programming
                    personality = st.session_state.get(f"personality_{st.session_state.current_session}", "You are a helpful assistant.")
                    response = generate_streaming_response(
                        "programming",
                        st.session_state.pending_user_message,
                        chat_history,
                        st.session_state.current_language,
                        personality=personality
                    )
            # Check for web search needs first (prioritize web search over RAG)
            elif needs_search:
                # Handle web search logic
                if search_chain:
                    with st.spinner("Searching the web for information..."):
                        # Use streaming response for web search
                        personality = st.session_state.get(f"personality_{st.session_state.current_session}", "You are a helpful assistant.")
                        response = generate_streaming_response(
                            "web_search",
                            st.session_state.pending_user_message,
                            chat_history,
                            st.session_state.current_language,
                            personality=personality
                        )
                else:
                    st.warning("Web search disabled. Please set up web search in the sidebar.")
                    response = direct_openai_response(
                        st.session_state.pending_user_message,
                        st.session_state.current_session
                    )
            elif should_use_rag:
                # Pass the current personality to the RAG chain
                with st.spinner("Analyzing documents and images..."):
                    try:
                        # Use streaming response for RAG
                        personality = st.session_state.get(f"personality_{st.session_state.current_session}", "You are a helpful assistant.")
                        
                        # First check if there are images for this session
                        has_images = rag_chain.has_images_for_session(st.session_state.current_session)
                        has_documents = rag_chain.document_processor.has_documents(st.session_state.current_session)
                        
                        # Log resource status for debugging
                        if has_images:
                            print(f"Session {st.session_state.current_session} has uploaded images available")
                        if has_documents:
                                print(f"Session {st.session_state.current_session} has documents available")
                        
                        # Analyze query complexity to determine appropriate chunk size for text docs
                        query_words = len(st.session_state.pending_user_message.split())
                        if query_words > 20:  # Very complex question
                            chunk_size = 2000  # Extra large chunk size
                            k = 6  # Fewer chunks but larger ones
                        elif query_words > 15:  # Complex question
                            chunk_size = 1500  # Larger chunk size
                            k = 8  # Standard number of chunks
                        elif query_words > 8:  # Medium complexity
                            chunk_size = 1000  # Default chunk size
                            k = 10  # More chunks of standard size
                        else:  # Simple question
                            chunk_size = 800   # Smaller chunk size
                            k = 12  # More chunks but smaller ones
                        
                        # Check for image-related queries
                        image_keywords = [
                            # Basic image terms
                            "image", "picture", "photo", "pic", "snapshot", "photograph", "shot", "visual",
                            
                            # Visual elements
                            "wearing", "clothes", "dress", "outfit", "attire", "shirt", "pants", "shoes", "hat", "appearance", 
                            "color", "background", "foreground", "scene", "setting", "logo", "pattern", "design",
                            
                            # People and objects
                            "person", "man", "woman", "boy", "girl", "child", "people", "guy", "lady", "object", "item",
                            "figure", "character", "animal", "pet", "building", "structure", "tree", "plant", "car", "vehicle",
                            
                            # Visual actions
                            "show", "display", "visible", "see", "look", "appears", "looks like", "shown", "depicted",
                            
                            # Questions about images
                            "what is in the", "who is in", "what does it show", "describe", "explain the image",
                            "what can you see", "tell me about the", "what's in the", "analyze this", "whats in this"
                        ]
                        
                        query_lower = st.session_state.pending_user_message.lower()
                        
                        # Detect if the query is explicitly about images
                        is_explicit_image_query = any(keyword in query_lower for keyword in image_keywords)
                        
                        # Detect if it's a general description request (for queries like "describe the image" without specific keywords)
                        is_general_description = has_images and (
                            query_lower.startswith("describe") or 
                            query_lower.startswith("tell me about") or 
                            query_lower.startswith("what") or 
                            query_lower.startswith("who") or
                            query_lower.startswith("how") or
                            "given image" in query_lower or
                            "this image" in query_lower or
                            "the image" in query_lower
                        )
                        
                        # Combine both detection methods
                        is_image_query = has_images and (is_explicit_image_query or is_general_description)
                        
                        # If we have images but no text documents, always treat as image query
                        if has_images and not has_documents:
                            is_image_query = True
                            print("No text documents but images available - treating as image query")
                        
                        # Set default response to None
                        response = None
                        
                        # Process based on resources and query type
                        if is_image_query:
                            # Prioritize image analysis for image-related queries
                            st.info("Analyzing your images to answer the question...")
                            
                            # Get image analysis directly
                            image_analysis = rag_chain.document_processor.vision_processor.analyze_images(
                                st.session_state.current_session, 
                                st.session_state.pending_user_message
                            )
                            
                            if image_analysis:
                                # Get language instruction and create system prompt
                                language_instruction = get_language_instruction(st.session_state.current_language)
                                # Create a system prompt for this analysis
                                system_prompt = language_instruction + "\n\n" + (personality if personality else "You are a helpful assistant.")
                                
                                # Add image analysis to the prompt
                                doc_content = f"Image Analysis:\n{image_analysis[0]['content']}"
                                
                                # If text documents are available, also add relevant document content
                                if has_documents:
                                    try:
                                        # Get relevant documents
                                        docs = rag_chain.get_relevant_documents(
                                            st.session_state.pending_user_message,
                                            st.session_state.current_session,
                                            k=5
                                        )
                                        
                                        # Add document content if available
                                        if docs and len(docs) > 0:
                                            doc_texts = []
                                            for i, doc in enumerate(docs):
                                                # Safely access document content
                                                if isinstance(doc, dict) and "page_content" in doc:
                                                    content = doc["page_content"]
                                                elif hasattr(doc, "page_content"):
                                                    content = doc.page_content
                                                else:
                                                    continue
                                                
                                                doc_texts.append(f"Document {i+1}:\n{content}")
                                            
                                            if doc_texts:
                                                doc_content += "\n\nDocument Analysis:\n" + "\n\n".join(doc_texts)
                                                st.info("Analyzing both images and relevant documents together")
                                    except Exception as doc_err:
                                        st.warning(f"Could not retrieve text documents: {doc_err}")
                                
                                # Prepare the system prompt with image analysis
                                system_prompt += f"\n\nYou have access to the following content. Use it to answer the user's question:\n\n{doc_content}"
                                
                                # Define special instructions for image analysis
                                special_instructions = "\n\nIMPORTANT: You are capable of analyzing and describing images AND relevant documents. THE USER HAS UPLOADED CONTENT, AND YOU MUST REFER TO IT IN YOUR RESPONSE. Analyze both images and documents when answering the query. If the query relates images to documents (like 'is this image suitable for my resume'), be sure to address both aspects."
                                
                                # Add this instruction to system prompt
                                system_prompt += special_instructions
                                
                                # Pass the modified system prompt directly to OpenAI
                                client = OpenAI(api_key=st.session_state.openai_api_key)
                                
                                # Get the selected model
                                selected_model_name = st.session_state.get("selected_model", "GPT-3.5 Turbo")
                                model_name = available_models[selected_model_name]
                                
                                # Format messages for API
                                formatted_messages = [
                                    {"role": "system", "content": system_prompt}
                                ]
                                
                                # Add recent chat history
                                for msg in chat_history[-5:]:
                                    formatted_messages.append({"role": msg["role"], "content": msg["message"]})
                                
                                # Add the current message
                                formatted_messages.append({"role": "user", "content": st.session_state.pending_user_message})
                                
                                # Make the API call with streaming
                                with st.chat_message("assistant"):
                                    placeholder = st.empty()
                                    placeholder.markdown('<div class="typing-indicator"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
                                    
                                    stream = client.chat.completions.create(
                                        model=model_name,
                                        messages=formatted_messages,
                                        temperature=0.7,
                                        stream=True,
                                    )
                                    
                                    # Process the stream
                                    full_response = ""
                                    for chunk in stream:
                                        if hasattr(chunk.choices[0].delta, "content"):
                                            content = chunk.choices[0].delta.content
                                            if content is not None:
                                                full_response += content
                                                # Update the displayed message with the accumulated response
                                                placeholder.markdown(full_response)
                                    
                                    # Use the full response
                                    response = full_response
                                
                                # If image analysis failed
                                if not image_analysis:
                                    # Fall through to general knowledge if image analysis fails too
                                    response = None
                            else:
                                st.warning("Unable to analyze the image. Using general knowledge instead.")
                                response = direct_openai_response(
                                    st.session_state.pending_user_message,
                                    st.session_state.current_session
                                )
                        elif has_documents:
                            # Try document retrieval for non-image queries or when documents are available
                            docs = None
                            try:
                                docs = rag_chain.get_relevant_documents(
                                    st.session_state.pending_user_message, 
                                    st.session_state.current_session, 
                                    k=k
                                )
                            except Exception as doc_err:
                                st.error(f"Error retrieving documents: {doc_err}")
                                docs = None
                            
                            # Check if we got relevant documents
                            if docs and len(docs) > 0:
                                try:
                                    # Calculate total tokens to avoid exceeding context limits
                                    doc_content_list = []
                                    total_chars = 0
                                    max_chars = 12000  # Generous limit but still within token constraints
                                    
                                    # Format document chunks with priority to most relevant ones
                                    for i, doc in enumerate(docs):
                                        # Safely extract content from document
                                        try:
                                            # Check if doc is a dictionary or an object and access page_content accordingly
                                            if isinstance(doc, dict) and "page_content" in doc:
                                                content = doc["page_content"]
                                            elif hasattr(doc, "page_content"):
                                                content = doc.page_content
                                            else:
                                                # Skip this document if we can't determine its content format
                                                continue
                                            
                                            chunk_text = f"Document {i+1}:\n{content}"
                                            # Add most relevant chunks first, until we approach the limit
                                            if total_chars + len(chunk_text) < max_chars:
                                                doc_content_list.append(chunk_text)
                                                total_chars += len(chunk_text)
                                            else:
                                                # Stop when we reach the character limit
                                                break
                                        except Exception as extract_err:
                                            print(f"Error processing document chunk {i}: {extract_err}")
                                            continue
                                    
                                    if doc_content_list:
                                        doc_content = "\n\n".join(doc_content_list)
                                        # Initialize system_prompt here with language_instruction
                                        language_instruction = get_language_instruction(st.session_state.current_language)
                                        system_prompt = language_instruction + "\n\n" + (personality if personality else "You are a helpful assistant.")
                                        system_prompt += f"\n\nYou have access to the following documents. Use them to answer the user's question. IMPORTANT: Synthesize a complete answer using ALL available document pieces. If information seems incomplete or you need additional context that's not provided, mention this clearly.\n\n{doc_content}"
                                        
                                        response = generate_streaming_response(
                                            "rag",
                                            st.session_state.pending_user_message,
                                            chat_history,
                                            st.session_state.current_language,
                                            personality=personality
                                        )
                                    else:
                                        # Fall through to next section if no document contents could be processed
                                        docs = None
                                except Exception as process_err:
                                    st.error(f"Error processing document content: {process_err}")
                                    docs = None
                            
                            # If document retrieval failed but images are available, try image analysis as fallback
                            if (not docs or len(docs) == 0) and has_images:
                                st.info("No relevant text documents found, but analyzing available images...")
                                image_analysis = rag_chain.document_processor.vision_processor.analyze_images(
                                    st.session_state.current_session, 
                                    st.session_state.pending_user_message
                                )
                                
                                if image_analysis:
                                    # Create a system prompt if not already defined
                                    if 'system_prompt' not in locals():
                                        # Get language instruction
                                        language_instruction = get_language_instruction(st.session_state.current_language)
                                        system_prompt = language_instruction + "\n\n" + (personality if personality else "You are a helpful assistant.")
                                    
                                    # Format the image analysis as a document
                                    doc_content = f"Image Analysis:\n{image_analysis[0]['content']}"
                                    
                                    # If text documents are available, also add relevant document content
                                    if has_documents:
                                        try:
                                            # Get relevant documents
                                            docs = rag_chain.get_relevant_documents(
                                                st.session_state.pending_user_message,
                                                st.session_state.current_session,
                                                k=5
                                            )
                                            
                                            # Add document content if available
                                            if docs and len(docs) > 0:
                                                doc_texts = []
                                                for i, doc in enumerate(docs):
                                                    # Safely access document content
                                                    if isinstance(doc, dict) and "page_content" in doc:
                                                        content = doc["page_content"]
                                                    elif hasattr(doc, "page_content"):
                                                        content = doc.page_content
                                                    else:
                                                        continue
                                                    
                                                    doc_texts.append(f"Document {i+1}:\n{content}")
                                                
                                                if doc_texts:
                                                    doc_content += "\n\nDocument Analysis:\n" + "\n\n".join(doc_texts)
                                                    st.info("Analyzing both images and relevant documents together")
                                        except Exception as doc_err:
                                            st.warning(f"Could not retrieve text documents: {doc_err}")
                                    
                                    # Prepare the system prompt with content
                                    system_prompt += f"\n\nYou have access to the following content. Use it to answer the user's question:\n\n{doc_content}"
                                    
                                    # Define special instructions
                                    special_instructions = "\n\nIMPORTANT: You are capable of analyzing and describing images AND relevant documents. THE USER HAS UPLOADED CONTENT, AND YOU MUST REFER TO IT IN YOUR RESPONSE. Analyze both images and documents when answering the query. If the query relates images to documents (like 'is this image suitable for my resume'), be sure to address both aspects."
                                    
                                    # Add this instruction to system prompt
                                    system_prompt += special_instructions
                                
                                # Pass the modified system prompt directly to OpenAI
                                client = OpenAI(api_key=st.session_state.openai_api_key)
                                
                                # Get the selected model
                                selected_model_name = st.session_state.get("selected_model", "GPT-3.5 Turbo")
                                model_name = available_models[selected_model_name]
                                
                                # Format messages for API
                                formatted_messages = [
                                    {"role": "system", "content": system_prompt}
                                ]
                                
                                # Add recent chat history
                                for msg in chat_history[-5:]:
                                    formatted_messages.append({"role": msg["role"], "content": msg["message"]})
                                
                                # Add the current message
                                formatted_messages.append({"role": "user", "content": st.session_state.pending_user_message})
                                
                                # Make the API call with streaming
                                with st.chat_message("assistant"):
                                    placeholder = st.empty()
                                    placeholder.markdown('<div class="typing-indicator"><span></span><span></span><span></span></div>', unsafe_allow_html=True)
                                    
                                    stream = client.chat.completions.create(
                                        model=model_name,
                                        messages=formatted_messages,
                                        temperature=0.7,
                                        stream=True,
                                    )
                                    
                                    # Process the stream
                                    full_response = ""
                                    for chunk in stream:
                                        if hasattr(chunk.choices[0].delta, "content"):
                                            content = chunk.choices[0].delta.content
                                            if content is not None:
                                                full_response += content
                                                # Update the displayed message with the accumulated response
                                                placeholder.markdown(full_response)
                                    
                                    # Use the full response
                                    response = full_response
                                
                                # If image analysis failed
                                if not image_analysis:
                                    # Fall through to general knowledge if image analysis fails too
                                    response = None
                        
                        # If no response has been generated, use general knowledge
                        if response is None:
                            st.warning("No relevant documents or images found for your query. Using general knowledge instead.")
                            response = direct_openai_response(
                                st.session_state.pending_user_message,
                                st.session_state.current_session
                            )
                    
                    except Exception as e:
                        st.error(f"Error in document and image analysis: {str(e)}")
                        # Fall back to direct response
                        response = direct_openai_response(
                            st.session_state.pending_user_message,
                            st.session_state.current_session
                        )
                    
                    # Note: Web search logic has been moved higher in the priority order
                    # to ensure Force Web Search always takes precedence over RAG

            # If we got a response, display it
            if response:
                try:
                    # Check if it's a streaming response object or a string
                    if hasattr(response, "__iter__") and not isinstance(response, str):
                        # Handle streaming response
                        full_response = display_streaming_response(response)
                        response = full_response  # Update response with the full text for DB
                    else:
                        # If we got a string response (error message, etc.), display it directly
                        with st.chat_message("assistant"):
                            st.write(response)
                except Exception as e:
                    st.error(f"Error displaying response: {str(e)}")
                    response = f"I encountered an error while generating a response. Please try again."
                    with st.chat_message("assistant"):
                        st.write(response)
                
                # Save assistant response to database (only if it's a valid string)
                if isinstance(response, str):
                    save_message_to_supabase(
                        st.session_state.current_session,
                        "assistant",
                        response
                    )
                else:
                    save_message_to_supabase(
                        st.session_state.current_session,
                        "assistant",
                        "Error generating response. Please try again."
                    )
                
                # Update session metadata (restore this)
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
