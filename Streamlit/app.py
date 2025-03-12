import sys
import os
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
import json
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
import docx
import tempfile

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

# Update the direct_openai_response function to include chat history
def direct_openai_response(message, session_id=None):
    """Get a direct response from OpenAI API with chat history"""
    try:
        from openai import OpenAI
        import os
        
        # Get API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            api_key = st.session_state.openai_api_key
            
        if not api_key:
            st.error("OpenAI API key not found")
            return None
        
        # Get the current language and model from session state
        language = st.session_state.get("current_language", "English")
        model_name = available_models[st.session_state.selected_model]
        
        # Initialize client
        client = OpenAI(api_key=api_key)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": f"You are a helpful assistant. Please respond in {language}."}
        ]
        
        # Add chat history if session_id is provided
        if session_id:
            chat_history = get_chat_history_from_supabase(session_id)
            # Add last 5 messages for context
            for msg in chat_history[-5:]:
                messages.append({"role": msg["role"], "content": msg["message"]})
        
        # Add the current message
        messages.append({"role": "user", "content": message})
        
        # Make API call with selected model
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
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
    st.sidebar.success("Web search capability enabled!")
    if st.sidebar.button("Test Google Search API"):
        with st.sidebar:
            with st.spinner("Testing Google Search API..."):
                try:
                    test_results = search_chain.web_search_tool.google_search("test query")
                    if test_results and not test_results[0]["title"].startswith("Search Error"):
                        st.success("Google Search API is working correctly!")
                    else:
                        st.error(f"Google Search API test failed: {test_results[0]['snippet']}")
                except Exception as e:
                    st.error(f"Google Search API test failed: {str(e)}")
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
</style>
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
def save_session_to_supabase(session_id, session_name, language="English"):
    """Save session metadata to Supabase sessions table"""
    try:
        # Check if session exists
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
if "current_session" not in st.session_state:
    existing_sessions = get_all_sessions()
    
    if existing_sessions:
        # Use the most recent session
        st.session_state.current_session = existing_sessions[0]["session_id"]
        st.session_state.current_session_name = existing_sessions[0]["name"]
        st.session_state.current_language = existing_sessions[0]["language"]
    else:
        # Create a new session
        new_session_id = str(uuid.uuid4())
        st.session_state.current_session = new_session_id
        st.session_state.current_session_name = "New Chat"
        st.session_state.current_language = "English"
        save_session_to_supabase(new_session_id, "New Chat")

if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = None

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
        new_session_id = str(uuid.uuid4())
        st.session_state.current_session = new_session_id
        st.session_state.current_session_name = "New Chat"
        st.session_state.current_language = "English"
        save_session_to_supabase(new_session_id, "New Chat")
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
        st.session_state.current_session = selected_session_id
        selected_session = next((s for s in existing_sessions if s["session_id"] == selected_session_id), None)
        if selected_session:
            st.session_state.current_session_name = selected_session.get("name", "Untitled Chat")
            st.session_state.current_language = selected_session.get("language", "English")
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
            
            for session_display in sessions_to_delete:
                # Look up the full session object using our map
                if session_display in session_display_map:
                    session = session_display_map[session_display]
                    session_id = session["session_id"]
                    
                    # If deleting current session, flag for rerun and session change
                    if session_id == st.session_state.current_session:
                        need_rerun = True
                    
                    # Delete the session
                    if delete_session(session_id):
                        st.sidebar.success(f"Deleted: {session.get('name', 'Untitled')}")
                        deleted_any = True
            
            if deleted_any:
                if need_rerun:
                    # Get remaining sessions to switch to
                    remaining_sessions = get_all_sessions()
                    if remaining_sessions:
                        st.session_state.current_session = remaining_sessions[0]["session_id"]
                        st.session_state.current_session_name = remaining_sessions[0].get("name", "Untitled Chat")
                        st.session_state.current_language = remaining_sessions[0].get("language", "English")
                st.rerun()

# Delete Current Chat button (ensures at least one session remains)
if st.sidebar.button("🗑️ Delete Current Chat"):
    if len(existing_sessions) > 1:
        if delete_session(st.session_state.current_session):
            # Set current session to the next available session
            remaining_sessions = [s for s in existing_sessions if s["session_id"] != st.session_state.current_session]
            if remaining_sessions:
                st.session_state.current_session = remaining_sessions[0]["session_id"]
                st.session_state.current_session_name = remaining_sessions[0].get("name", "Untitled Chat")
                st.session_state.current_language = remaining_sessions[0].get("language", "English")
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
    
    if st.button("Clear All Data"):
        rag_chain.document_processor.vision_processor.image_store.pop(st.session_state.current_session, None)
        if rag_chain.clear_documents(st.session_state.current_session):
            st.success("Cleared all documents and images")

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

# Main Chat Interface
st.title("🤖 PolyBot")

if not st.session_state.openai_api_key:
    st.error("Please enter your OpenAI API key in the sidebar to start chatting!")
else:
    st.subheader(st.session_state.current_session_name)
    
    # Get chat history
    chat_history = get_chat_history_from_supabase(st.session_state.current_session)
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["message"])
    
    # Check if we should use RAG (if documents have been uploaded)
    use_rag = st.session_state.current_session in rag_chain.document_processor.vectorstores
    
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
            if prompt := st.chat_input("Message PolyBot..."):
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
        
        # Determine processing mode
        is_url = search_chain.is_url(st.session_state.pending_user_message) if search_chain else False
        is_programming = programming_assistant.is_programming_question(st.session_state.pending_user_message)
        
        # Check for document/image availability
        has_resources = rag_chain.has_documents_for_session(st.session_state.current_session)
        
        # Determine if RAG should be used - only if documents are available AND query is relevant
        should_use_rag = False
        if has_resources:
            # If force_rag is enabled, always use RAG
            if st.session_state.get("force_rag"):
                should_use_rag = True
            else:
                # Check if the query is relevant to the documents
                should_use_rag = rag_chain.is_relevant_to_documents(
                    st.session_state.pending_user_message, 
                    st.session_state.current_session
                )
        
        # Check if it needs a web search (only if not using RAG or URL)
        needs_search = False
        if search_chain and not is_url and not should_use_rag:
            needs_search = search_chain.needs_search(st.session_state.pending_user_message)
            
            # Force search if requested
            if st.session_state.get("force_search", False):
                needs_search = True

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
                    st.info(f"Processing URL: {url}")
                    url_response = search_chain.process_url(
                        url,
                        st.session_state.pending_user_message,
                        chat_history,
                        st.session_state.current_language
                    )
                    response = url_response["answer"]
            elif is_programming:
                programming_response = programming_assistant.answer_programming_question(
                    st.session_state.pending_user_message,
                    chat_history,
                    st.session_state.current_language
                )
                response = programming_response["answer"]
            elif should_use_rag:
                rag_response = rag_chain.answer_question(
                    st.session_state.pending_user_message,
                    st.session_state.current_session,
                    chat_history,
                    st.session_state.current_language
                )
                response = rag_response["answer"]
            elif needs_search:
                st.info("Searching the web for information...")
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