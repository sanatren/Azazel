# Azazel AI - FastAPI Backend

Modern FastAPI backend for the Azazel AI Assistant, providing RESTful APIs for chat, document processing, and web search.

## Architecture

```
api/
├── main.py              # FastAPI application entry point
├── core/               # Core configuration
│   ├── config.py       # Settings and environment variables
│   └── __init__.py
├── models/             # Pydantic schemas
│   ├── schemas.py      # Request/response models
│   └── __init__.py
├── routes/             # API endpoints
│   ├── chat.py         # Chat and code execution
│   ├── documents.py    # Document processing and RAG
│   ├── search.py       # Web search
│   ├── sessions.py     # Session management
│   └── __init__.py
└── requirements.txt    # Python dependencies
```

## Features

- ✅ RESTful APIs for all Azazel functions
- ✅ Streaming chat responses
- ✅ Document upload and processing
- ✅ RAG (Retrieval-Augmented Generation)
- ✅ Web search integration
- ✅ Code execution
- ✅ Session management
- ✅ CORS support
- ✅ Docker support

## Quick Start

### Local Development

1. Install dependencies:
```bash
cd api
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```env
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_cse_id
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

3. Run the server:
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

4. Access API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Docker Deployment

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

This will start both:
- FastAPI backend: http://localhost:8000
- Streamlit frontend: http://localhost:8501

## API Endpoints

### Chat Operations

#### Send Message
```http
POST /api/chat/message
Content-Type: application/json

{
  "message": "Hello!",
  "session_id": "session-123",
  "api_key": "sk-...",
  "language": "English",
  "personality": "You are a helpful assistant."
}
```

#### Stream Response
```http
POST /api/chat/stream
Content-Type: application/json

{
  "message": "Tell me a story",
  "session_id": "session-123",
  "api_key": "sk-...",
  "language": "English",
  "chat_history": []
}
```

#### Execute Code
```http
POST /api/chat/code
Content-Type: application/json

{
  "question": "Write a function to calculate fibonacci",
  "session_id": "session-123",
  "api_key": "sk-...",
  "language": "English"
}
```

### Document Operations

#### Upload Document
```http
POST /api/documents/upload
Content-Type: multipart/form-data

file: [file]
session_id: session-123
api_key: sk-...
```

#### Query Documents
```http
POST /api/documents/query
Content-Type: application/json

{
  "query": "What is this document about?",
  "session_id": "session-123",
  "api_key": "sk-...",
  "k": 8
}
```

#### Clear Documents
```http
POST /api/documents/clear
Content-Type: application/json

{
  "session_id": "session-123"
}
```

### Search Operations

#### Web Search
```http
POST /api/search/web
Content-Type: application/json

{
  "query": "Latest news about AI",
  "session_id": "session-123",
  "api_key": "sk-...",
  "language": "English",
  "chat_history": []
}
```

### Session Management

#### Create Session
```http
POST /api/sessions/create
Content-Type: application/json

{
  "user_id": "user-123"
}
```

#### Get History
```http
GET /api/sessions/history/{session_id}?limit=50
```

## Deployment

### Deploy to Cloud

#### Option 1: Railway
```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
cd api
railway up
```

#### Option 2: Render
1. Create a new Web Service
2. Connect your GitHub repository
3. Build Command: `pip install -r api/requirements.txt`
4. Start Command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

#### Option 3: Google Cloud Run
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/azazel-api api/

# Deploy
gcloud run deploy azazel-api \
  --image gcr.io/PROJECT_ID/azazel-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Environment Variables

Required for deployment:
- `GOOGLE_API_KEY` - Google API key for search
- `GOOGLE_CSE_ID` - Custom Search Engine ID
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_KEY` - Supabase API key

## Performance Optimizations

1. **Connection Pooling**: Reuses AI model instances per API key
2. **Streaming**: Supports server-sent events for real-time responses
3. **Caching**: In-memory caching of search chains and RAG instances
4. **Async**: Full async/await support for concurrent requests

## Monitoring

Access metrics at:
- Health check: `GET /health`
- API docs: `GET /docs`

## Security

- API keys are passed per request (no storage)
- CORS configured for specific origins
- Input validation with Pydantic
- Error handling with proper HTTP status codes

## License

MIT